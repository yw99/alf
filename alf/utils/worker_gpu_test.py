# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for explicit single-host GPU mapping and process ownership.

Set ALF_TEST_SHARED_GPU=1 to exercise four-rank CUDA DDP on three GPUs.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from alf.utils.worker_gpu import resolve_worker_gpus, run_workers

_IMPORT_VISIBILITY = os.environ.get('CUDA_VISIBLE_DEVICES')


def _record_worker(rank, world_size, directory):
    Path(directory, str(rank)).write_text(json.dumps({
        'rank': rank, 'world_size': world_size, 'pid': os.getpid(),
        'pgid': os.getpgrp(), 'import_visibility': _IMPORT_VISIBILITY,
        'visibility': os.environ.get('CUDA_VISIBLE_DEVICES')}))


def _with_descendant(rank, world_size, directory, fail, queue):
    queue.put(rank)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    Path(directory, str(rank)).write_text(json.dumps([os.getpid(), child.pid]))
    if fail and rank == 0:
        deadline = time.monotonic() + 20
        while not Path(directory, '1').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('sibling did not start')
            time.sleep(.05)
        raise SystemExit(3)
    time.sleep(120)


def _gradient_worker(rank, world_size, directory):
    import datetime
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    assert torch.cuda.device_count() == 1
    assert _IMPORT_VISIBILITY == os.environ['CUDA_VISIBLE_DEVICES']
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=30))
    try:
        value = torch.tensor(float(rank + 1), device='cuda')
        dist.all_reduce(value)
        assert value.item() == 10
        model = torch.nn.Linear(1, 1, bias=False, device='cuda')
        with torch.no_grad():
            model.weight.fill_(1)
        model = DDP(model, device_ids=None)
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        for _ in range(10):
            opt.zero_grad()
            model(torch.tensor([[float(rank + 1)]], device='cuda')).sum().backward()
            assert torch.allclose(model.module.weight.grad,
                                  torch.full_like(model.module.weight, 2.5))
            opt.step()
        assert torch.allclose(model.module.weight,
                              torch.full_like(model.module.weight, -1.5))
        _record_worker(rank, world_size, directory)
    finally:
        dist.destroy_process_group()


def _alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class WorkerGpuTest(unittest.TestCase):
    def test_resolve(self):
        self.assertEqual(resolve_worker_gpus('0,1,2,0', '2,0,1', 3),
                         ['2', '0', '1', '2'])
        self.assertEqual(resolve_worker_gpus('1,0,1', 'GPU-a,GPU-b', 2),
                         ['GPU-b', 'GPU-a', 'GPU-b'])
        self.assertEqual(resolve_worker_gpus('0,1,2,0', None, 3),
                         ['0', '1', '2', '0'])
        self.assertEqual(resolve_worker_gpus('0', None, 1), ['0'])

    def test_invalid(self):
        for mapping in ('', ' ', '-1', '0,', ',0', '0,,1', 'a', '1.0', '3'):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                resolve_worker_gpus(mapping, '0,1,2', 3)
        for visibility, count in [('', 0), ('0,1,2,3', 3), ('0,0', 2)]:
            with self.subTest(visibility=visibility), self.assertRaises(ValueError):
                resolve_worker_gpus('0', visibility, count)
        for mode, ngpu in [('none', 1), ('multi-node-multi-gpu', 1), ('multi-gpu', 2)]:
            with self.subTest(mode=mode, ngpu=ngpu), self.assertRaises(ValueError):
                resolve_worker_gpus('0', None, 3, mode, ngpu)

    def test_legacy_is_not_reinterpreted(self):
        for count, mode, ngpu in [(4, 'multi-gpu', 1), (4, 'multi-gpu', 2),
                                  (3, 'multi-gpu', 1), (0, 'none', 1)]:
            self.assertIsNone(resolve_worker_gpus(None, None, count, mode, ngpu))

    def test_spawn_visibility_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='original'):
                run_workers(['2', '0'], _record_worker, (directory,))
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'original')
            for rank, gpu in enumerate(['2', '0']):
                record = json.loads(Path(directory, str(rank)).read_text())
                self.assertEqual(record['rank'], rank)
                self.assertEqual(record['world_size'], 2)
                self.assertEqual(record['visibility'], gpu)
                self.assertEqual(record['import_visibility'], gpu)
                self.assertNotEqual(record['pid'], os.getpid())
                self.assertEqual(record['pid'], record['pgid'])
                self.assertFalse(_alive(record['pid']))

    def _check_cleanup(self, terminate):
        with tempfile.TemporaryDirectory() as directory:
            code = ('from alf.utils.worker_gpu import run_workers; '
                    'from alf.utils.worker_gpu_test import _with_descendant; '
                    f'run_workers(["0", "1"], _with_descendant, '
                    f'({directory!r}, {not terminate!r}), with_queue=True)')
            coordinator = subprocess.Popen([sys.executable, '-c', code],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 40
                while len(list(Path(directory).iterdir())) < 2:
                    self.assertIsNone(coordinator.poll())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.1)
                rows = subprocess.check_output(
                    ['ps', '-eo', 'pid=,ppid='], text=True).splitlines()
                parents = {int(p): int(pp) for p, pp in (row.split() for row in rows)}
                owned = {coordinator.pid}
                while True:
                    children = {p for p, pp in parents.items() if pp in owned}
                    if children <= owned:
                        break
                    owned.update(children)
                if terminate:
                    coordinator.send_signal(signal.SIGTERM)
                self.assertNotEqual(coordinator.wait(timeout=20), 0)
                deadline = time.monotonic() + 3
                while any(_alive(pid) for pid in owned) and time.monotonic() < deadline:
                    time.sleep(.05)
                for pid in owned:
                    self.assertFalse(_alive(pid), f'leaked PID {pid}')
            finally:
                if coordinator.poll() is None:
                    coordinator.terminate()
                    coordinator.wait(timeout=15)

    def test_worker_failure_cleans_descendants(self):
        self._check_cleanup(terminate=False)

    def test_parent_termination_cleans_descendants(self):
        self._check_cleanup(terminate=True)

    @unittest.skipUnless(os.getenv('ALF_TEST_SHARED_GPU') == '1',
                         'requires three CUDA GPUs')
    def test_shared_gpu_gradients(self):
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port)):
                run_workers(['0', '1', '2', '0'], _gradient_worker, (directory,))
            self.assertEqual(len(list(Path(directory).iterdir())), 4)


if __name__ == '__main__':
    unittest.main()

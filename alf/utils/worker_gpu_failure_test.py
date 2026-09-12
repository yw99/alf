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
"""Regression tests for failure notification before multiprocessing teardown."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from alf.utils.worker_gpu import run_workers
from alf.utils.worker_gpu_test import _alive


def _child(directory, rank, ignore_term):
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    Path(directory, f'child-{rank}').touch()
    time.sleep(120)


def _failure_target(rank, world_size, directory, mode, queue):
    queue.put(rank)
    child = mp.get_context('spawn').Process(
        target=_child, args=(directory, rank, mode == 'blocked'))
    child.start()
    while not Path(directory, f'child-{rank}').exists():
        time.sleep(.02)
    Path(directory, f'ready-{rank}').write_text(
        json.dumps([os.getpid(), child.pid]))
    while not Path(directory, 'go').exists():
        time.sleep(.02)
    if mode == 'normal':
        child.terminate()
        child.join()
        return
    if rank != 0:
        time.sleep(120)
        return
    Path(directory, 'failed').touch()
    if mode == 'abrupt':
        os._exit(7)
    if mode == 'eof':
        from alf.utils import worker_gpu
        worker_gpu._failure_connection.close()
        os._exit(8)
    if mode in ('blocked', 'teardown_error'):
        from alf.bin import train

        def cleanup():
            if mode == 'teardown_error':
                raise RuntimeError('secondary teardown failure')
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            time.sleep(120)

        with patch.object(train, '_setup_device'), \
             patch.object(train, 'PerProcessContext'), \
             patch.object(train, '_setup_remote_configs_if_needed'), \
             patch.object(train.common, 'parse_conf_file',
                          side_effect=RuntimeError('original training failure')), \
             patch.object(train.alf, 'close_env', side_effect=cleanup):
            train.training_worker(0, 1, 'test.py', directory, flags_parsed=True)
    else:
        raise RuntimeError('original training failure')


class WorkerFailureTest(unittest.TestCase):
    def _run_case(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            code = ('from alf.utils.worker_gpu import run_workers; '
                    'from alf.utils.worker_gpu_failure_test import _failure_target; '
                    f'run_workers(["", ""], _failure_target, '
                    f'({directory!r}, {mode!r}), with_queue=True)')
            log = Path(directory, 'coordinator.log')
            with log.open('w') as stream:
                coordinator = subprocess.Popen([sys.executable, '-c', code],
                                               stdout=stream, stderr=stream)
            try:
                deadline = time.monotonic() + 60
                while len(list(Path(directory).glob('ready-*'))) != 2:
                    self.assertIsNone(coordinator.poll(), log.read_text())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.05)
                parents = dict(tuple(map(int, row.split())) for row in
                               subprocess.check_output(
                                   ['ps', '-eo', 'pid=,ppid='], text=True).splitlines())
                owned = {coordinator.pid}
                while True:
                    children = {p for p, pp in parents.items() if pp in owned}
                    if children <= owned:
                        break
                    owned.update(children)
                started = time.monotonic()
                if mode in ('sigint', 'sigterm'):
                    coordinator.send_signal(signal.SIGINT if mode == 'sigint'
                                            else signal.SIGTERM)
                else:
                    Path(directory, 'go').touch()
                code = coordinator.wait(timeout=20)
                self.assertLess(time.monotonic() - started, 20)
                if mode == 'normal':
                    self.assertEqual(code, 0, log.read_text())
                else:
                    self.assertNotEqual(code, 0, log.read_text())
                if mode in ('exception', 'blocked', 'teardown_error'):
                    self.assertIn('ChildProcessError: DDP rank 0 (PID ', log.read_text())
                    self.assertIn('RuntimeError: original training failure',
                                  log.read_text().split('ChildProcessError:')[-1])
                deadline = time.monotonic() + 3
                while any(_alive(pid) for pid in owned) and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertFalse([pid for pid in owned if _alive(pid)])
            finally:
                if coordinator.poll() is None:
                    coordinator.terminate()
                    coordinator.wait(timeout=15)

    def test_failure_before_multiprocessing_child_join(self):
        self._run_case('exception')

    def test_training_failure_before_blocked_teardown(self):
        self._run_case('blocked')

    def test_original_failure_survives_teardown_error(self):
        self._run_case('teardown_error')

    def test_abrupt_exit(self):
        self._run_case('abrupt')

    def test_eof_is_not_success(self):
        self._run_case('eof')

    def test_normal_completion(self):
        self._run_case('normal')

    def test_coordinator_sigint(self):
        self._run_case('sigint')

    def test_coordinator_sigterm(self):
        self._run_case('sigterm')


if __name__ == '__main__':
    unittest.main()

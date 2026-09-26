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
"""Compatibility and configuration checks for mapped training workers."""
from contextlib import nullcontext
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import alf
from alf.bin import train
from alf.config_helpers import adjust_config_by_multi_process_divider


def _logging_worker(rank, world_size, directory):
    sys.argv = ['logging_worker', '--root_dir', directory]
    marker = f'worker-log-marker-{rank}'
    with patch.object(train.torch.cuda, 'is_available', return_value=True), \
         patch.object(train.torch.cuda, 'device_count', return_value=1), \
         patch.object(train.torch, 'empty'), \
         patch.object(train.torch.cuda, 'synchronize'), \
         patch.object(train, '_setup_device'), \
         patch.object(train, '_setup_remote_configs_if_needed'), \
         patch.object(train.dist, 'init_process_group'), \
         patch.object(train.dist, 'is_initialized', return_value=False), \
         patch.object(train.common, 'parse_conf_file'), \
         patch.object(train, '_train',
                      side_effect=lambda **kw: train.logging.info(marker)), \
         patch.object(train, '_setup_logging', wraps=train._setup_logging) as setup:
        train._mapped_training_worker(rank, world_size, 'test.py', directory,
                                      MagicMock())
        assert setup.call_count == 1
        train.logging.flush()
    Path(directory, f'worker-{rank}.json').write_text(
        json.dumps({'pid': os.getpid(), 'marker': marker}))


class TrainWorkerGpuTest(unittest.TestCase):
    def tearDown(self):
        alf.reset_configs()

    def test_four_worker_configuration(self):
        for rank in range(4):
            alf.reset_configs()
            alf.config('create_environment', num_parallel_environments=1)
            alf.config('TrainerConfig', mini_batch_size=256, num_env_steps=600000,
                       initial_collect_steps=10000, evaluate=False)
            adjust_config_by_multi_process_divider(rank, 4)
            for name, expected in [('create_environment.num_parallel_environments', 1),
                                   ('TrainerConfig.mini_batch_size', 64),
                                   ('TrainerConfig.initial_collect_steps', 2500),
                                   ('TrainerConfig.num_env_steps', 150000)]:
                self.assertEqual(alf.get_config_value(name), expected)

    def test_master_port_context_honors_explicit_port(self):
        with patch.dict(os.environ, {'MASTER_PORT': '29600'}), \
             patch.object(train.common, 'get_unused_port') as allocate:
            with train._master_port_context() as port:
                self.assertEqual(port, 29600)
            allocate.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(train.common, 'get_unused_port',
                          return_value=nullcontext(12345)) as allocate:
            with train._master_port_context() as port:
                self.assertEqual(port, 12345)
            allocate.assert_called_once_with(12355)
        for invalid in ('0', '65536', 'not-a-port'):
            with self.subTest(invalid=invalid), \
                 patch.dict(os.environ, {'MASTER_PORT': invalid}):
                with self.assertRaises(ValueError):
                    train._master_port_context()

    def test_legacy_worker_allocation(self):
        for gpus, ngpu, expected in [('0,1,2,3', 1, 4), ('0,1,2,3', 2, 2),
                                     ('0,1,2', 1, 3)]:
            with self.subTest(gpus=gpus, ngpu=ngpu), tempfile.TemporaryDirectory() as root:
                options = SimpleNamespace(root_dir=root, worker_gpus=None,
                    distributed='multi-gpu', store_snapshot=False,
                    num_gpus_per_ddp_worker=ngpu)
                context = MagicMock()
                with patch.object(train, 'FLAGS', options), \
                     patch.object(train, 'check_valid_launch'), \
                     patch.object(train.common, 'get_conf_file', return_value='test.py'), \
                     patch.object(train.common, 'get_unused_port', return_value=nullcontext(12345)), \
                     patch.object(train.mp, 'Manager'), \
                     patch.object(train.multiprocessing, 'get_context', return_value=context), \
                     patch.object(train, 'training_worker') as worker, \
                     patch.object(train, '_train_with_worker_gpus') as mapped, \
                     patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpus):
                    train.main(None)
                    mapped.assert_not_called()
                    self.assertEqual(context.Process.call_count, expected - 1)
                    for rank, call in enumerate(context.Process.call_args_list, start=1):
                        self.assertEqual(call.kwargs['args'][:2], (rank, expected))
                    self.assertEqual(worker.call_args.args[:2], (0, expected))

    def test_spawned_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ('from alf.utils.worker_gpu import run_workers; '
                    'from alf.bin.train_worker_gpu_test import _logging_worker; '
                    f'run_workers(["", ""], _logging_worker, ({directory!r},))')
            result = subprocess.run([sys.executable, '-c', code],
                                    capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            for rank in (0, 1):
                record = json.loads(Path(directory, f'worker-{rank}.json').read_text())
                marker = record['marker']
                lines = [line for line in result.stderr.splitlines() if marker in line]
                self.assertEqual(len(lines), 1, result.stderr)
                self.assertRegex(lines[0], r'^I[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}')
                files = [p for p in Path(directory).glob(f'*.{record["pid"]}')
                         if not p.is_symlink()]
                self.assertEqual(len(files), 1, list(Path(directory).iterdir()))
                content = files[0].read_text()
                self.assertEqual(content.count(marker), 1)
                self.assertRegex(content, r'I[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}')

    def test_mapped_worker_cuda_failure(self):
        options = MagicMock()
        with patch.object(train, '_define_flags') as define, \
             patch.object(train, 'FLAGS', options), \
             patch.object(train, '_setup_logging'), \
             patch.object(train.logging, 'use_absl_handler'), \
             patch.object(train.torch.cuda, 'is_available', return_value=False), \
             patch.object(train, 'training_worker') as worker, \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-unavailable'):
            with self.assertRaisesRegex(RuntimeError, 'rank 3, assigned GPU GPU-unavailable'):
                train._mapped_training_worker(3, 4, 'test.py', '/tmp/test', None)
            define.assert_called_once()
            options.assert_called_once()
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()

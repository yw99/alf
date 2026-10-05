# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import alf


class BafcV3SingleLayerTransformerConfigTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._config = self._repo_root / 'alf/examples/bafcv3_dmc_conf.py'
        self._launcher = self._repo_root / (
            'alf/examples/run_humanoid_run_bafcv3_single_layer_transformer_seed0123-4g.sh')

    def test_config_default_and_opt_in(self):
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                pre_config = ({} if enabled is None else
                              {'bafcv3_use_single_layer_transformer_encoder': enabled})
                code = f'''
import json
import alf
from alf.utils import common
alf.pre_config({pre_config!r})
common.parse_conf_file({str(self._config)!r}, create_env=False)
keys = ['use_single_layer_transformer_encoder', 'use_actor_id_encoding',
        'detach_actor_policy_input', 'num_actor_eval_samples',
        'actor_encoding_dim', 'actor_eval_type', 'eval_samples_source']
print(json.dumps({{key: alf.get_config_value('BafcAlgorithmV3.' + key)
                  for key in keys}}))
'''
                result = subprocess.run(
                    [sys.executable, '-c', code], cwd=self._repo_root,
                    check=True, text=True, capture_output=True)
                config = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(config, dict(
                    use_single_layer_transformer_encoder=bool(enabled),
                    use_actor_id_encoding=False,
                    detach_actor_policy_input=False,
                    num_actor_eval_samples=512, actor_encoding_dim=None,
                    actor_eval_type='last_two', eval_samples_source='trainable'))

    def _dry_run(self, results, *options, env=None):
        return subprocess.run(
            ['bash', str(self._launcher), '--dry-run', '--dir', str(results),
             *options], cwd=self._repo_root,
            env=env or dict(os.environ, PYTHON_BIN=sys.executable),
            text=True, capture_output=True)

    def test_launcher_commands_and_overrides(self):
        subprocess.run(['bash', '-n', str(self._launcher)], check=True)
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp:
                results = Path(tmp) / 'results with spaces'
                options = (['--steps', '1200', '--checkpoints', '2',
                            '--gpus', '4,5,6,7', '--base-port', '31000']
                           if override else [])
                run = self._dry_run(results, *options)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertFalse(results.exists())
                commands = [shlex.split(line) for line in run.stdout.splitlines()
                            if line.startswith('CUDA_VISIBLE_DEVICES=')]
                self.assertEqual(len(commands), 4)
                roots = []
                for seed, command in enumerate(commands):
                    self.assertEqual(command[0], 'CUDA_VISIBLE_DEVICES=' +
                                     ('4,5,6,7' if override else '0,1,2,3'))
                    self.assertEqual(command[1], 'MASTER_PORT=' +
                                     str((31000 if override else 29900) + seed))
                    root = Path(command[command.index('--root_dir') + 1])
                    roots.append(root)
                    self.assertEqual(root.name, f'seed_{seed}')
                    self.assertRegex(root.parent.name, r'^\d{8}T\d{6}Z$')
                    self.assertEqual(root.parent.parent, results / 'humanoid_run' /
                                     'bafcv3_single_layer_transformer_seed0123_4g_actor_lnTrue')
                    self.assertEqual(command[command.index('--distributed') + 1],
                                     'multi-gpu')
                    settings = dict(command[i + 1].split('=', 1)
                                    for i, item in enumerate(command)
                                    if item == '--conf_param')
                    expected = {
                        'TrainerConfig.random_seed': str(seed),
                        'TrainerConfig.num_env_steps': '1200' if override else '600000',
                        'TrainerConfig.num_checkpoints': '2' if override else '10',
                        'TrainerConfig.num_updates_per_train_iter': '12',
                        'BafcAlgorithmV3.actor_utd': '1',
                        'BafcAlgorithmV3.critic_utd': '11',
                        'debug_mode': 'False',
                        'bafcv3_use_single_layer_transformer_encoder': 'True',
                        'bafcv3_use_actor_id_encoding': 'False',
                        'bafcv3_detach_actor_policy_input': 'False',
                        'bafcv3_actor_use_ln': 'True',
                        'bafcv3_actor_critic_pairing': 'False',
                        'bafcv3_num_actor_critic': '10',
                        'bafcv3_num_sampled_critics_for_actor': '8',
                        'bafcv3_use_random_critic_targets': 'True',
                        'bafcv3_num_sampled_critic_targets': '1',
                        'make_ddp_performer.find_unused_parameters': 'True',
                        'create_environment.env_name': "'humanoid:run'",
                    }
                    for key, value in expected.items():
                        self.assertEqual(settings[key], value)
                self.assertEqual(len(set(roots)), 4)
                self.assertEqual(len({root.parent for root in roots}), 1)

    def test_launcher_rejects_invalid_arguments_and_directory_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / 'results'
            for options in (['--gpus', '0,1,2'], ['--gpus', '0,1,2,2'],
                            ['--gpus', '0,1,2,3,4'], ['--steps', '0'],
                            ['--checkpoints', '-1'], ['--base-port', '65533'],
                            ['--base-port', '9999999999999999999999']):
                with self.subTest(options=options):
                    run = self._dry_run(results, *options)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse(results.exists())
            # Fix the clock so collision rejection does not depend on timing.
            fake_bin = Path(tmp) / 'bin'
            fake_bin.mkdir()
            date = fake_bin / 'date'
            date.write_text('#!/bin/sh\necho 20260930T120000Z\n')
            date.chmod(0o755)
            collision = (results / 'humanoid_run' /
                         'bafcv3_single_layer_transformer_seed0123_4g_actor_lnTrue' / '20260930T120000Z')
            collision.mkdir(parents=True)
            run = self._dry_run(results, env=dict(
                os.environ, PYTHON_BIN=sys.executable,
                PATH=str(fake_bin) + os.pathsep + os.environ['PATH']))
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Refusing to reuse', run.stderr)
            self.assertEqual(list(collision.iterdir()), [])


class BafcV3PreLNQKNormConfigTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._config = self._repo_root / 'alf/examples/bafcv3_dmc_conf.py'
        self._launcher = self._repo_root / (
            'alf/examples/run_humanoid_run_bafcv3_preln_qknorm_seed01-8g.sh')

    def _dry_run(self, results, *options, env=None):
        return subprocess.run(
            ['bash', str(self._launcher), '--dry-run', '--dir', str(results),
             *options], cwd=self._repo_root,
            env=env or dict(os.environ, PYTHON_BIN=sys.executable),
            text=True, capture_output=True)

    def test_config_default_and_opt_in(self):
        algorithm_flags = (
            'use_target_actor_encoder', 'use_legacy_actor_gradient',
            'debug_gradient_chain', 'debug_gradient_chain_compare_backends')
        transformer_flags = ('norm_first', 'final_norm', 'normalize_qk')
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                settings = {}
                if enabled is not None:
                    settings.update(('bafcv3_' + key, enabled)
                                    for key in algorithm_flags)
                    settings.update(('bafcv3_transformer_' + key, enabled)
                                    for key in transformer_flags)
                    settings['bafcv3_transformer_qk_norm_eps'] = 2e-6
                code = f"""
import json
import alf
from alf.utils import common
alf.pre_config({settings!r})
common.parse_conf_file({str(self._config)!r}, create_env=False)
keys = {{'BafcAlgorithmV3': {algorithm_flags!r},
        'TransformerEncoder': {transformer_flags!r} + ('qk_norm_eps',)}}
print(json.dumps({{scope + '.' + key: alf.get_config_value(scope + '.' + key)
                  for scope, fields in keys.items() for key in fields}}))
"""
                result = subprocess.run(
                    [sys.executable, '-c', code], cwd=self._repo_root,
                    check=True, text=True, capture_output=True)
                config = json.loads(result.stdout.strip().splitlines()[-1])
                expected = {'BafcAlgorithmV3.' + key: bool(enabled)
                            for key in algorithm_flags}
                expected.update(('TransformerEncoder.' + key, bool(enabled))
                                for key in transformer_flags)
                expected['TransformerEncoder.qk_norm_eps'] = (
                    1e-6 if enabled is None else 2e-6)
                self.assertEqual(config, expected)

    def test_launcher_commands_and_overrides(self):
        subprocess.run(['bash', '-n', str(self._launcher)], check=True)
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp:
                results = Path(tmp) / 'results with spaces'
                options = (['--env', 'humanoid:walk', '--steps', '1200',
                            '--checkpoints', '2', '--gpus', '7,6,5,4,3,2,1,0',
                            '--base-port', '31000'] if override else [])
                run = self._dry_run(results, *options)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertFalse(results.exists())
                commands = [shlex.split(line) for line in run.stdout.splitlines()
                            if line.startswith('CUDA_VISIBLE_DEVICES=')]
                self.assertEqual(len(commands), 8)
                roots, ports, configurations = set(), set(), set()
                for index, command in enumerate(commands):
                    group, within_group = divmod(index, 4)
                    qk_index, seed = divmod(within_group, 2)
                    actor_ln = ('True', 'False')[group]
                    qk_norm = ('False', 'True')[qk_index]
                    gpu_groups = (('7,6,5,4', '3,2,1,0') if override else
                                  ('0,1,2,3', '4,5,6,7'))
                    self.assertEqual(command[0],
                                     'CUDA_VISIBLE_DEVICES=' + gpu_groups[group])
                    self.assertEqual(command[1], 'MASTER_PORT=' +
                                     str((31000 if override else 29920) + index))
                    ports.add(command[1])
                    root = Path(command[command.index('--root_dir') + 1])
                    roots.add(root)
                    self.assertEqual(root.name, f'seed_{seed}')
                    self.assertEqual(root.parent.name, 'qk_norm' + qk_norm)
                    self.assertEqual(root.parent.parent.name, 'actor_ln' + actor_ln)
                    self.assertRegex(root.parents[2].name, r'^\d{8}T\d{6}Z$')
                    self.assertEqual(root.parents[3], results /
                                     ('humanoid_walk' if override else 'humanoid_run') /
                                     'bafcv3_preln_qknorm_seed01_8g')
                    self.assertEqual(command[command.index('--distributed') + 1],
                                     'multi-gpu')
                    settings = dict(command[i + 1].split('=', 1)
                                    for i, item in enumerate(command)
                                    if item == '--conf_param')
                    expected = {
                        'TrainerConfig.random_seed': str(seed),
                        'TrainerConfig.confirm_checkpoint_upon_crash': 'False',
                        'TrainerConfig.num_env_steps': '1200' if override else '600000',
                        'TrainerConfig.num_checkpoints': '2' if override else '10',
                        'TrainerConfig.num_updates_per_train_iter': '12',
                        'TrainerConfig.debug_summaries': 'True',
                        'BafcAlgorithmV3.actor_utd': '1',
                        'BafcAlgorithmV3.critic_utd': '11',
                        'debug_mode': 'False',
                        'bafcv3_use_single_layer_transformer_encoder': 'True',
                        'bafcv3_use_actor_id_encoding': 'False',
                        'bafcv3_detach_actor_policy_input': 'False',
                        'bafcv3_actor_use_ln': actor_ln,
                        'bafcv3_transformer_norm_first': 'True',
                        'bafcv3_transformer_final_norm': 'True',
                        'bafcv3_transformer_normalize_qk': qk_norm,
                        'bafcv3_num_attention_heads': '1',
                        'bafcv3_use_target_actor_encoder': 'True',
                        'bafcv3_use_legacy_actor_gradient': 'False',
                        'bafcv3_debug_gradient_chain': 'True',
                        'bafcv3_debug_gradient_chain_compare_backends': 'True',
                        'bafcv3_actor_critic_pairing': 'False',
                        'bafcv3_num_actor_critic': '10',
                        'bafcv3_num_sampled_critics_for_actor': '8',
                        'bafcv3_use_random_critic_targets': 'True',
                        'bafcv3_num_sampled_critic_targets': '1',
                        'make_ddp_performer.find_unused_parameters': 'True',
                        'create_environment.env_name': (
                            "'humanoid:walk'" if override else "'humanoid:run'"),
                    }
                    self.assertEqual(settings, expected)
                    configurations.add((actor_ln, qk_norm, seed))
                self.assertEqual(len(roots), 8)
                self.assertEqual(len(ports), 8)
                self.assertEqual(len(configurations), 8)
                self.assertEqual(len({root.parents[2] for root in roots}), 1)

    def test_launcher_rejects_invalid_arguments_and_directory_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / 'results'
            invalid = (['--gpus', '0,1,2,3'], ['--gpus', '0,1,2,3,4,5,6,6'],
                       ['--gpus', '0,1,2,3,4,5,6,7,8'], ['--steps', '0'],
                       ['--steps', '1.5'], ['--checkpoints', '-1'],
                       ['--base-port', '65529'], ['--base-port', '0'],
                       ['--base-port', '9999999999999999999999'],
                       ['--env', 'humanoid/run'], ['--unknown'], ['--gpus'])
            for options in invalid:
                with self.subTest(options=options):
                    run = self._dry_run(results, *options)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse(results.exists())
            fake_bin = Path(tmp) / 'bin'
            fake_bin.mkdir()
            date = fake_bin / 'date'
            date.write_text('#!/bin/sh\necho 20261004T120000Z\n')
            date.chmod(0o755)
            collision = (results / 'humanoid_run' /
                         'bafcv3_preln_qknorm_seed01_8g' / '20261004T120000Z')
            collision.mkdir(parents=True)
            run = self._dry_run(results, env=dict(
                os.environ, PYTHON_BIN=sys.executable,
                PATH=str(fake_bin) + os.pathsep + os.environ['PATH']))
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('Refusing to reuse', run.stderr)
            self.assertEqual(list(collision.iterdir()), [])

    def test_launcher_accepts_last_valid_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / 'results'
            run = self._dry_run(results, '--base-port', '65528')
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('MASTER_PORT=65535 ', run.stdout)
            self.assertFalse(results.exists())


if __name__ == '__main__':
    alf.test.main()

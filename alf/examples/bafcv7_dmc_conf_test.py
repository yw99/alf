# Copyright (c) 2026 Horizon Robotics and ALF Contributors. All Rights Reserved.

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import alf


class BafcV7DmcConfigTest(alf.test.TestCase):

    def setUp(self):
        super().setUp()
        self._repo_root = Path(__file__).resolve().parents[2]
        self._config = self._repo_root / "alf/examples/bafcv7_dmc_conf.py"
        self._launcher = (
            self._repo_root /
            "alf/examples/run_hopper_hop_bafcv7_seeds0123-4g.sh")

    def _parse_variant(self, variant):
        code = """
import json
import alf
from alf.utils import common
alf.pre_config({
    "bafcv7_variant": %r,
    "bafcv7_env_name": "hopper:hop",
})
common.parse_conf_file(%r, create_env=False)
keys = [
    "BafcAlgorithmV7.num_actors",
    "BafcAlgorithmV7.num_critics",
    "BafcAlgorithmV7.temporal_noise_mix",
    "BafcAlgorithmV7.training_policy",
    "BafcAlgorithmV7.actor_update_mode",
    "BafcAlgorithmV7.actor_utd",
    "BafcAlgorithmV7.critic_utd",
    "BafcAlgorithmV7.num_sampled_critic_targets",
    "TrainerConfig.num_updates_per_train_iter",
    "TrainerConfig.num_env_steps",
]
print(json.dumps({key: alf.get_config_value(key) for key in keys}))
""" % (variant, str(self._config))
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self._repo_root,
            check=True,
            text=True,
            capture_output=True)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_both_presets_parse(self):
        ensemble = self._parse_variant("ensemble_base")
        self.assertEqual(ensemble["BafcAlgorithmV7.num_actors"], 10)
        self.assertEqual(ensemble["BafcAlgorithmV7.num_critics"], 10)
        self.assertEqual(ensemble["BafcAlgorithmV7.training_policy"], "base")
        self.assertEqual(ensemble["BafcAlgorithmV7.actor_update_mode"],
                         "paired")
        self.assertEqual(ensemble["BafcAlgorithmV7.temporal_noise_mix"], 0.1)

        seeded = self._parse_variant("single_seeded")
        self.assertEqual(seeded["BafcAlgorithmV7.num_actors"], 1)
        self.assertEqual(seeded["BafcAlgorithmV7.num_critics"], 10)
        self.assertEqual(seeded["BafcAlgorithmV7.training_policy"], "seeded")
        self.assertEqual(seeded["BafcAlgorithmV7.actor_update_mode"],
                         "min_all")
        self.assertEqual(seeded["BafcAlgorithmV7.temporal_noise_mix"], 0.9)

        for preset in (ensemble, seeded):
            self.assertEqual(preset["BafcAlgorithmV7.actor_utd"], 1)
            self.assertEqual(preset["BafcAlgorithmV7.critic_utd"], 3)
            self.assertEqual(
                preset["BafcAlgorithmV7.num_sampled_critic_targets"], 1)
            self.assertEqual(
                preset["TrainerConfig.num_updates_per_train_iter"], 12)
            self.assertEqual(preset["TrainerConfig.num_env_steps"], 800000)

    def test_config_requires_launcher_environment(self):
        code = """
from alf.utils import common
common.parse_conf_file(%r, create_env=False)
""" % str(self._config)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self._repo_root,
            text=True,
            capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "BAFCv7 does not define a default environment",
            result.stderr + result.stdout)

    def test_launcher_dry_run_emits_eight_isolated_jobs(self):
        environment = os.environ.copy()
        environment["PYTHON_BIN"] = sys.executable
        result = subprocess.run(
            [
                "bash", str(self._launcher), "--dry-run", "--dir",
                "/tmp/bafcv7_launcher_test", "--base-port", "31000"
            ],
            cwd=self._repo_root,
            env=environment,
            check=True,
            text=True,
            capture_output=True)
        commands = [
            line for line in result.stdout.splitlines()
            if " --root_dir " in line
        ]
        self.assertEqual(len(commands), 8)
        ports = re.findall(r"MASTER_PORT=(\d+)", result.stdout)
        self.assertEqual(ports, [str(port) for port in range(31000, 31008)])
        self.assertEqual(
            sum("bafcv7_variant=\\'ensemble_base\\'" in line
                for line in commands), 4)
        self.assertEqual(
            sum("bafcv7_variant=\\'single_seeded\\'" in line
                for line in commands), 4)
        self.assertTrue(all(
            "bafcv7_env_name=\\'hopper:hop\\'" in line
            for line in commands))
        self.assertTrue(all(
            "create_environment.env_name" not in line for line in commands))
        self.assertIn("launched none", result.stdout)


if __name__ == "__main__":
    alf.test.main()

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
        self._single_seed_launcher = (
            self._repo_root /
            "alf/examples/run_hopper_hop_bafcv7_seed-4g.sh")

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
    "BafcAlgorithmV7.policy_feature_mode",
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
            self.assertEqual(preset["BafcAlgorithmV7.policy_feature_mode"],
                             "mean_log_std")
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
        root_dirs = [
            re.search(r"--root_dir ([^ ]+)", line).group(1)
            for line in commands
        ]
        self.assertEqual(len(set(root_dirs)), 8)
        self.assertTrue(all(
            "/hopper_hop/bafcv7_policy_features_4g/mean_log_std/" in line
            for line in commands))
        self.assertTrue(all(
            "BafcAlgorithmV7.policy_feature_mode=\\'mean_log_std\\'" in line
            for line in commands))
        self.assertIn("Entropy regularization: disabled", result.stdout)
        self.assertIn("Policy features: mean_log_std", result.stdout)
        for variant, temporal_noise_mix in (("ensemble_base", "0.10"),
                                            ("single_seeded", "0.90")):
            variant_commands = [
                line for line in commands
                if "bafcv7_variant=\\'%s\\'" % variant in line
            ]
            self.assertEqual(len(variant_commands), 4)
            self.assertTrue(all(
                "/%s/lambda%s/actor_utd1_critic_utd3/seed_" %
                (variant, temporal_noise_mix) in line
                for line in variant_commands))
            self.assertTrue(all(
                "BafcAlgorithmV7.temporal_noise_mix=%s" %
                temporal_noise_mix in line for line in variant_commands))
            self.assertTrue(all(
                "BafcAlgorithmV7.actor_utd=1" in line
                and "BafcAlgorithmV7.critic_utd=3" in line
                for line in variant_commands))
        self.assertTrue(all(
            "bafcv7_env_name=\\'hopper:hop\\'" in line
            for line in commands))
        self.assertTrue(all(
            "create_environment.env_name" not in line for line in commands))
        self.assertIn("launched none", result.stdout)

    def test_single_seed_launcher_defaults_to_seed_zero(self):
        environment = os.environ.copy()
        environment["PYTHON_BIN"] = sys.executable
        result = subprocess.run(
            [
                "bash", str(self._single_seed_launcher), "--dry-run",
                "--dir", "/tmp/bafcv7_single_seed_launcher_test",
                "--base-port", "32000"
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
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            re.findall(r"MASTER_PORT=(\d+)", result.stdout),
            ["32000", "32001"])
        self.assertTrue(all(
            "/hopper_hop/bafcv7_policy_features_4g/mean_log_std/" in line
            for line in commands))
        self.assertTrue(all(
            "BafcAlgorithmV7.policy_feature_mode=\\'mean_log_std\\'" in line
            for line in commands))
        self.assertIn("Entropy regularization: disabled", result.stdout)
        self.assertIn("Policy features: mean_log_std", result.stdout)
        self.assertTrue(all(
            "TrainerConfig.random_seed=0" in line and "/seed_0 " in line
            for line in commands))
        for variant, temporal_noise_mix in (("ensemble_base", "0.10"),
                                            ("single_seeded", "0.90")):
            matching = [
                line for line in commands
                if "bafcv7_variant=\\'%s\\'" % variant in line
            ]
            self.assertEqual(len(matching), 1)
            self.assertIn(
                "/%s/lambda%s/actor_utd1_critic_utd3/seed_0 " %
                (variant, temporal_noise_mix), matching[0])
            self.assertIn(
                "BafcAlgorithmV7.temporal_noise_mix=%s" %
                temporal_noise_mix, matching[0])
            self.assertIn("BafcAlgorithmV7.actor_utd=1", matching[0])
            self.assertIn("BafcAlgorithmV7.critic_utd=3", matching[0])
        self.assertIn("emitted two jobs and launched none", result.stdout)

    def test_launchers_accept_explicit_action_quantile_mode(self):
        environment = os.environ.copy()
        environment["PYTHON_BIN"] = sys.executable
        for launcher, job_count, base_port in (
                (self._launcher, 8, 33000),
                (self._single_seed_launcher, 2, 33100)):
            with self.subTest(launcher=launcher.name):
                result = subprocess.run(
                    [
                        "bash", str(launcher), "--dry-run", "--dir",
                        "/tmp/bafcv7_quantile_launcher_test", "--base-port",
                        str(base_port), "--policy-features",
                        "action_quantiles"
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
                self.assertEqual(len(commands), job_count)
                self.assertTrue(all(
                    "/hopper_hop/bafcv7_policy_features_4g/"
                    "action_quantiles/" in line for line in commands))
                self.assertTrue(all(
                    "BafcAlgorithmV7.policy_feature_mode="
                    "\\'action_quantiles\\'" in line for line in commands))
                self.assertIn("Policy features: action_quantiles",
                              result.stdout)

    def test_launchers_reject_invalid_policy_feature_mode(self):
        environment = os.environ.copy()
        environment["PYTHON_BIN"] = sys.executable
        for launcher in (self._launcher, self._single_seed_launcher):
            with self.subTest(launcher=launcher.name):
                result = subprocess.run(
                    [
                        "bash", str(launcher), "--dry-run",
                        "--policy-features", "invalid"
                    ],
                    cwd=self._repo_root,
                    env=environment,
                    text=True,
                    capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--policy-features must be",
                              result.stderr + result.stdout)

    def test_launcher_shell_syntax(self):
        for launcher in (self._launcher, self._single_seed_launcher):
            with self.subTest(launcher=launcher.name):
                subprocess.run(
                    ["bash", "-n", str(launcher)],
                    cwd=self._repo_root,
                    check=True)


if __name__ == "__main__":
    alf.test.main()

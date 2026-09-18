# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.

"""Tests for plot_dog_humanoid_bafc_comparison."""

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np

import alf
from alf.utils import plot_dog_humanoid_bafc_comparison as plotter


class PlotDogHumanoidBafcComparisonTest(alf.test.TestCase):

    def test_initial_eval_trust_threshold(self):
        self.assertEqual(plotter.INITIAL_EVAL_TRUST_THRESHOLD, 30.0)

    def test_parse_args_accepts_one_or_more_tasks(self):
        with mock.patch("sys.argv", [
                "plotter", "--tasks", "dog_stand", "humanoid_stand",
                "--task", "dog"]):
            args = plotter._parse_args()
        self.assertEqual(args.tasks,
                         ["dog_stand", "humanoid_stand", "dog"])

    def test_main_only_plots_selected_task_family(self):
        args = SimpleNamespace(
            tasks=["dog_trot"], workspace_root="/ws",
            local_results_root="/local", server_copy_root="/server1",
            server2_copy_root="/server2", server4_copy_root="/server4",
            server5_copy_root=None, server6_copy_root=None,
            output_root="/plots")
        with mock.patch.object(plotter, "_parse_args", return_value=args), \
                mock.patch.object(plotter, "plot_average_return") as plot, \
                mock.patch.object(plotter, "plot_eval_trust_over_max") as trust, \
                mock.patch.object(plotter, "plot_raw_eval_trust_metric") as raw:
            plotter.main()

        self.assertEqual(plot.call_count, 3)
        self.assertEqual([call.args[0] for call in plot.call_args_list],
                         ["dog_trot", "dog_trot", "dog_trot"])
        baseline = plot.call_args_list[0]
        self.assertEqual(baseline.kwargs["colors"],
                         plotter.BASELINE_ALGORITHM_COLORS)
        self.assertEqual(len(baseline.args[1]["SAC+"]), 4)
        self.assertIn("seed0123", baseline.kwargs["filename"])
        trust.assert_not_called()
        raw.assert_not_called()

    def test_baselines_discover_across_copies_and_prefer_complete_budget(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in range(4):
                for name, server in (("dog_walk_sac_600k", "server5_copy"),
                                     ("dog_walk_sac_800k", "server5_copy"),
                                     ("dog_walk_td3v2", "server%d_copy" % (seed + 2))):
                    os.makedirs(os.path.join(root, server,
                                             "%s_s%d" % (name, seed), "train"))
            # An incomplete longer run must not replace a complete seed set.
            os.makedirs(os.path.join(root, "server6_copy",
                                     "dog_walk_sac_1m_s0", "train"))
            rlpd = {"dog": {"RLPD": ["rlpd_s%d" % s for s in range(4)]}}
            groups = plotter.build_baseline_run_groups(
                root, rlpd, [], tasks=["dog"])
            self.assertEqual(list(groups["dog"]), ["SAC", "TD3", "SAC+"])
            self.assertEqual(groups["dog"]["SAC+"], rlpd["dog"]["RLPD"][:4])
            for seed, path in enumerate(groups["dog"]["SAC"]):
                self.assertTrue(path.endswith("dog_walk_sac_800k_s%d" % seed))
            self.assertEqual(len(groups["dog"]["TD3"]), 4)
            self.assertIn("server2_copy", groups["dog"]["TD3"][0])

    def test_baselines_match_unnamed_budget_from_saved_config(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in range(4):
                name = ("humanoid_walk_sac_600k_s%d" % seed if seed < 3
                        else "humanoid_walk_sac_s3")
                run = os.path.join(root, "server4_copy", name)
                os.makedirs(os.path.join(run, "train"))
                if seed == 3:
                    with open(os.path.join(run, "alf_config.py"), "w") as f:
                        f.write("alf.pre_config({'TrainerConfig.num_env_steps': 600000})")
            rlpd = {"humanoid": {"RLPD": ["s0", "s1", "s2", "s3"]}}
            groups = plotter.build_baseline_run_groups(
                root, rlpd, [], tasks=["humanoid"])
            self.assertEqual(len(groups["humanoid"]["SAC"]), 4)
            self.assertTrue(groups["humanoid"]["SAC"][3].endswith(
                "humanoid_walk_sac_s3"))

    def test_baselines_report_missing_and_reject_ambiguous_copies(self):
        with tempfile.TemporaryDirectory() as root:
            rlpd = {"humanoid": {"RLPD": ["s0", "s1", "s2", "s3"]}}
            custom = os.path.join(root, "custom")
            for seed in range(4):
                os.makedirs(os.path.join(custom,
                                         "humanoid_walk_sac_s%d" % seed, "train"))
            with mock.patch("builtins.print") as report:
                groups = plotter.build_baseline_run_groups(
                    root, rlpd, [custom], tasks=["humanoid"])
            self.assertEqual(list(groups["humanoid"]), ["SAC", "SAC+"])
            self.assertIn("omitting TD3", report.call_args.args[0])
            os.makedirs(os.path.join(root, "server6_copy",
                                     "humanoid_walk_sac_s0", "train"))
            with self.assertRaisesRegex(ValueError, "Ambiguous humanoid SAC seed 0"):
                plotter.build_baseline_run_groups(
                    root, rlpd, [custom], tasks=["humanoid"])

    def test_aggregate_interpolates_over_overlap_with_population_std(self):
        curves = [
            plotter.ScalarCurve(np.array([0., 2., 4.]),
                                np.array([0., 2., 4.])),
            plotter.ScalarCurve(np.array([1., 3., 5.]),
                                np.array([2., 4., 6.])),
        ]
        aggregate = plotter.aggregate_curves(curves, ["seed0", "seed1"],
                                             "tag")
        np.testing.assert_allclose(aggregate.steps, [1., 2., 4.])
        np.testing.assert_allclose(aggregate.mean, [1.5, 2.5, 4.5])
        np.testing.assert_allclose(aggregate.std, [.5, .5, .5])

    def test_read_scalar_curve_retains_last_duplicate_step(self):
        events = [SimpleNamespace(step=1, value=2.),
                  SimpleNamespace(step=1, value=3.),
                  SimpleNamespace(step=2, value=4.)]
        accumulator = mock.Mock()
        accumulator.Tags.return_value = {"scalars": ["tag"]}
        accumulator.Scalars.return_value = events
        with tempfile.TemporaryDirectory() as logdir, mock.patch.object(
                plotter, "EventAccumulator", return_value=accumulator):
            curve = plotter._read_scalar_curve(logdir, "tag")
        np.testing.assert_allclose(curve.steps, [1., 2.])
        np.testing.assert_allclose(curve.values, [3., 4.])

    def test_missing_directory_and_tag_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            plotter._read_scalar_curve("/definitely/missing", "tag")

        accumulator = mock.Mock()
        accumulator.Tags.return_value = {"scalars": ["other"]}
        with tempfile.TemporaryDirectory() as logdir, mock.patch.object(
                plotter, "EventAccumulator", return_value=accumulator):
            with self.assertRaisesRegex(ValueError, "Required scalar tag"):
                plotter._read_scalar_curve(logdir, "tag")

    def test_run_mapping_separates_four_seed_and_seed01_groups(self):
        groups = plotter.build_run_groups("/ws", "/local", "/server1",
                                          "/server2", "/server4")
        for env in ("dog", "dog_fetch", "humanoid"):
            for label in ("RLPD", "BAFCv3", "BAFC_TR"):
                self.assertEqual(len(groups[env][label]), 4)
                for seed, path in enumerate(groups[env][label]):
                    self.assertIn("%d" % seed, os.path.basename(path))
        self.assertEqual(set(groups["dog_run"]),
                         {"RLPD", "BAFCv3", "BAFCv6"})
        for label in ("RLPD", "BAFCv3", "BAFCv6"):
            self.assertEqual(len(groups["dog_run"][label]), 4)
            for seed, path in enumerate(groups["dog_run"][label]):
                self.assertIn("%d" % seed, os.path.basename(path))
        self.assertIn("server2", groups["dog_run"]["BAFCv6"][0])
        self.assertIn("server2", groups["dog_run"]["BAFCv6"][1])
        self.assertIn("server1", groups["dog_run"]["BAFCv6"][2])
        self.assertIn("server1", groups["dog_run"]["BAFCv6"][3])
        self.assertNotIn("BAFCv6", groups["humanoid"])
        self.assertNotIn("BAFCv6", groups["dog"])
        self.assertEqual(set(groups["dog_trot"]), {"RLPD", "BAFCv3"})
        for run_dirs in groups["dog_trot"].values():
            self.assertEqual(len(run_dirs), 4)
        self.assertEqual(set(groups["dog_stand"]), {"RLPD", "BAFCv3"})
        self.assertTrue(all("server1" in path for path in
                            groups["dog_stand"]["RLPD"]))
        self.assertTrue(all("server2" in path for path in
                            groups["dog_stand"]["BAFCv3"]))
        self.assertEqual(set(groups["humanoid_run"]), {"RLPD", "BAFCv3"})
        self.assertTrue(all("server4" in path for path in
                            groups["humanoid_run"]["RLPD"]))
        for run_dirs in groups["humanoid_run"].values():
            self.assertEqual(len(run_dirs), 4)
        self.assertEqual(set(groups["humanoid_stand"]), {"RLPD", "BAFCv3"})
        self.assertTrue(all("server4" in path for path in
                            groups["humanoid_stand"]["RLPD"]))
        self.assertTrue(all("server1" in path for path in
                            groups["humanoid_stand"]["BAFCv3"]))
        for run_dirs in groups["humanoid_stand"].values():
            self.assertEqual(len(run_dirs), 4)
        self.assertEqual(
            set(groups["humanoid_seed01"]),
            {"RLPD", "BAFCv3", "BAFC_TR", "BAFCv6"})
        for run_dirs in groups["humanoid_seed01"].values():
            self.assertEqual(len(run_dirs), 2)
            for seed, path in enumerate(run_dirs):
                self.assertIn("%d" % seed, os.path.basename(path))

    def test_rlpd_ours_mapping_uses_requested_tasks_and_hopper_utds(self):
        groups = plotter.build_rlpd_ours_run_groups(
            "/ws", "/local", "/server1", "/server2", "/server4")
        self.assertEqual(
            set(groups), {"dog_fetch", "dog_run", "dog_stand", "dog_trot",
                          "dog", "humanoid", "humanoid_run",
                          "humanoid_stand", "hopper_hop"})
        for env in ("dog_fetch", "dog_run", "dog_stand", "dog_trot",
                    "dog", "humanoid", "humanoid_run", "humanoid_stand"):
            expected = ["Ours", "RLPD"]
            if env in ("dog", "dog_run", "dog_stand", "dog_trot", "humanoid",
                       "humanoid_run", "humanoid_stand"):
                expected.append("TD3+")
            self.assertEqual(list(groups[env]), expected)
            self.assertNotIn("BAFCv6", groups[env])
            for run_dirs in groups[env].values():
                self.assertEqual(len(run_dirs), 4)
                for seed, path in enumerate(run_dirs):
                    self.assertIn("%d" % seed, os.path.basename(path))

        hopper = groups["hopper_hop"]
        self.assertEqual(list(hopper), ["Ours", "RLPD"])
        self.assertEqual(len(hopper["RLPD"]), 3)
        self.assertEqual(len(hopper["Ours"]), 3)
        self.assertIn("nCritic8_utd11", hopper["Ours"][0])
        self.assertIn("nCritic8_utd11", hopper["Ours"][1])
        self.assertIn("nCritic8_utd3_focused", hopper["Ours"][2])
        for seed, path in enumerate(hopper["Ours"]):
            self.assertTrue(path.endswith("_s%d" % seed))

    def test_additional_run_mapping(self):
        groups = plotter.build_additional_run_groups(
            "/ws", "/local", "/server1", "/server2", "/server4")

        hopper = groups["hopper_hop_ncritic"]
        self.assertEqual(list(hopper),
                         ["RLPD", "BAFC_nCritic1", "BAFC_nCritic8"])
        for label, expected_fragment in (
                ("RLPD", "hopper_hop_rlpd"),
                ("BAFC_nCritic1", "nCritic1_utd3_updates12"),
                ("BAFC_nCritic8", "nCritic8_utd3_updates12")):
            self.assertEqual([os.path.basename(path).rsplit("_", 1)[-1]
                              for path in hopper[label]],
                             ["s0", "s2", "s3"])
            self.assertIn(expected_fragment, hopper[label][0])

        aggregate = groups["humanoid_reweight"]
        self.assertEqual(list(aggregate), ["RLPD", "BAFCv3_TR2_reweight"])
        self.assertEqual(len(aggregate["RLPD"]), 4)
        self.assertEqual(len(aggregate["BAFCv3_TR2_reweight"]), 4)
        self.assertTrue(all("server4" in path for path in
                            aggregate["BAFCv3_TR2_reweight"]))

        individual = groups["humanoid_reweight_individual"]
        self.assertEqual(len(individual), 8)
        self.assertTrue(all(len(run_dirs) == 1
                            for run_dirs in individual.values()))

        seed0 = groups["hopper_hop_seed0_v7"]
        self.assertEqual(list(seed0), ["BAFCv7", "RLPD", "Ours"])
        self.assertTrue(all(len(run_dirs) == 1
                            for run_dirs in seed0.values()))
        self.assertIn("lambda010", seed0["BAFCv7"][0])
        self.assertIn("nCritic1_utd3_updates12", seed0["Ours"][0])

    def test_focused_colors_match_standard_algorithm_colors(self):
        self.assertEqual(plotter.FOCUSED_ALGORITHM_COLORS["SAC+"],
                         plotter.ALGORITHM_COLORS["RLPD"])
        self.assertEqual(plotter.FOCUSED_ALGORITHM_COLORS["Ours"],
                         plotter.ALGORITHM_COLORS["Ours"])
        self.assertEqual(plotter.ALGORITHM_COLORS["Ours"], "tab:blue")
        self.assertEqual(plotter.ALGORITHM_COLORS["BAFCv3"], "tab:blue")
        self.assertEqual(plotter.ALGORITHM_COLORS["RLPD"], "tab:orange")

    def test_environment_step_formatter_uses_k_suffix(self):
        self.assertEqual(plotter._format_environment_steps(0, None), "0")
        self.assertEqual(plotter._format_environment_steps(500, None), "500")
        self.assertEqual(plotter._format_environment_steps(50_000, None),
                         "50k")
        self.assertEqual(plotter._format_environment_steps(200_000, None),
                         "200k")


if __name__ == "__main__":
    alf.test.main()

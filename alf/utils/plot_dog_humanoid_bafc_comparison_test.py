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
                                          "/server2")
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
        self.assertEqual(
            set(groups["humanoid_seed01"]),
            {"RLPD", "BAFCv3", "BAFC_TR", "BAFCv6"})
        for run_dirs in groups["humanoid_seed01"].values():
            self.assertEqual(len(run_dirs), 2)
            for seed, path in enumerate(run_dirs):
                self.assertIn("%d" % seed, os.path.basename(path))

    def test_rlpd_ours_mapping_uses_requested_tasks_and_hopper_utds(self):
        groups = plotter.build_rlpd_ours_run_groups(
            "/ws", "/local", "/server1", "/server2")
        self.assertEqual(
            set(groups), {"dog_fetch", "dog_run", "dog_trot", "dog",
                          "humanoid", "hopper_hop"})
        for env in ("dog_fetch", "dog_run", "dog_trot", "dog",
                    "humanoid"):
            self.assertEqual(list(groups[env]), ["Ours", "RLPD"])
            self.assertNotIn("BAFCv6", groups[env])
            for run_dirs in groups[env].values():
                self.assertEqual(len(run_dirs), 4)
                for seed, path in enumerate(run_dirs):
                    self.assertIn("%d" % seed, os.path.basename(path))

        hopper = groups["hopper_hop"]
        self.assertEqual(list(hopper), ["Ours", "RLPD"])
        self.assertEqual(len(hopper["RLPD"]), 3)
        self.assertEqual(len(hopper["Ours"]), 3)
        self.assertIn("critic_utd11", hopper["Ours"][0])
        self.assertIn("critic_utd11", hopper["Ours"][1])
        self.assertIn("critic_utd3", hopper["Ours"][2])
        for seed, path in enumerate(hopper["Ours"]):
            self.assertEqual(os.path.basename(path), "seed_%d" % seed)

    def test_additional_run_mapping(self):
        groups = plotter.build_additional_run_groups(
            "/ws", "/local", "/server1", "/server2", "/server4")

        hopper = groups["hopper_hop_ncritic"]
        self.assertEqual(list(hopper),
                         ["RLPD", "BAFC_nCritic1", "BAFC_nCritic8"])
        for label, expected_fragment in (
                ("RLPD", "critic_utd10"),
                ("BAFC_nCritic1", "num_sampled_critic1"),
                ("BAFC_nCritic8", "num_sampled_critic8")):
            self.assertEqual([os.path.basename(path)
                              for path in hopper[label]],
                             ["seed_0", "seed_2", "seed_3"])
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
        self.assertEqual(list(seed0), ["BAFCv7", "RLPD",
                                      "BAFC_nCritic1", "BAFC_nCritic8"])
        self.assertTrue(all(len(run_dirs) == 1
                            for run_dirs in seed0.values()))
        self.assertIn("lambda010", seed0["BAFCv7"][0])

    def test_focused_colors_swap_rlpd_and_ours(self):
        self.assertEqual(plotter.FOCUSED_ALGORITHM_COLORS["RLPD"],
                         plotter.ALGORITHM_COLORS["Ours"])
        self.assertEqual(plotter.FOCUSED_ALGORITHM_COLORS["Ours"],
                         plotter.ALGORITHM_COLORS["RLPD"])

    def test_environment_step_formatter_uses_k_suffix(self):
        self.assertEqual(plotter._format_environment_steps(0, None), "0")
        self.assertEqual(plotter._format_environment_steps(500, None), "500")
        self.assertEqual(plotter._format_environment_steps(50_000, None),
                         "50k")
        self.assertEqual(plotter._format_environment_steps(200_000, None),
                         "200k")


if __name__ == "__main__":
    alf.test.main()

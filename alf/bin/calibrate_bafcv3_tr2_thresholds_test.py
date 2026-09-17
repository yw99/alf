"""Tests for checkpoint-specific, all-ranks quantile calibration."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

from alf.bin.calibrate_bafcv3_tr2_thresholds import (
    calibrate_threshold, parser, select_thresholds)


class QuantileCalibrationTest(unittest.TestCase):
    def test_default_and_conservative_quantile(self):
        self.assertEqual(parser().parse_args(['--study', 'unused']).threshold_quantile, .33)
        values = [100., 140., 200., 220., 280.]
        conservative = calibrate_threshold(values)
        median = calibrate_threshold(values, .5)
        self.assertAlmostEqual(conservative['eval_trust_max'], 159.2)
        self.assertLess(conservative['eval_trust_max'], median['eval_trust_max'])
        self.assertEqual(conservative['empirical_threshold_pass_fraction'], .4)

    def test_invalid_inputs(self):
        for q in [-.01, 1.01, float('nan'), float('inf')]:
            with self.assertRaises(argparse.ArgumentTypeError):
                calibrate_threshold([1., 2.], q)
        for values in [[], [-1., 2.], [1., float('nan')], [float('inf')], [[1., 2.]]]:
            with self.assertRaises(ValueError):
                calibrate_threshold(values)

    def test_ties_do_not_imply_requested_pass_fraction(self):
        result = calibrate_threshold([4., 4., 4.], .33)
        self.assertEqual(result['eval_trust_max'], 4.)
        self.assertEqual(result['empirical_threshold_pass_fraction'], 1.)

    def fixture(self, root):
        self.cross = []
        jobs, points = [], []
        for seed, scale in [(0, 1.), (1, 10.)]:
            run = f'/example/dog_bafcv3_s{seed}'
            jobs.append({'run': run, 'checkpoint': run + '/train/algorithm/ckpt-75075',
                         'status': 'ok', 'missing_ranks': [], 'expected_ranks': [0, 1]})
            points.append({'run': run, 'checkpoint': 'ckpt-75075', 'task': 'dog:walk',
                           'seed': seed, 'variant': 'same_configuration', 'env_steps': 75000,
                           'repetitions': 3, 'num_ranks': 2})
            for rep, (a, b) in enumerate([(1, 9), (9, 1), (5, 5)]):
                self.cross.append({'run': run, 'checkpoint': 'ckpt-75075', 'env_steps': 75000,
                                   'repetition': rep, 'n': 128, 'ridge': 1e-4,
                                   'rank_min': min(a, b) * scale,
                                   'rank_mean': (a + b) / 2 * scale,
                                   'rank_max': max(a, b) * scale})
        (root / 'manifest.json').write_text(json.dumps({'jobs': jobs}))
        self.write_csv(root / 'checkpoint_means.csv', points)
        self.write_csv(root / 'cross_rank.csv', self.cross)

    def write_csv(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_rank_max_before_quantile_and_separate_seed_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            results = select_thresholds(root)
            self.assertEqual(len(results), 2)
            # Quantile(max across ranks) differs from max(rank quantiles).
            self.assertAlmostEqual(results[0]['eval_trust_max'], 7.64)
            self.assertAlmostEqual(results[1]['eval_trust_max'], 76.4)
            self.assertEqual(results[0]['rollout_skip_sync_mode'], 'min')
            self.assertFalse(results[0]['enable_eval_trust_max_decay'])
            self.assertEqual(results, select_thresholds(root))
            self.assertEqual(len(select_thresholds(root, seeds=[1])), 1)

    def test_missing_requested_checkpoint_is_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'No complete measurements'):
                select_thresholds(root, checkpoints=[75075, 105105])

    def test_incomplete_ranks_and_repetitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['jobs'][0]['missing_ranks'] = [1]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Complete, successful rank inputs'):
                select_thresholds(root)
            self.fixture(root)
            self.write_csv(root / 'cross_rank.csv', self.cross[1:])
            with self.assertRaisesRegex(ValueError, 'Missing or unexpected repetitions'):
                select_thresholds(root)

    def test_duplicate_repetitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.write_csv(root / 'cross_rank.csv', self.cross + [self.cross[0]])
            with self.assertRaisesRegex(ValueError, 'Duplicate repetition'):
                select_thresholds(root)


if __name__ == '__main__':
    unittest.main()

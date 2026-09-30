"""Validate replay sampling and RMS aggregation independently of ALF."""
import unittest

import numpy as np
import torch
from scipy.spatial.distance import pdist
from plot_replay_actor_action_rms import sample_replay, action_pair_mse, pooled_rms


class ReplayActionRmsTest(unittest.TestCase):

    def test_wrapped_partial_buffers_exclude_terminals_and_unused_slots(self):
        obs=torch.arange(10).reshape(2,5,1).float()
        state={'_replay_buffer.time_step|observation':obs,
               '_replay_buffer.time_step|step_type':torch.tensor([[2,1,9,9,1],[0,1,9,9,9]]),
               '_replay_buffer._current_size':torch.tensor([3,2]),
               '_replay_buffer._current_pos':torch.tensor([7,2])}
        sampled,population,indices=sample_replay(state,100,np.random.default_rng(0))
        self.assertEqual(population,4)
        self.assertEqual(set(sampled.flatten().tolist()),{1,4,5,6})
        self.assertEqual(len({tuple(x) for x in indices}),4)
        a=sample_replay(state,2,np.random.default_rng(3))[2]
        b=sample_replay(state,2,np.random.default_rng(3))[2]
        np.testing.assert_array_equal(a,b)

    def test_action_mse_matches_flattened_pair_distances(self):
        actions=torch.from_numpy(np.random.default_rng(4).normal(size=(19,10,7)))
        features=actions.permute(1,0,2).reshape(10,-1).numpy()
        expected=pdist(features,'sqeuclidean')/features.shape[1]
        np.testing.assert_allclose(action_pair_mse(actions),expected)
        changed=actions+2
        np.testing.assert_allclose(action_pair_mse(changed),expected)

    def test_pool_before_sqrt_with_population_weights(self):
        actual=pooled_rms(np.array([[1.,4.],[9.,16.]]),[1,3])
        np.testing.assert_allclose(actual,np.sqrt([7.,13.]))
        self.assertNotEqual(actual[0],(1+3*3)/4)


if __name__=='__main__':
    unittest.main()

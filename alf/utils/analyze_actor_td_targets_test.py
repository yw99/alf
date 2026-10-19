"""Checks for TD analysis replay boundaries, time alignment and head sharing."""
import unittest
import numpy as np
import torch
from alf.utils.analyze_actor_td_targets import sample_transitions, target_moments
from alf.algorithms.bafc_algorithm_v3 import BafcInfo
from alf.algorithms.one_step_loss import OneStepTDLoss


class TargetAnalysisTest(unittest.TestCase):
    def test_wrapped_replay_pairs_exclude_last_reset_and_unpaired_newest(self):
        # Ring chronological order is slots 2,3,4,5,0,1. Only 2->3, 3->4, 0->1 valid.
        fields = {'_current_size': torch.tensor([6]), '_current_pos': torch.tensor([8]),
                  'time_step|step_type': torch.tensor([[0,1,0,1,2,2]])}
        for name in ('observation','reward','discount'):
            fields['time_step|'+name]=torch.arange(6.).reshape(1,6)
        fields['rollout_info|rl|action']=torch.arange(10.,16.).reshape(1,6)
        state={'_replay_buffer.'+k:v for k,v in fields.items()}
        samples,population,indices=sample_transitions(state,100,np.random.default_rng(1))
        self.assertEqual(population,3)
        self.assertEqual(set(map(tuple,indices.tolist())),{(0,2,3),(0,3,4),(0,0,1)})
        np.testing.assert_equal(samples['reward'][1].numpy(),indices[:,2])
        np.testing.assert_equal(samples['action'].numpy(),10+indices[:,1])

    def test_next_reward_terminal_and_time_limit_discount(self):
        info=BafcInfo(reward=torch.tensor([[999.,999.,999.],[2.,3.,4.]]),
                      discount=torch.tensor([[1.,1.,1.],[1.,0.,1.]]),
                      step_type=torch.tensor([[0,1,1],[1,2,2]]))
        values=torch.full((2,3,10,10),10.)
        target=OneStepTDLoss(gamma=.9).compute_td_target(info,values)[0]
        torch.testing.assert_close(target[:,0,0],torch.tensor([11.,3.,13.]))

    def test_shared_target_head_does_not_create_fake_actor_differences(self):
        y=np.broadcast_to(np.arange(10.),(7,10,10)).copy()
        m=target_moments(y,np.zeros_like(y))
        np.testing.assert_equal(m['target_pair_mse'],0)
        np.testing.assert_allclose(m['head_variance'],8.25)
        np.testing.assert_allclose(m['td_error_mse'],28.5)

    def test_target_and_regressed_head_axes_are_independent(self):
        rng=np.random.default_rng(3)
        y,q=rng.normal(size=(2,11,10,10))
        m=target_moments(y,q)
        brute=((y[:,:,:,None]-q[:,:,None,:])**2).mean(axis=(0,2,3))
        np.testing.assert_allclose(m['td_error_mse'],brute)
        i,j=np.triu_indices(10,1)
        np.testing.assert_allclose(m['actor_variance'],m['target_pair_mse'].mean()*9/20)
        self.assertTrue(np.all(m['expected_target_pair_mse']<=m['target_pair_mse']+1e-12))


if __name__=='__main__':unittest.main()

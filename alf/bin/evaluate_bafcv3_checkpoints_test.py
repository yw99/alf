"""Tests for offline BAFC checkpoint diagnostics."""
from pathlib import Path
import tempfile
import os
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from alf.bin.evaluate_bafcv3_checkpoints import (
    TrustAdapter, accumulated_rewards, atomic_json, cache_matches,
    chronological_replay, covariance_scores, discover, fingerprints,
    read_json, replay_windows, stable_seed, tr2_skip_windows, load_model,
    normalize, q_values, tensor_fingerprint, variant_configuration,
    collapse_checkpoint_rows)
from alf.algorithms.bafc_algorithm_v3_tr2 import BafcAlgorithmV3TR2


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def adapter(self):
        a = SimpleNamespace(_trust_cov_reg=1e-4)
        for name in ('_compute_feature_inv_cov', '_compute_weighted_feature_norm', '_compute_eval_trust_from_features'):
            setattr(a,name,getattr(BafcAlgorithmV3TR2,name).__get__(a))
        return a

    def test_analytic_effective_dimension_and_rank_deficiency(self):
        for dimension in [1, 4]:
            phi=torch.eye(dimension).unsqueeze(1)
            # Add an uncovered direction with identically zero observations.
            phi=torch.cat([phi,torch.zeros(dimension,1,1)],-1)
            a=self.adapter()
            rows,eigen=covariance_scores(a,phi,phi,{'target':phi},[1e-4,.01])
            for row in rows:
                expected=dimension/(1+row['ridge']*dimension)
                self.assertAlmostEqual(row['scores']['target'][0],expected,places=4)
                self.assertAlmostEqual(row['effective_dimension'][0],expected,places=8)
                a._trust_cov_reg=row['ridge']
                self.assertAlmostEqual(float(a._compute_eval_trust_from_features(phi,phi)),expected,places=4)

    def test_nullspace_target_has_ridge_penalty(self):
        b=torch.tensor([[[1.,0.]],[[1.,0.]]]);t=torch.tensor([[[0.,1.]]])
        row,_=covariance_scores(self.adapter(),b,b,{'target':t},[.01])
        self.assertAlmostEqual(row[0]['scores']['target'][0],100.,places=3)

    def test_eigh_triangle_retry(self):
        phi = torch.eye(3).unsqueeze(1)
        original = torch.linalg.eigh
        def fail_lower(matrix, UPLO='L'):
            if UPLO == 'L':
                raise torch.linalg.LinAlgError('simulated finite covariance convergence failure')
            return original(matrix, UPLO=UPLO)
        with mock.patch('torch.linalg.eigh', side_effect=fail_lower):
            rows, _ = covariance_scores(self.adapter(), phi, phi, {'target': phi}, [.01])
        self.assertEqual(rows[0]['covariance_solver'], 'torch.linalg.eigh:U_after_L_failure')
        self.assertAlmostEqual(rows[0]['effective_dimension'][0], 3 / 1.03, places=8)

    def test_variant_includes_optimizer_and_excludes_seed(self):
        a = {'Agent.optimizer': 'Adam(lr=0.001)', 'TrainerConfig.random_seed': 0}
        b = dict(a, **{'TrainerConfig.random_seed': 1})
        self.assertEqual(variant_configuration(a), variant_configuration(b))
        b['Agent.optimizer'] = 'Adam(lr=0.002)'
        self.assertNotEqual(variant_configuration(a), variant_configuration(b))

    def test_rank_aggregation_labels_sampling_variation(self):
        rows = [{'target_mean': 4., 'target_sd': 3., 'rank': 0},
                {'target_mean': 12., 'target_sd': 4., 'rank': 1}]
        result = collapse_checkpoint_rows(rows)
        self.assertEqual(result['target_mean'], 8.)
        self.assertEqual(result['sampling_sd_of_rank_mean'], 2.5)
        self.assertEqual(result['num_ranks'], 2)
        self.assertNotIn('rank', result)
        self.assertNotIn('target_sd', result)

    def test_replay_ring_order(self):
        s={'_replay_buffer._current_size':torch.tensor([4,2]),
           '_replay_buffer._current_pos':torch.tensor([6,2]),
           '_replay_buffer.time_step|observation':torch.tensor([[4,5,2,3],[10,11,99,99]])}
        out=chronological_replay(s)
        self.assertEqual(out[0]['time_step|observation'].tolist(),[2,3,4,5])
        self.assertEqual(out[1]['time_step|observation'].tolist(),[10,11])

    def episode(self):
        return {'time_step|step_type':torch.tensor([0,1,2,0,1,2,0,1]),
                'time_step|reward':torch.tensor([99.,2.,3.,99.,5.,6.,99.,8.]),
                'time_step|discount':torch.tensor([1.,1.,0.,1.,1.,1.,1.,1.])}

    def test_windows_and_rewards(self):
        d=self.episode()
        self.assertEqual(replay_windows([d],1),[(0,0,1),(0,1,2),(0,3,4),(0,4,5),(0,6,7)])
        self.assertEqual(replay_windows([d],32),[(0,0,2),(0,1,2),(0,3,5),(0,4,5)])
        reward,factor=accumulated_rewards(d,0,2,.5)
        self.assertEqual(reward,3.5);self.assertEqual(factor,0.)
        reward,factor=accumulated_rewards(d,3,5,.5)
        self.assertEqual(reward,8.);self.assertEqual(factor,.25)

    def test_discovery_atomic_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'server2_copy';root.mkdir()
            run=root/'example';(run/'train/algorithm').mkdir(parents=True)
            (run/'alf_config.py').write_text("import alf\nalf.pre_config({'create_environment.env_name':'dog:walk','TrainerConfig.random_seed':1})\n# BafcAlgorithmV3\n")
            (run/'train/algorithm/ckpt-2').write_text('model')
            (run/'train/algorithm/ckpt-2-optimizer').write_text('optimizer')
            (root/'dog_bafcv3_s2').mkdir()
            tr=root/'dog_bafcv3_tr2_s0';tr.mkdir();(tr/'alf_config.py').write_text('# BafcAlgorithmV3TR2')
            result=discover(str(root))
            self.assertEqual(len(result),2)
            selected=discover(str(root),['dog:walk'],[1],[2])
            self.assertEqual(len(selected),1);self.assertEqual(len(selected[0]['checkpoints']),1)
            self.assertEqual(selected[0]['checkpoints'][0]['shards'],{})
            path=Path(tmp)/'output.json.gz';atomic_json(path,{'x':[1,2]})
            self.assertEqual(read_json(path),{'x':[1,2]});self.assertFalse(Path(str(path)+'.tmp').exists())
            source=run/'train/algorithm/ckpt-2';fp=fingerprints([source])
            saved={'inputs':fp,'options':{'a':1},'code':{'v':1}}
            self.assertTrue(cache_matches(saved,fp,{'a':1},{'v':1}))
            self.assertFalse(cache_matches(saved,fp,{'a':2},{'v':1}))
            source.write_text('changed')
            self.assertFalse(cache_matches(saved,fingerprints([source]),{'a':1},{'v':1}))

    def test_skip_counter_windows(self):
        b='train/BafcAlgorithmV3TR2/'
        curves={'r':{b+'rollout_skip_due_eval_gate_count':[[1,0,0],[2,3,1],[3,0,2]],
                     b+'rollout_opportunity_count':[[1,0,0],[2,6,1],[3,0,2]],
                     'train/Metrics/EnvironmentSteps':[[1,10,0],[2,13,1]]}}
        result=tr2_skip_windows(curves)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['skip_fraction'],.5)
        self.assertEqual(result[0]['env_steps_end'],13)

    def test_deterministic_sampling(self):
        seed=stable_seed(1,'job',3)
        self.assertEqual(np.random.default_rng(seed).choice(100,10,False).tolist(),
                         np.random.default_rng(seed).choice(100,10,False).tolist())
        self.assertNotEqual(seed,stable_seed(1,'job',4))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cpu_gpu_metric_agreement(self):
        g=torch.Generator().manual_seed(3)
        phi=torch.randn(64,2,16,generator=g);phi=phi/phi.norm(dim=-1,keepdim=True)
        target=torch.randn(20,2,16,generator=g);target=target/target.norm(dim=-1,keepdim=True)
        a=self.adapter()
        cpu=a._compute_eval_trust_from_features(target,phi)
        gpu=a._compute_eval_trust_from_features(target.cuda(),phi.cuda()).cpu()
        torch.testing.assert_close(cpu,gpu,rtol=2e-4,atol=2e-4)



@unittest.skipUnless(os.getenv('BAFC_CHECKPOINT_TEST'), 'Set BAFC_CHECKPOINT_TEST for real-checkpoint parity checks')
class CheckpointIntegrationTest(unittest.TestCase):
    def test_forward_and_original_training_target_parity(self):
        from alf.algorithms.bafc_algorithm_v3 import BafcCriticState, BafcInfo
        model=Path(os.environ['BAFC_CHECKPOINT_TEST'])
        run={'config':str(model.parents[2]/'alf_config.py')}
        torch.set_num_threads(2)
        with torch.no_grad():
            alg,norm,meta=load_model(run,str(model),'cpu')
            before=(tensor_fingerprint(alg),tensor_fingerprint(norm))
            a=TrustAdapter(alg); enc=a._compute_actor_encoding(alg._actor_networks)
            raw=torch.load(str(model)+'-replay_buffer-rank0',weights_only=True,map_location='cpu')['algorithm']
            replay=chronological_replay(raw)[0]
            obs=normalize(norm,replay['time_step|observation'][:8],'cpu')
            act=replay['action'][:8]
            current=q_values(alg,enc,obs,act)
            torch.manual_seed(17);target=q_values(alg,enc,obs,act,target=True)
            actions=alg._actor_networks(obs)[0].reshape(-1,act.shape[-1])
            torch.manual_seed(17)
            _,info=alg._critic_train_step(obs,BafcCriticState((),()),BafcInfo(action=act),actions,alg._actor_eval_samples)
            torch.testing.assert_close(current,info.critic,rtol=2e-5,atol=2e-5)
            expected=info.target_critic
            if expected.ndim==2:expected=expected.unsqueeze(-1).expand_as(target)
            torch.testing.assert_close(target,expected,rtol=2e-5,atol=2e-5)
            cpu=a._compute_snapshot_feature_map(obs,enc,act)
            if torch.cuda.is_available():
                alg.cuda();a=TrustAdapter(alg)
                encgpu=a._compute_actor_encoding(alg._actor_networks)
                gpu=a._compute_snapshot_feature_map(obs.cuda(),encgpu,act.cuda()).cpu()
                torch.testing.assert_close(cpu,gpu,rtol=3e-4,atol=3e-5)
                alg.cpu()
            self.assertEqual(before,(tensor_fingerprint(alg),tensor_fingerprint(norm)))

if __name__=='__main__':unittest.main()


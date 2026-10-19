"""Check exact shared/actor-centered loss and gradient decomposition."""
import itertools
import unittest
import torch
from alf.utils.analyze_encoder_td_signal import decompose


class TDSignalTest(unittest.TestCase):
    def test_loss_and_gradient_identity(self):
        torch.manual_seed(3)
        q=torch.randn(7,10,3,dtype=torch.float64,requires_grad=True)
        y=torch.randn(7,10,4,dtype=torch.float64)
        total,common,contrast=decompose(q,y)
        torch.testing.assert_close(total,common+contrast,rtol=1e-12,atol=1e-14)
        gt=torch.autograd.grad(total,q,retain_graph=True)[0]
        gc=torch.autograd.grad(common,q,retain_graph=True)[0]
        gx=torch.autograd.grad(contrast,q)[0]
        torch.testing.assert_close(gt,gc+gx,rtol=1e-12,atol=1e-14)

    def test_shared_target_has_no_actor_target_variance(self):
        q=torch.ones(5,10,2)
        y=torch.arange(4.).view(1,1,4).expand(5,10,4)
        total,common,contrast=decompose(q,y)
        self.assertEqual(float(contrast),0)
        torch.testing.assert_close(total,common)

    def test_permutation_expectation_preserves_same_head_comparison(self):
        torch.manual_seed(7)
        q=torch.randn(5,3,2,dtype=torch.float64)
        y=torch.randn(5,3,4,dtype=torch.float64)
        _,common,_=decompose(q,y)
        expected=common+(q-q.mean(1,keepdim=True)).square().mean()+(y-y.mean(1,keepdim=True)).square().mean()
        brute=torch.stack([decompose(q[:,list(p)],y)[0] for p in itertools.permutations(range(3))]).mean()
        torch.testing.assert_close(expected,brute,rtol=1e-12,atol=1e-14)

    def test_target_head_noise_does_not_change_expected_gradient(self):
        torch.manual_seed(17)
        q=torch.randn(3,10,2,dtype=torch.float64,requires_grad=True)
        y=torch.randn(3,10,4,dtype=torch.float64)
        total,_,_=decompose(q,y)
        learnable=(q-y.mean(2,keepdim=True)).square().mean()
        torch.testing.assert_close(total,learnable+y.var(2,unbiased=False).mean(),rtol=1e-12,atol=1e-14)
        gt=torch.autograd.grad(total,q,retain_graph=True)[0]
        gl=torch.autograd.grad(learnable,q)[0]
        torch.testing.assert_close(gt,gl,rtol=1e-12,atol=1e-14)


if __name__=='__main__':unittest.main()

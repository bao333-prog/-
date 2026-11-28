"""额外EMA验证不能消耗live训练后续使用的CPU随机数。"""
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

class ValidationRngTests(unittest.TestCase):
    def test_actual_loader_iteration_does_not_change_global_rng(self):
        from rng_utils import isolated_validation_rng
        loader=DataLoader(TensorDataset(torch.ones(4,2)),batch_size=2,shuffle=False,num_workers=0)
        torch.manual_seed(24);before=torch.get_rng_state().clone()
        with isolated_validation_rng():list(loader)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_exception_also_restores_rng(self):
        from rng_utils import isolated_validation_rng
        torch.manual_seed(25);before=torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with isolated_validation_rng():torch.rand(10);raise RuntimeError('expected')
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

if __name__=='__main__':unittest.main()

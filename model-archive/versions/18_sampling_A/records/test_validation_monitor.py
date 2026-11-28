"""验证日志必须正确统计错类、缺类和非有限输出。"""
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from validation_monitor import validate_with_class_metrics


class ValidationMonitorChecks(unittest.TestCase):
    def test_counts_macro_and_loss_match_known_predictions(self):
        logits=torch.tensor([[3.,1.,0.],[3.,1.,0.],[1.,3.,0.],[1.,3.,0.]])
        labels=torch.tensor([0,1,0,1])
        loader=DataLoader(TensorDataset(logits,labels),batch_size=3)
        loss,accuracy,stats=validate_with_class_metrics(torch.nn.Identity(),loader,torch.nn.CrossEntropyLoss(),torch.device('cpu'),['A','B','C'])
        self.assertAlmostEqual(loss,float(torch.nn.functional.cross_entropy(logits,labels)),places=6)
        self.assertEqual(accuracy,.5)
        self.assertEqual(stats['per_class']['A'],dict(correct=1,total=2))
        self.assertEqual(stats['per_class']['B'],dict(correct=1,total=2))
        self.assertEqual(stats['per_class']['C'],dict(correct=0,total=0))
        self.assertEqual(stats['macro_accuracy'],.5)
        self.assertEqual(stats['evaluated_classes'],2)

    def test_nonfinite_prediction_stops_validation(self):
        loader=DataLoader(TensorDataset(torch.tensor([[float('nan'),0.]]),torch.tensor([0])),batch_size=1)
        with self.assertRaises(FloatingPointError):
            validate_with_class_metrics(torch.nn.Identity(),loader,torch.nn.CrossEntropyLoss(),torch.device('cpu'),['A','B'])


if __name__=='__main__': unittest.main()

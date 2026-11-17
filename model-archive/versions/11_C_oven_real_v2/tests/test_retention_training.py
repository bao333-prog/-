"""旧知识约束的风险测试：新类抢占、教师梯度/BN、集合隔离和完整主体。"""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from oven_food import augment, model, train
from oven_food.config import ExperimentConfig, ConfigError, load_experiment_config
from oven_food.manifest import ManifestRecord, write_manifest, read_manifest


class RetentionTrainingTests(unittest.TestCase):
    def helpers(self):
        self.assertIsNotNone(importlib.util.find_spec('oven_food.continual'),'旧类训练约束模块尚未实现')
        return importlib.import_module('oven_food.continual')

    def test_distillation_penalizes_new_class_probability_and_detaches_teacher(self):
        helper=self.helpers()
        teacher=torch.tensor([[4.,0.],[4.,0.]],requires_grad=True)
        low=torch.tensor([[4.,0.,-100.],[4.,0.,-100.]],requires_grad=True)
        high=torch.tensor([[4.,0.,8.],[4.,0.,8.]],requires_grad=True)
        # 第二张是新增类别，不能要求它模仿旧教师。
        labels=torch.tensor([0,2])
        reference,n=helper.old_class_distillation_loss(low,teacher,labels,temperature=2.)
        actual,m=helper.old_class_distillation_loss(high,teacher,labels,temperature=2.)
        self.assertEqual((n,m),(1,1))
        self.assertLess(abs(float(reference.detach())),1e-5)
        self.assertGreater(float(actual.detach()),1.)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(high.grad[0,2]),0)
        torch.testing.assert_close(high.grad[1],torch.zeros(3))

    def test_wrong_teacher_and_all_new_batches_produce_differentiable_zero(self):
        helper=self.helpers()
        for labels in [torch.tensor([1,1]),torch.tensor([2,2])]:
            student=torch.randn(2,3,requires_grad=True)
            loss,count=helper.old_class_distillation_loss(student,torch.tensor([[4.,0.],[4.,0.]]),labels,temperature=2.)
            self.assertEqual(count,0);self.assertEqual(float(loss.detach()),0.)
            loss.backward();torch.testing.assert_close(student.grad,torch.zeros_like(student))
        with self.assertRaisesRegex(ValueError,'temperature'):
            helper.old_class_distillation_loss(torch.zeros(1,3),torch.zeros(1,2),torch.tensor([0]),temperature=0)

    def test_optimizer_groups_have_scaled_rates_and_no_duplicate_or_frozen_parameters(self):
        helper=self.helpers()
        net=model.build_model('mobilenet_v3_small',num_classes=3,pretrained=False)
        groups=helper.optimizer_parameter_groups(net,learning_rate=.001,backbone_lr_scale=.2)
        self.assertEqual([g['lr'] for g in groups],[.0002,.001])
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in net.parameters() if p.requires_grad})
        for p in net.features.parameters():p.requires_grad_(False)
        groups=helper.optimizer_parameter_groups(net,learning_rate=.001,backbone_lr_scale=.2)
        self.assertEqual(len(groups),1)
        self.assertEqual({id(p) for g in groups for p in g['params']},{id(p) for p in net.classifier.parameters()})

    def test_configuration_defaults_and_invalid_new_options(self):
        self.assertIn('distillation_weight',ExperimentConfig.__dataclass_fields__)
        defaults=ExperimentConfig(mode='train',experiment_name='x',manifest='x',data_root='.',output_dir='x')
        self.assertEqual(defaults.distillation_weight,0.)
        self.assertEqual(defaults.backbone_lr_scale,1.)
        self.assertEqual(defaults.lr_schedule,'none')
        base=dict(mode='train',experiment_name='x',manifest='x',data_root='.',output_dir='x',
                  train_environments=['proxy'],validation_environments=['proxy'],checkpoint='old.pt',new_classes=['new'],
                  distillation_weight=1.,distillation_temperature=2.,backbone_lr_scale=.2,lr_schedule='cosine')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json';path.write_text(json.dumps(base),encoding='utf-8')
            self.assertEqual(load_experiment_config(path).distillation_weight,1.)
            for key,value in [('distillation_weight',-1),('distillation_weight',float('nan')),('distillation_temperature',0),
                              ('backbone_lr_scale',0),('backbone_lr_scale',2),('lr_schedule','unknown')]:
                bad={**base,key:value};path.write_text(json.dumps(bad),encoding='utf-8')
                with self.subTest(key=key,value=value),self.assertRaises(ConfigError):load_experiment_config(path)
            path.write_text(json.dumps({**base,'new_classes':[]}),encoding='utf-8')
            with self.assertRaisesRegex(ConfigError,'distillation.*increment'):
                load_experiment_config(path)

    def test_new_augmentation_preserves_both_ends_transparency_and_seed(self):
        self.assertIn('gentle_retention',augment.PROFILES)
        image=Image.new('RGBA',(300,50),(0,0,0,0))
        image.paste((255,0,0,255),(0,0,45,50));image.paste((0,0,255,255),(255,0,300,50))
        for seed in range(20):
            first=augment.ProxyAugmenter('gentle_retention',seed=seed,resize_mode='letterbox')(image)
            second=augment.ProxyAugmenter('gentle_retention',seed=seed,resize_mode='letterbox')(image)
            p=np.asarray(first);np.testing.assert_array_equal(p,np.asarray(second))
            self.assertEqual(first.size,(224,224));self.assertEqual(first.mode,'RGB')
            self.assertTrue(((p[:,:,0]>180)&(p[:,:,2]<80)).any())
            self.assertTrue(((p[:,:,2]>180)&(p[:,:,0]<80)).any())
        with self.assertRaisesRegex(ValueError,'letterbox'):
            augment.ProxyAugmenter('gentle_retention')

    def test_real_training_keeps_teacher_frozen_and_saves_only_student(self):
        self.assertIn('distillation_weight',ExperimentConfig.__dataclass_fields__)
        self.assertIn('gentle_retention',augment.PROFILES)
        previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,previous_threads)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);net=model.build_model('mobilenet_v3_small',num_classes=2,pretrained=False)
            parent=root/'B.pt';torch.save(dict(model_state=net.state_dict(),model_name='mobilenet_v3_small',
                        class_names=['old_a','old_b'],image_size=32,preprocessing={'resize_mode':'letterbox'}),parent)
            digest=hashlib.sha256(parent.read_bytes()).hexdigest()
            records=[]
            for i,label in enumerate(['old_a']*3+['old_b']*3+['new']*2+['old_a','old_b','new']):
                path=root/f'{i}.png';Image.new('RGB',(48,32),(40+i*10,90,130)).save(path)
                records.append(ManifestRecord(str(path),label,'proxy',str(i),'unknown','unknown','none','none','train' if i<8 else 'val'))
            manifest=root/'manifest.csv';write_manifest(manifest,records)
            config=ExperimentConfig(mode='train',experiment_name='kd',manifest=str(manifest),data_root=temp,output_dir=str(root/'train'),
                    checkpoint=str(parent),new_classes=('new',),replay_per_class=2,train_environments=('proxy',),validation_environments=('proxy',),
                    image_size=32,batch_size=2,epochs=2,device='cpu',pretrained=False,augmentation='gentle_retention',resize_mode='letterbox',
                    distillation_weight=1.,distillation_temperature=2.,backbone_lr_scale=.2,lr_schedule='cosine')
            captured=[];original_copy=copy.deepcopy
            def capture(value):
                result=original_copy(value)
                if isinstance(result,torch.nn.Module):captured.append((result,{k:v.clone() for k,v in result.state_dict().items()}))
                return result
            # 只替换训练模块的绑定，不能改全局copy.deepcopy，否则会拦截torch内部的memo调用。
            with patch('oven_food.train.copy',SimpleNamespace(deepcopy=capture)):
                path=train.train_experiment(config)
            self.assertEqual(len(captured),1)
            teacher,before=captured[0]
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            for key,value in before.items():self.assertTrue(torch.equal(value,teacher.state_dict()[key]),key)
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            self.assertEqual(checkpoint['distillation']['teacher_sha256'],digest)
            self.assertFalse(any('teacher' in k for k in checkpoint['model_state']))
            self.assertEqual(checkpoint['model_state']['classifier.3.weight'].shape[0],3)
            history=json.loads((root/'train/history.json').read_text(encoding='utf-8'))
            self.assertTrue(any(h['distillation_examples']>0 and h['distillation_loss']>0 for h in history))
            self.assertLess(history[1]['head_learning_rate'],history[0]['head_learning_rate'])
            self.assertAlmostEqual(history[0]['backbone_learning_rate'],.2*history[0]['head_learning_rate'])
            self.assertEqual(sum(r.split=='train' for r in read_manifest(root/'train/training_manifest.csv')),6)
            self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()

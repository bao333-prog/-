"""覆盖真实风险：主体裁掉、标签不平衡、最佳模型同分及预处理不一致。"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from oven_food import augment, model, train
from oven_food.config import ConfigError, load_experiment_config
from oven_food.manifest import ManifestRecord


def record(label):
    return ManifestRecord("unused.png",label,"proxy",label,"unknown","unknown","none","none","train")


class TrainingImprovementTests(unittest.TestCase):
    def test_letterbox_preserves_both_ends_and_flattens_transparency_to_white(self):
        self.assertTrue(hasattr(augment,"prepare_image"),"统一预处理函数尚未实现")
        image=Image.new("RGBA",(200,40),(0,0,0,0))
        image.paste((255,0,0,255),(0,0,30,40))
        image.paste((0,0,255,255),(170,0,200,40))
        output=augment.prepare_image(image,image_size=100,resize_mode="letterbox")
        pixels=np.asarray(output)
        self.assertEqual(output.mode,"RGB")
        self.assertEqual(output.size,(100,100))
        self.assertEqual(output.getpixel((50,50)),(255,255,255))
        self.assertTrue(((pixels[:,:,0]>200)&(pixels[:,:,2]<50)).any())
        self.assertTrue(((pixels[:,:,2]>200)&(pixels[:,:,0]<50)).any())

    def test_default_resize_matches_legacy_center_crop(self):
        self.assertTrue(hasattr(augment,"prepare_image"))
        from PIL import ImageOps
        image=Image.fromarray(np.arange(40*80*3,dtype=np.uint8).reshape(40,80,3))
        expected=ImageOps.fit(image,(24,24),method=Image.Resampling.BILINEAR)
        np.testing.assert_array_equal(np.asarray(augment.prepare_image(image,image_size=24,resize_mode="center_crop")),np.asarray(expected))

    def test_standard_and_targeted_letterbox_share_white_alpha_handling(self):
        image=Image.new("RGBA",(80,40),(0,0,0,0));image.paste((220,30,30,255),(10,10,70,30))
        white=Image.alpha_composite(Image.new("RGBA",image.size,"white"),image).convert("RGB")
        for profile in ["standard","targeted"]:
            with self.subTest(profile=profile):
                actual=augment.ProxyAugmenter(profile,image_size=32,resize_mode="letterbox",seed=7)(image)
                expected=augment.ProxyAugmenter(profile,image_size=32,resize_mode="letterbox",seed=7)(white)
                np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))

    def test_gentle_augmentation_keeps_both_subject_ends_and_is_seeded(self):
        self.assertIn("gentle",augment.PROFILES)
        image=Image.new("RGB",(300,50),"white")
        image.paste((255,0,0),(0,0,45,50));image.paste((0,0,255),(255,0,300,50))
        outputs=[np.asarray(augment.ProxyAugmenter("gentle",image_size=80,seed=42,resize_mode="letterbox")(image)) for _ in range(2)]
        np.testing.assert_array_equal(*outputs)
        p=outputs[0]
        self.assertTrue(((p[:,:,0]>180)&(p[:,:,2]<70)).any())
        self.assertTrue(((p[:,:,2]>180)&(p[:,:,0]<70)).any())

    def test_class_weights_give_equal_total_mass_per_training_class(self):
        self.assertTrue(hasattr(train,"_class_loss_weights"))
        weights=train._class_loss_weights([record("common")]*3+[record("rare")],["rare","common"])
        self.assertAlmostEqual(weights[0],2)
        self.assertAlmostEqual(weights[1],2/3)
        self.assertAlmostEqual(weights[0],3*weights[1])
        with self.assertRaisesRegex(ValueError,"training samples"):
            train._class_loss_weights([record("common")],["rare","common"])

    def test_weighted_epoch_loss_does_not_depend_on_batch_partition(self):
        import torch
        self.assertTrue(hasattr(train,"_loss_totals"))
        logits=torch.tensor([[3.,0.],[0.,2.],[1.,0.],[0.,1.]])
        labels=torch.tensor([0,1,0,0]);weights=torch.tensor([2/3,2.])
        expected=float(torch.nn.functional.cross_entropy(logits,labels,weight=weights))
        for boundaries in [(0,1,4),(0,2,4),(0,3,4)]:
            numerator=denominator=0.
            for start,stop in zip(boundaries,boundaries[1:]):
                loss=torch.nn.functional.cross_entropy(logits[start:stop],labels[start:stop],weight=weights)
                total,mass=train._loss_totals(loss,labels[start:stop],weights)
                numerator+=total;denominator+=mass
            self.assertAlmostEqual(numerator/denominator,expected,places=6)

    def test_accuracy_ties_choose_lower_loss_and_ignore_nan(self):
        self.assertTrue(hasattr(train,"_is_better_checkpoint"))
        self.assertTrue(train._is_better_checkpoint(1.0,0.01,1.0,0.03))
        self.assertFalse(train._is_better_checkpoint(1.0,0.05,1.0,0.03))
        self.assertFalse(train._is_better_checkpoint(0.99,0.001,1.0,0.03))
        self.assertFalse(train._is_better_checkpoint(float("nan"),0.01,1.0,0.03))
        self.assertFalse(train._is_better_checkpoint(1.0,float("nan"),1.0,0.03))

    def test_old_and_formal_A_checkpoints_resolve_preprocessing(self):
        self.assertTrue(hasattr(model,"checkpoint_resize_mode"))
        self.assertEqual(model.checkpoint_resize_mode({}),"center_crop")
        self.assertEqual(model.checkpoint_resize_mode({"preprocessing":{"resize":"aspect_ratio_pad"}}),"letterbox")
        self.assertEqual(model.checkpoint_resize_mode({"preprocessing":{"resize_mode":"letterbox"}}),"letterbox")
        with self.assertRaisesRegex(ValueError,"resize"):
            model.checkpoint_resize_mode({"preprocessing":{"resize_mode":"invalid"}})

    def test_new_config_options_and_invalid_resize(self):
        config=dict(mode="train",experiment_name="new",manifest="unused.csv",data_root=".",output_dir="unused",
                    train_environments=["proxy"],validation_environments=["proxy"],
                    augmentation="gentle",resize_mode="letterbox",class_balanced_loss=True)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"config.json";path.write_text(json.dumps(config),encoding="utf-8")
            try:
                parsed=load_experiment_config(path)
            except ConfigError as exc:
                self.fail(str(exc))
            self.assertTrue(parsed.class_balanced_loss)
            self.assertEqual(parsed.resize_mode,"letterbox")
            config["resize_mode"]="stretch_unknown";path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaisesRegex(ConfigError,"resize_mode"):
                load_experiment_config(path)

    def test_gentle_requires_subject_preserving_resize(self):
        config=dict(mode="train",experiment_name="new",manifest="unused.csv",data_root=".",output_dir="unused",
                    train_environments=["proxy"],validation_environments=["proxy"],augmentation="gentle")
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"config.json";path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaisesRegex(ConfigError,"gentle.*letterbox"):
                load_experiment_config(path)
        with self.assertRaisesRegex(ValueError,"gentle.*letterbox"):
            augment.ProxyAugmenter("gentle",resize_mode="center_crop")

    def test_real_training_checkpoint_and_evaluation_use_same_raw_image_preprocessing(self):
        import torch
        from oven_food.config import ExperimentConfig
        from oven_food.evaluate import evaluate_experiment
        from oven_food.manifest import write_manifest
        from oven_food.dataset import FoodImageDataset
        previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,previous_threads)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);records=[]
            for i,label in enumerate(["common","common","common","rare","common","rare"]):
                path=root/f"{i}.png"
                image=Image.new("RGBA",(70,20),(0,0,0,0))
                image.paste((200,30,30,255) if label=="common" else (30,180,30,255),(0,2,70,18));image.save(path)
                records.append(ManifestRecord(str(path),label,"proxy",str(i),"unknown","unknown","none","none","train" if i<4 else "val"))
            manifest=root/"manifest.csv";write_manifest(manifest,records)
            config=ExperimentConfig(mode="train",experiment_name="integration",manifest=str(manifest),data_root=temp,
                                    output_dir=str(root/"train"),train_environments=("proxy",),validation_environments=("proxy",),
                                    augmentation="gentle",resize_mode="letterbox",class_balanced_loss=True,
                                    epochs=2,batch_size=2,image_size=32,pretrained=False,device="cpu")
            checkpoint_path=train.train_experiment(config)
            checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["preprocessing"]["resize_mode"],"letterbox")
            np.testing.assert_allclose(checkpoint["class_loss_weights"],[2/3,2])
            history=json.loads((root/"train/history.json").read_text(encoding="utf-8"))
            expected=min(history,key=lambda h:(-h["validation_accuracy"],h["validation_loss"]))
            self.assertEqual(checkpoint["epoch"],expected["epoch"])
            evaluation=ExperimentConfig(mode="evaluate",experiment_name="reload",manifest=str(manifest),data_root=temp,
                                        output_dir=str(root/"evaluation"),checkpoint=str(checkpoint_path),
                                        evaluation_environments=("proxy",),evaluation_split="val",batch_size=2,device="cpu")
            metrics=json.loads(evaluate_experiment(evaluation).read_text(encoding="utf-8"))
            self.assertEqual(metrics["num_samples"],2)
            self.assertEqual(metrics["overall_accuracy"],checkpoint["validation_accuracy"])
            first=FoodImageDataset(records[:1],data_root=temp,image_size=32,resize_mode="letterbox")
            first.reseed(123)
            self.assertEqual(first.augmenter.resize_mode,"letterbox")


if __name__=="__main__":
    unittest.main()

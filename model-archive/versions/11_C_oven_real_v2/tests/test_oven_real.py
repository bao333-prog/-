"""固定机位增强保护小目标，训练配置兼容旧模型。"""
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
from oven_food import augment
from oven_food.config import load_experiment_config, ConfigError


class OvenRealTests(unittest.TestCase):
    def test_training_uses_smoothing_only_for_training_loss(self):
        import torch
        from oven_food.config import ExperimentConfig
        from oven_food.manifest import ManifestRecord, write_manifest
        from oven_food.train import train_experiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for label, color in [("one","red"),("two","blue")]:
                for split in ["train","val"]:
                    path = root / (label+split+".png")
                    Image.new("RGB",(32,32),color).save(path)
                    records.append(ManifestRecord(str(path),label,"proxy",label+split,
                                                  "fixed","oven","none","none",split))
            manifest = root/"manifest.csv"
            write_manifest(manifest,records)
            model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                        torch.nn.Flatten(),torch.nn.Linear(3,2))
            constructor = torch.nn.CrossEntropyLoss
            calls = []
            def capture(*args, **kwargs):
                calls.append(kwargs.copy())
                return constructor(*args, **kwargs)
            config = ExperimentConfig(mode="train",experiment_name="smooth",
                                      manifest=str(manifest),data_root=directory,
                                      output_dir=str(root/"result"),train_environments=("proxy",),
                                      validation_environments=("proxy",),image_size=32,
                                      epochs=1,batch_size=2,device="cpu",pretrained=False,
                                      label_smoothing=.05)
            with patch("oven_food.train.build_model",return_value=model), patch.object(torch.nn,"CrossEntropyLoss",side_effect=capture):
                checkpoint = train_experiment(config)
            self.assertEqual(calls[0]["label_smoothing"],.05)
            self.assertEqual(calls[1].get("label_smoothing",0),0)
            self.assertEqual(torch.load(checkpoint,weights_only=True)["config"]["label_smoothing"],.05)

    def test_illumination_changes_light_without_moving_pixels(self):
        image = Image.new("RGB", (64, 32), (80, 80, 80))
        image.paste((220, 20, 20), (28, 12, 36, 20))
        original = np.asarray(image).copy()
        output = np.asarray(augment.vary_illumination(image, left_gain=.8, right_gain=1.2))
        self.assertLess(output[0, 0, 0], output[0, -1, 0])
        red = (output[:,:,0] > 2*output[:,:,1].astype(float))
        self.assertEqual(int(red.sum()), 64)
        np.testing.assert_array_equal(np.argwhere(red).min(0), [12,28])
        np.testing.assert_array_equal(original, np.asarray(image))
        for value in [0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                augment.vary_illumination(image, left_gain=value, right_gain=1)

    def test_profile_keeps_padding_and_small_subject_size(self):
        image = Image.new("RGB", (224,224), "white")
        image.paste((20,20,20),(0,49,224,175))
        image.paste((220,30,30),(100,100,124,124))
        for seed in range(12):
            one = augment.ProxyAugmenter("oven_real",seed=seed,resize_mode="letterbox")(image)
            two = augment.ProxyAugmenter("oven_real",seed=seed,resize_mode="letterbox")(image)
            pixels = np.asarray(one)
            np.testing.assert_array_equal(pixels,np.asarray(two))
            self.assertTrue(np.all(pixels[:49]==255))
            self.assertTrue(np.all(pixels[175:]==255))
            red = (pixels[:,:,0]>90) & (pixels[:,:,0]>2*pixels[:,:,1].astype(float))
            self.assertGreater(red.sum(),450)
            position = np.argwhere(red)
            self.assertGreaterEqual(position[:,0].min(),99)
            self.assertLessEqual(position[:,0].max(),124)

    def test_config_defaults_and_smoothing_validation(self):
        config = dict(mode="train",experiment_name="real",manifest="unused",
                      data_root=".",output_dir="unused",train_environments=["target"],
                      validation_environments=["target"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"config.json"
            path.write_text(json.dumps(config),encoding="utf-8")
            self.assertEqual(load_experiment_config(path).label_smoothing,0)
            config.update(augmentation="oven_real",resize_mode="letterbox",label_smoothing=.05)
            path.write_text(json.dumps(config),encoding="utf-8")
            self.assertEqual(load_experiment_config(path).label_smoothing,.05)
            for value in [-.1,1,True,float("nan")]:
                config["label_smoothing"]=value
                path.write_text(json.dumps(config),encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_experiment_config(path)
        with self.assertRaisesRegex(ValueError,"letterbox"):
            augment.ProxyAugmenter("oven_real")


if __name__=="__main__":
    unittest.main()

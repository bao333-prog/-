"""增量回归：旧类别下标和权重必须保留，回放只能来自旧类训练集。"""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from oven_food import model, train
from oven_food.config import ConfigError, load_experiment_config
from oven_food.manifest import ManifestRecord


def record(label,index,split="train"):
    return ManifestRecord(f"{label}_{index}.png",label,"proxy",f"{label}_{index}","unknown","unknown","none","none",split)


class IncrementalTests(unittest.TestCase):
    def test_new_classes_append_without_reordering_old_classes(self):
        self.assertTrue(hasattr(model,"incremental_class_names"))
        checkpoint={"class_names":["toast","corn"]}
        self.assertEqual(model.incremental_class_names(checkpoint,["salmon","asparagus"],observed_labels={"toast","salmon"}),["toast","corn","salmon","asparagus"])
        for extra in [["corn"],["salmon","salmon"],[],[""]]:
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                model.incremental_class_names(checkpoint,extra)
        with self.assertRaisesRegex(ValueError,"unknown labels"):
            model.incremental_class_names(checkpoint,["salmon"],observed_labels={"chicken"})

    def test_expansion_preserves_old_logits_and_backbone_parameters(self):
        self.assertTrue(hasattr(model,"expand_classifier"))
        torch.manual_seed(42)
        network=model.build_model("mobilenet_v3_small",num_classes=2,pretrained=False).eval()
        old_weight=network.classifier[-1].weight.detach().clone();old_bias=network.classifier[-1].bias.detach().clone()
        old_backbone=network.features[0][0].weight.detach().clone()
        x=torch.randn(2,3,32,32)
        with torch.inference_mode():old_logits=network(x)
        model.expand_classifier(network,num_classes=4)
        self.assertTrue(torch.equal(network.classifier[-1].weight[:2],old_weight))
        self.assertTrue(torch.equal(network.classifier[-1].bias[:2],old_bias))
        self.assertTrue(torch.equal(network.features[0][0].weight,old_backbone))
        self.assertFalse(network.classifier[-1].training)
        with torch.inference_mode():new_logits=network(x)
        self.assertEqual(tuple(new_logits.shape),(2,4))
        torch.testing.assert_close(old_logits,new_logits[:,:2],atol=1e-6,rtol=1e-5)
        with self.assertRaises(ValueError):model.expand_classifier(network,num_classes=3)

    def test_replay_is_seeded_bounded_and_uses_only_training_records(self):
        self.assertTrue(hasattr(train,"_select_replay_records"))
        records=[record(c,i) for c in ["old_a","old_b","new"] for i in range(8)]
        one=train._select_replay_records(records,["old_a","old_b"],per_class=3,seed=42)
        two=train._select_replay_records(records,["old_a","old_b"],per_class=3,seed=42)
        self.assertEqual(one,two);self.assertEqual(len(one),6)
        self.assertEqual({r.food_label for r in one},{"old_a","old_b"})
        self.assertEqual(len({r.session_id for r in one}),6)
        with self.assertRaisesRegex(ValueError,"training"):
            train._select_replay_records(records+[record("old_a",99,"val")],["old_a"],per_class=3,seed=42)
        with self.assertRaisesRegex(ValueError,"no training"):
            train._select_replay_records(records,["missing"],per_class=3,seed=42)

    def test_incremental_config_requires_checkpoint_and_positive_replay_budget(self):
        config=dict(mode="train",experiment_name="B",manifest="unused.csv",data_root=".",output_dir="unused",
                    train_environments=["proxy"],validation_environments=["proxy"],new_classes=["salmon"],replay_per_class=20)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"config.json";path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaisesRegex(ConfigError,"checkpoint"):load_experiment_config(path)
            config["checkpoint"]="parent.pt";config["replay_per_class"]=0;path.write_text(json.dumps(config),encoding="utf-8")
            with self.assertRaisesRegex(ConfigError,"replay_per_class"):load_experiment_config(path)
            config["replay_per_class"]=20;path.write_text(json.dumps(config),encoding="utf-8")
            parsed=load_experiment_config(path)
            self.assertEqual(parsed.new_classes,("salmon",))
            self.assertEqual(parsed.replay_per_class,20)

    def test_real_incremental_training_retains_parent_order_and_audits_replay(self):
        from dataclasses import replace
        import hashlib
        from PIL import Image
        from oven_food.config import ExperimentConfig
        from oven_food.manifest import write_manifest,read_manifest
        previous_threads=torch.get_num_threads();torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,previous_threads)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent_model=model.build_model("mobilenet_v3_small",num_classes=2,pretrained=False)
            parent_path=root/"A.pt"
            torch.save({"model_state":parent_model.state_dict(),"model_name":"mobilenet_v3_small",
                        "class_names":["old_b","old_a"],"image_size":32,"preprocessing":{"resize_mode":"letterbox"}},parent_path)
            parent_hash=hashlib.sha256(parent_path.read_bytes()).hexdigest()
            records=[]
            labels=["old_a"]*3+["old_b"]*3+["new"]*2+["old_a","old_b","new"]
            for i,label in enumerate(labels):
                path=root/f"{i}.png";Image.new("RGB",(48,32),(i*20,40,80)).save(path)
                records.append(ManifestRecord(str(path),label,"proxy",str(i),"unknown","unknown","none","none","train" if i<8 else "val"))
            manifest=root/"manifest.csv";write_manifest(manifest,records)
            config=ExperimentConfig(mode="train",experiment_name="B",manifest=str(manifest),data_root=temp,
                                    output_dir=str(root/"B"),checkpoint=str(parent_path),train_environments=("proxy",),validation_environments=("proxy",),
                                    new_classes=("new",),replay_per_class=1,epochs=1,image_size=32,batch_size=2,
                                    augmentation="gentle",resize_mode="letterbox",class_balanced_loss=True,device="cpu")
            checkpoint_path=train.train_experiment(config)
            checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["class_names"],["old_b","old_a","new"])
            self.assertEqual(checkpoint["incremental"]["replay_counts"],{"old_b":1,"old_a":1})
            self.assertEqual(checkpoint["incremental"]["parent_sha256"],parent_hash)
            actual=read_manifest(root/"B/training_manifest.csv")
            self.assertEqual(sum(r.split=="train" for r in actual),4)
            self.assertEqual(sum(r.split=="val" for r in actual),3)
            self.assertEqual(hashlib.sha256(parent_path.read_bytes()).hexdigest(),parent_hash)
            with self.assertRaisesRegex(ValueError,"new classes need training"):
                train.train_experiment(replace(config,new_classes=("new","missing"),class_balanced_loss=False,output_dir=str(root/"bad")))


if __name__=="__main__":unittest.main()

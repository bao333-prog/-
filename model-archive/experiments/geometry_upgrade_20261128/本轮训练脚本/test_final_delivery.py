"""最终选模及交付的合成小夹具；不会读取本轮真实评测图或旧报告。"""
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def load_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class FinalDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.finalize = load_module('finalize_models')
        self.delivery = load_module('build_delivery')
        self.predict = load_module('predict')
        self.assertIsNotNone(self.finalize, '尚未实现最终选模脚本')
        self.assertIsNotNone(self.delivery, '尚未实现独立交付脚本')
        self.assertIsNotNone(self.predict, '尚未实现独立预测脚本')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'run'
        self.root.mkdir()

    def fixture(self):
        import torch
        for arm, correct in [('A224', 1), ('B320', 2), ('C320', 1)]:
            p = self.root / arm
            checkpoint = p / 'training/best_raw.pt'
            checkpoint.parent.mkdir(parents=True)
            stats = {'correct': correct, 'total': 2,
                     'per_class': {name:{'correct':correct if name=='玉米' else 0,
                                         'total':2 if name=='玉米' else 0} for name in self.predict.CLASS_NAMES}}
            candidate = {'path': str(checkpoint.resolve()), 'epoch': 1, 'loss': .5,
                         'stats': stats, 'key': [correct/2] * 3 + [-.5]}
            torch.save({'class_names':self.predict.CLASS_NAMES,'image_size':224 if arm=='A224' else 320,
                'preprocessing':{'resize_mode':'letterbox'},'variant':'raw','epoch':1,
                'selection_key':candidate['key'],'selection_stats':stats,'validation_loss':.5},checkpoint)
            digest = self.delivery.sha(checkpoint)
            dump(p / 'selection.json', dict(candidate, checkpoint_sha256=digest, test_used=False))
            dump(p / 'best_candidates.json', {'raw': candidate})
            dump(p / 'execution.json', {'status': 'completed', 'epochs': 1,
                 'selected': str(checkpoint.resolve()), 'checkpoint_sha256': digest,
                 'test_used_for_selection': False})
        return self.root

    def test_selects_global_by_validation_key_and_refuses_repeat(self):
        root = self.fixture()
        chosen, arms = self.finalize.choose_from_completed_arms(root)
        self.assertEqual(chosen['arm'], 'B320')
        self.assertEqual(set(arms), {'A224', 'B320', 'C320'})
        dump(root / 'final_evaluation.json', {})
        with self.assertRaises(RuntimeError):
            self.finalize.choose_from_completed_arms(root)

    def test_incomplete_or_tampered_selection_is_rejected(self):
        root = self.fixture()
        dump(root / 'C320/execution.json', {'status': 'running'})
        with self.assertRaises(RuntimeError):
            self.finalize.choose_from_completed_arms(root)
        self.fixture_reset(root)
        selection = json.loads((root/'B320/selection.json').read_text())
        selection['key'] = [.2, .2, .2, -.5]
        dump(root/'B320/selection.json', selection)
        with self.assertRaises(ValueError):
            self.finalize.choose_from_completed_arms(root)

    def fixture_reset(self, root):
        dump(root/'C320/execution.json', {'status': 'completed', 'epochs': 1,
             'selected': str((root/'C320/training/best_raw.pt').resolve()),
             'checkpoint_sha256': self.delivery.sha(root/'C320/training/best_raw.pt'), 'test_used_for_selection': False})

    def test_integrity_union_deduplicates_and_fails_closed(self):
        target = self.root/'source.bin'
        target.write_bytes(b'unchanged')
        expected = hashlib.sha256(b'unchanged').hexdigest()
        sources = [{str(target.resolve()): expected}, {str(target.resolve()): expected}]
        result = self.finalize.check_file_integrity(sources)
        self.assertTrue(result['passed'])
        self.assertEqual(result['unique_files'], 1)
        target.write_bytes(b'changed')
        result = self.finalize.check_file_integrity(sources)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['errors']), 1)
        with self.assertRaises(ValueError):
            self.finalize.check_file_integrity([sources[0], {str(target.resolve()): '0'*64}])

    def test_metrics_missing_class_and_confusion_are_honest(self):
        result = self.finalize.metrics_from_predictions(['a','b','absent'], [0,0,1], [1,1,1])
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['total'], 3)
        self.assertIsNone(result['per_class']['absent']['accuracy'])
        self.assertEqual(result['confusion_matrix'], [[0,2,0],[0,1,0],[0,0,0]])
        self.assertEqual(result['top_confusions'][0], {'true':'a','predicted':'b','count':2})
        self.assertEqual(result['worst3'][0]['label'], 'a')

    def test_archive_verifies_each_sha_and_crc_and_excludes_large_inputs(self):
        folder = self.root/'delivery'
        folder.mkdir()
        (folder/'model.pt').write_bytes(b'model fixture')
        (folder/'readme.txt').write_text('fixture')
        archive = self.root/'delivery.zip'
        manifest = self.delivery.make_archive(folder, archive)
        self.assertTrue(manifest['passed'])
        self.assertEqual(manifest['verified_files'], 2)
        (folder/'data').mkdir()
        (folder/'data/raw.jpg').write_bytes(b'large raw forbidden')
        with self.assertRaises(ValueError):
            self.delivery.make_archive(folder, self.root/'bad.zip')

    def test_predict_refuses_wrong_official_class_order_and_cuda_fallback(self):
        with self.assertRaises(ValueError):
            self.predict.validate_payload({'class_names': list(reversed(self.predict.CLASS_NAMES)),
                'image_size':224, 'preprocessing':{'resize_mode':'letterbox'}})
        class FakeTorch:
            class cuda:
                @staticmethod
                def is_available(): return False
        with self.assertRaises(RuntimeError):
            self.predict.require_device('cuda:0', FakeTorch)

    def test_delivery_requires_saved_fixed_evaluation_before_creating_output(self):
        target = self.root/'delivery'
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.delivery.build(self.root, target)
        self.assertFalse(target.exists())

    def checkpoint_fixture(self):
        import torch
        from PIL import Image
        torch.set_num_threads(2)
        model_builder,_ = self.predict.source_modules()
        model = model_builder('mobilenet_v3_small',num_classes=14,pretrained=False)
        with torch.no_grad():
            model.classifier[-1].weight.zero_()
            model.classifier[-1].bias.zero_()
            model.classifier[-1].bias[3]=2.
        payload={'class_names':self.predict.CLASS_NAMES,'image_size':224,
                 'preprocessing':{'resize_mode':'letterbox'},'model_name':'mobilenet_v3_small',
                 'model_state':model.state_dict(),'variant':'raw'}
        checkpoint=self.root/'synthetic.pt'
        torch.save(payload,checkpoint)
        image=self.root/'synthetic.png'
        Image.new('RGB',(80,40),(80,100,120)).save(image)
        return checkpoint,image,payload

    def test_cpu_predictor_loads_strict_real_14_class_fixture(self):
        checkpoint,image,payload=self.checkpoint_fixture()
        result=self.predict.predict_image(checkpoint,image,device='cpu')
        self.assertEqual(result['label'],'玉米')
        self.assertEqual(result['image_size'],224)
        for flags in ([str(image)],['--image',str(image)]):
            result=subprocess.run([sys.executable,str(Path(__file__).parent/'predict.py'),
                *flags,'--checkpoint',str(checkpoint),'--device','cpu'],
                capture_output=True,text=True,encoding='utf-8',check=True,
                env=dict(os.environ,PYTHONIOENCODING='utf-8'))
            self.assertEqual(json.loads(result.stdout)['label'],'玉米')
        import torch
        del payload['model_state']['classifier.3.bias']
        torch.save(payload,checkpoint)
        with self.assertRaises(RuntimeError):
            self.predict.load_predictor(checkpoint,device='cpu')

    def test_full_synthetic_delivery_is_independent_and_fingerprints_match(self):
        # 真实14类模型结构+临时单图，完全不调用最终测试评分入口。
        checkpoint,image,payload=self.checkpoint_fixture()
        root=self.fixture()
        source=Path(__file__).parent
        copy=lambda a,b:(b.parent.mkdir(parents=True,exist_ok=True),shutil.copy2(a,b))
        copy(source/'predict.py',root/'predict.py')
        for path in (source/'source_snapshot').rglob('*.py'):
            copy(path,root/'source_snapshot'/path.relative_to(source/'source_snapshot'))
        chosen,arms=self.finalize.choose_from_completed_arms(root)
        for arm,item in arms.items():
            import torch
            torch.save(dict(payload,image_size=224 if arm=='A224' else 320,epoch=item['epoch'],
                       selection_key=item['key'],selection_stats=item['stats'],validation_loss=item['loss']),item['path'])
            item['checkpoint_sha256']=self.delivery.sha(item['path'])
            item['completed_epochs']=1
            for name in ['config.json','options.json','selection.json','execution.json','best_candidates.json',
                         'validation_history.json','sample_counts.json','bn_audit.json','training/history.json',
                         'training.json','raw_training.json','validation.json','test.json','weights.json',
                         'preflight.json','data_audit.json']:
                dump(root/arm/name,{})
            for name in ['train_validation.csv','test_only.csv']:
                (root/arm/name).write_text('fixture')
        chosen=arms['B320']
        dump(root/'final_integrity_check.json',{'passed':True})
        selection={'chosen':chosen,'arms':arms,'integrity_check_sha256':self.delivery.sha(root/'final_integrity_check.json'),
          'code_and_configuration_sha256':{},'training_read_audit':{'C320':{'foreground_usage':[{'count':1}]*59}}}
        dump(root/'final_selection.json',selection)
        metrics=self.finalize.metrics_from_predictions(self.predict.CLASS_NAMES,[3],[3])
        evaluation={'validation':metrics,'test':metrics,'variant':'raw','image_size':320,
          'test_inference_passes':1,'test_used_for_selection':False,
          'final_selection_sha256':self.delivery.sha(root/'final_selection.json'),
          'checkpoint_sha256':chosen['checkpoint_sha256']}
        dump(root/'final_evaluation.json',evaluation)
        dump(root/'cpu_gpu_diagnostic.json',{'top1_equal':True})
        for name in ['local_inference_benchmark.json','previous_version_comparison.json','foreground_preview_manifest.json']:
            dump(root/name,{})
        dump(root/'test_predictions.json',[])
        dump(root/'validation_predictions.json',[{'predicted':'玉米'}])
        dump(root/'B320/validation.json',[{'source':str(image),'actual_input':str(image)}])
        dump(root/'C320/training.json',[{'kind':'foreground_rotation','family_id':str(n)} for n in range(59)])
        output=root/'delivery'
        result=self.delivery.build(root,output)
        self.assertTrue(Path(result['archive']).is_file())
        self.assertTrue(json.loads((output/'CPU独立加载验证.json').read_text(encoding='utf-8'))['passed'])
        hashes=json.loads((output/'文件指纹.json').read_text(encoding='utf-8'))
        for name,digest in hashes.items():
            self.assertEqual(self.delivery.sha(output/name),digest)
        self.assertFalse(any('__pycache__' in name for name in hashes))

    def test_arm_payload_contract_rejects_wrong_size_variant_epoch_and_preprocessing(self):
        root=self.fixture()
        selected,arms=self.finalize.choose_from_completed_arms(root)
        import torch
        payload=torch.load(selected['path'],map_location='cpu',weights_only=False)
        self.assertTrue(hasattr(self.predict,'validate_arm_payload'),'缺少分支权重元数据核对')
        self.predict.validate_arm_payload(payload,'B320',selected)
        for change in ({'image_size':224},{'epoch':2},{'variant':'ema'},
                       {'class_names':list(reversed(self.predict.CLASS_NAMES))},
                       {'preprocessing':{'resize_mode':'center_crop'}}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.predict.validate_arm_payload(dict(payload,**change),'B320',selected)
        torch.save(dict(payload,image_size=224),selected['path'])
        for name in ['selection.json','execution.json']:
            data=json.loads((root/'B320'/name).read_text())
            data['checkpoint_sha256']=self.delivery.sha(selected['path'])
            dump(root/'B320'/name,data)
        with self.assertRaises(ValueError):
            self.finalize.choose_from_completed_arms(root)

    def test_readme_explains_inference_only_dependencies_and_historical_training_paths(self):
        root=self.fixture();chosen,arms=self.finalize.choose_from_completed_arms(root)
        stats=self.finalize.metrics_from_predictions(self.predict.CLASS_NAMES,[3],[3])
        text=self.delivery.readme_text({'image_size':320,'variant':'raw','validation':stats,'test':stats},
                                      {'chosen':chosen,'arms':arms})
        for phrase in ['仅推理','非完整自包含训练复现','source_snapshot','绝对',
                       'masked_samples_20261123_135614','background_fill.py',
                       'segmentation_20261122_203914/dependencies','SciPy','OpenCV',
                       'foreground_reviewed.json','root_visual_review_complete=true']:
            with self.subTest(phrase=phrase):self.assertIn(phrase,text)

    def sequence_fixture(self):
        stats=self.finalize.metrics_from_predictions(self.predict.CLASS_NAMES,[3],[3]);stats['loss']=.5
        chosen={'arm':'B320','path':'unused.pt','checkpoint_sha256':'1'*64,'epoch':1,
                'variant':'raw','key':[1.,1.,1.,-.5],'stats':stats,'loss':.5}
        selection={'chosen':chosen,'arms':{'B320':chosen}}
        dump(self.root/'final_selection.json',selection)
        predictions=[{'source':'synthetic.png','predicted':'玉米'}]
        diagnostic={'top1_equal':True,'cpu_batch_size':1,'gpu_batch_size':1,
                    'gpu_single_vs_batch_top1_equal':True}
        return selection,stats,predictions,diagnostic

    def test_pretest_validation_or_diagnostic_failure_does_not_create_marker(self):
        self.assertTrue(hasattr(self.finalize,'score_test_once'),'缺少预检与唯一测试提交边界')
        selection,stats,predictions,diagnostic=self.sequence_fixture()
        for metrics,diag in [(dict(stats,loss=.8),diagnostic),
                             (dict(stats,correct=0),diagnostic),
                             (stats,dict(diagnostic,top1_equal=False))]:
            with self.subTest(metrics=metrics,diagnostic=diag),self.assertRaises(RuntimeError):
                self.finalize.score_test_once(self.root,selection,device='synthetic',
                    val_metrics=metrics,val_predictions=predictions,diagnostic=diag,benchmark={},
                    test_evaluator=lambda:(_ for _ in ()).throw(AssertionError('不得评分测试')))
            self.assertFalse((self.root/'test_inference_started.json').exists())

    def test_postprocess_failure_preserves_official_results_and_retry_never_rescores(self):
        self.assertTrue(hasattr(self.finalize,'score_test_once'),'缺少唯一测试保存边界')
        self.assertTrue(hasattr(self.finalize,'postprocess_saved_results'),'缺少仅后处理重试路径')
        selection,stats,predictions,diagnostic=self.sequence_fixture()
        calls=[]
        def evaluate():
            calls.append('test')
            return stats,predictions
        with self.assertRaises(FileNotFoundError):
            self.finalize.score_test_once(self.root,selection,device='synthetic',
                val_metrics=stats,val_predictions=predictions,diagnostic=diagnostic,benchmark={},
                test_evaluator=evaluate)
        self.assertEqual(calls,['test'])
        self.assertTrue((self.root/'final_evaluation.json').is_file())
        self.assertTrue((self.root/'test_predictions.json').is_file())
        saved=(self.root/'test_predictions.json').read_bytes()
        previous=self.root.parent/'asparagus_half_20261126_172248/final_evaluation.json'
        # 两个目录都位于当前测试的独占临时根中，由TemporaryDirectory统一清理。
        dump(previous,{'synthetic_previous':True})
        self.finalize.postprocess_saved_results(self.root)
        self.finalize.postprocess_saved_results(self.root)
        process=subprocess.run([sys.executable,str(Path(__file__).parent/'finalize_models.py'),
                  '--run',str(self.root),'--postprocess'],capture_output=True,text=True,encoding='utf-8',
                  env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
        self.assertEqual(json.loads(process.stdout)['status'],'completed')
        with self.assertRaises(RuntimeError):
            self.finalize.choose_from_completed_arms(self.root)
        with self.assertRaises(RuntimeError):
            self.finalize.score_test_once(self.root,selection,device='synthetic',
                val_metrics=stats,val_predictions=predictions,diagnostic=diagnostic,benchmark={},test_evaluator=evaluate)
        self.assertEqual(calls,['test'])
        self.assertEqual((self.root/'test_predictions.json').read_bytes(),saved)

    def test_delivery_rechecks_arm_metadata_even_when_changed_sha_matches(self):
        import torch
        root=self.fixture();chosen,arms=self.finalize.choose_from_completed_arms(root)
        path=Path(chosen['path'])
        payload=torch.load(path,map_location='cpu',weights_only=False)
        torch.save(dict(payload,image_size=224),path)
        chosen['checkpoint_sha256']=self.delivery.sha(path)
        with self.assertRaises(ValueError):
            self.delivery.verify_arm_checkpoint(path,'B320',chosen)


if __name__ == '__main__':
    unittest.main()

"""训练完成后新进程检查、评测两套协议和打包，不依据测试改选轮次。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN/'source_snapshot'))
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest, validate_manifest
from oven_food.model import build_model, checkpoint_class_names, checkpoint_resize_mode
from joint_support import hierarchical_weights


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(path, records, tag, names):
    cp = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint_class_names(cp, observed_labels=[r.food_label for r in records]) == names
    assert checkpoint_resize_mode(cp) == 'letterbox'
    model = build_model(cp['model_name'], num_classes=14, pretrained=False)
    model.load_state_dict(cp['model_state'], strict=True)
    model.to('cuda:0').eval()
    dataset = FoodImageDataset(records, data_root=ROOT, class_names=names, augmentation_profile='baseline', image_size=224, resize_mode='letterbox')
    matrix = np.zeros((14,14), dtype=np.int64)
    predictions = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=32, num_workers=0):
            logits = model(images.to('cuda:0'))
            assert torch.isfinite(logits).all()
            loss_sum += float(torch.nn.functional.cross_entropy(logits, labels.to('cuda:0'), reduction='sum'))
            probabilities = logits.softmax(1).cpu()
            for target, predicted, p in zip(labels.tolist(), probabilities.argmax(1).tolist(), probabilities):
                record = records[len(predictions)]
                matrix[target,predicted] += 1
                predictions.append(dict(image=record.image_path, expected=names[target], predicted=names[predicted], probability=float(p[predicted])))
    assert len(predictions) == len(records)
    with (RUN/(tag+'_predictions.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0])); w.writeheader(); w.writerows(predictions)
    per = {n: dict(correct=int(matrix[i,i]), total=int(matrix[i].sum())) for i,n in enumerate(names)}
    lowest = sorted([n for n in names if per[n]['total']], key=lambda n:(per[n]['correct']/per[n]['total'], names.index(n)))[:3]
    worst = [dict(label=n, **per[n], misclassified_as={names[j]:int(v) for j,v in enumerate(matrix[names.index(n)]) if v and names[j]!=n}) for n in lowest]
    return dict(correct=int(matrix.trace()), total=int(matrix.sum()), loss=loss_sum/len(records), per_class=per, confusion_matrix=matrix.tolist(), worst_three=worst)


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    audit = read(RUN/'data_audit.json'); names = audit['class_names']
    execution = read(RUN/'execution.json'); assert execution['status']=='completed' and execution['epochs']==20
    records = read_manifest(RUN/'manifest.csv'); validate_manifest(records, data_root=ROOT, check_files=True)
    train = [r for r in records if r.split=='train']; assert len(train)==5734 and all(r.split!='test' for r in records)
    pool = read(RUN/'training_inventory.json'); weights = read(RUN/'sampling_weights.json')
    assert len(pool)==5734 and {r['actual_input'] for r in pool}=={r.image_path for r in train}
    assert np.allclose(weights, hierarchical_weights(pool,target_share=.75), rtol=0,atol=1e-15)
    beef = [r for r in pool if r['label']=='牛排']
    assert len(beef)==205 and all(r['environment']=='target' and r['era']=='second_real' for r in beef)
    forbidden = {r['raw_sha256'] for r in read(RUN/'excluded_first_beef.json')}
    assert not forbidden & {r['raw_sha256'] for r in pool}
    assert not {r['source'].casefold().replace('\\','/') for r in pool} & {r['source'].casefold().replace('\\','/') for r in read(RUN/'quality_exclusions.json')}
    counts = read(RUN/'sample_counts.json')
    assert set(counts)=={r.image_path for r in train} and min(counts.values())>=1 and sum(counts.values())==5734*20
    pixel_audit=read(RUN/'pixel_overlap_audit.json')
    assert pixel_audit['train_second_pixel_overlap']==pixel_audit['train_first_validation_pixel_overlap']==0
    init_audit = read(RUN/'initialization_audit.json')
    assert not init_audit['initialized_from_previous_model']
    assert sha(RUN/'official_initialization.pt')==init_audit['initial_sha256']==audit['initial_sha256']
    initial = torch.load(RUN/'official_initialization.pt',map_location='cpu',weights_only=True)
    assert initial['epoch']==0 and initial['class_names']==names
    official=torch.load(Path(torch.hub.get_dir())/'checkpoints/mobilenet_v3_small-047dcff4.pth',map_location='cpu',weights_only=True)
    for k,v in official.items():
        if k not in {'classifier.3.weight','classifier.3.bias'}: assert torch.equal(v,initial['model_state'][k])

    def check_image(r):
        assert sha(r['actual_input'])==r['input_sha256']
        if r['raw_sha256']: assert sha(r['source'])==r['raw_sha256']
        if r.get('metadata'): assert sha(r['metadata'])==r['metadata_sha256']
    second=read(RUN/'second_eval_inventory.json'); first=read(RUN/'first_validation_inventory.json')
    assert len(second)==167 and len(first)==193 and all(r['label']!='牛排' for r in first)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(check_image,pool+second+first))
    assert not {r['raw_sha256'] for r in pool if r['raw_sha256']} & {r['raw_sha256'] for r in second+first}
    val_A=read_manifest(RUN/'validation_A.csv'); test_A=read_manifest(RUN/'test_A.csv')
    val_B=read_manifest(RUN/'validation_B.csv'); test_B=read_manifest(RUN/'test_B.csv')
    assert tuple(map(len,[val_A,test_A,val_B,test_B]))==(83,84,193,167)
    assert {r.image_path for r in val_A}.isdisjoint(r.image_path for r in test_A)
    assert {r.image_path for r in test_B}=={r.image_path for r in val_A+test_A}
    assert {r.session_id for r in train}&{r.session_id for r in val_A}=={'real|南瓜|session-20261117-2fc02ecc9c09'}
    histories={p:read(RUN/f'selection_{p}/history.json') for p in ['A','B']}
    cps={p:torch.load(RUN/f'selection_{p}/best.pt',map_location='cpu',weights_only=True) for p in ['A','B']}
    hashes={p:sha(RUN/f'selection_{p}/best.pt') for p in ['A','B']}
    for p,cp in cps.items():
        h=histories[p]; assert len(h)==20 and [x['epoch'] for x in h]==list(range(1,21))
        assert all(np.isfinite(x[k]) for x in h for k in ['train_loss','validation_loss','validation_accuracy'])
        best=sorted(h,key=lambda x:(-x['validation_accuracy'],x['validation_loss']))[0]
        assert cp['epoch']==best['epoch'] and cp['validation_accuracy']==best['validation_accuracy']
        assert cp['class_names']==names and cp['class_loss_weights'] is None and not cp['incremental'] and not cp['distillation']
        assert all(torch.isfinite(v).all() for v in cp['model_state'].values())
        assert cp['config']['checkpoint']==str(RUN/'official_initialization.pt')
    assert all(a['train_loss']==b['train_loss'] for a,b in zip(histories['A'],histories['B']))
    print('来源、牛排排除、官方初始化、20轮与A/B各自选模核验通过；现在固定权重评分。',flush=True)
    result=dict(class_names=names, execution=execution, training_images=5734, real_training_images=3990, web_training_images=1744,
                beef_only_second_real=205, checkpoint_sha256=hashes, best_epochs={p:cp['epoch'] for p,cp in cps.items()},
                independent_test=False, shared_training_trajectory=True, sampling=read(RUN/'sampling_execution.json'),
                same_model_parameters=all(torch.equal(v,cps['B']['model_state'][k]) for k,v in cps['A']['model_state'].items()),
                gpu=dict(name=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),cuda_runtime=torch.version.cuda))
    result['A']=dict(validation=evaluate(RUN/'selection_A/best.pt',val_A,'A_validation83',names),
                     test=evaluate(RUN/'selection_A/best.pt',test_A,'A_test84',names))
    result['B']=dict(validation=evaluate(RUN/'selection_B/best.pt',val_B,'B_validation193',names),
                     test=evaluate(RUN/'selection_B/best.pt',test_B,'B_test167',names),
                     common84=evaluate(RUN/'selection_B/best.pt',test_A,'B_common84',names))
    for p,size in [('A',83),('B',193)]:
        assert result[p]['validation']['correct']/size==cps[p]['validation_accuracy']
        assert abs(result[p]['validation']['loss']-cps[p]['validation_loss'])<1e-5
    baseline=ROOT/'runs/C_pumpkin_croissant_20261118_160802/training/best.pt'
    baseline_sha=sha(baseline)
    assert baseline_sha=='9db30ce0eedee1fe8d53843eaff34d3b49e435c3f2222417ce79db21584c674a'
    result['baseline']=dict(checkpoint_sha256=baseline_sha,
                            common84=evaluate(baseline,test_A,'baseline_common84',names),
                            full167=evaluate(baseline,test_B,'baseline_full167',names))
    assert sha(baseline)==baseline_sha
    for p in ['A','B']: assert sha(RUN/f'selection_{p}/best.pt')==hashes[p]
    dump(RUN/'final_report.json',result)
    from build_report import create_report_and_packages
    delivered=create_report_and_packages(RUN,ROOT,result)
    dump(RUN/'final_checks.json',dict(status='passed',train_images=5734,epochs=20,first_beef_excluded=True,official_initialization=True,
                                     A_validation83_test84=True,B_validation193_test167=True,original_images_unchanged=True,
                                     checkpoints_reloaded_in_new_process=True,checkpoint_sha256=hashes,deliveries=delivered))
    (ROOT/'runs/latest_joint_AB_completed.txt').write_text(str(RUN),encoding='utf-8')
    print(json.dumps(dict(best_epochs=result['best_epochs'],A_test=result['A']['test']['correct'],B_test=result['B']['test']['correct'],
                          B_common=result['B']['common84']['correct'],baseline84=result['baseline']['common84']['correct'],delivery=delivered),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

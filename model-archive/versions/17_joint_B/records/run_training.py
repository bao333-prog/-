"""一次联合训练，按两套验证规则分别选模；不依据测试成绩改变训练。

B验证用于原训练入口的标准日志和保存；A验证在每轮末单独计算并保存。
使用A时保留同源南瓜，明确为开发对照；不更改项目的分组校验器。
"""
from pathlib import Path
from collections import Counter
from dataclasses import asdict
import json
import sys
import time

import torch
from torch.utils.data import DataLoader

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(RUN / 'source_snapshot'))
from oven_food.config import load_experiment_config
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest
import oven_food.train as training_module
from joint_support import CoverageThenWeightedSampler


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert torch.cuda.is_available(), 'GPU不可用时停止，不自动改用CPU'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_experiment_config(RUN / 'train_config.json')
    audit = read(RUN / 'data_audit.json')
    names = audit['class_names']
    rows = read(RUN / 'training_inventory.json')
    weights = read(RUN / 'sampling_weights.json')
    assert len(rows) == len(weights) == audit['training_images']
    val_A = read_manifest(RUN / 'validation_A.csv')
    dataset_A = FoodImageDataset(val_A, data_root=ROOT, class_names=names, augmentation_profile='baseline',
                                image_size=224, resize_mode='letterbox')
    loader_A = DataLoader(dataset_A, batch_size=32, num_workers=0, shuffle=False,
                          generator=torch.Generator().manual_seed(1042))
    (RUN / 'selection_A').mkdir(exist_ok=False)
    history_A = []
    best_A_accuracy, best_A_loss = -1.0, float('inf')
    original_validate = training_module._validate_epoch
    original_getitem = FoodImageDataset.__getitem__
    counts = Counter()

    def tracked_getitem(dataset, index):
        sample = original_getitem(dataset, index)
        if dataset.augmentation_profile == 'oven_geometry':
            counts[dataset.records[index].image_path] += 1
        return sample

    def scoped_sampler(records, target_fraction, torch_module, sampler_type):
        assert target_fraction == .75
        assert [r.image_path for r in records] == [r['actual_input'] for r in rows]
        return CoverageThenWeightedSampler(weights, seed=config.seed)

    def dual_validate(model, loader_B, criterion, device, torch_module):
        nonlocal best_A_accuracy, best_A_loss
        B_loss, B_accuracy = original_validate(model, loader_B, criterion, device, torch_module)
        # A使用独立DataLoader随机源；验证不消耗训练Dropout/采样的随机数。
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(device).clone()
        A_loss, A_accuracy = original_validate(model, loader_A, criterion, device, torch_module)
        assert torch.equal(rng_cpu, torch.get_rng_state())
        assert torch.equal(rng_cuda, torch.cuda.get_rng_state(device))
        epoch = len(history_A) + 1
        history_A.append(dict(epoch=epoch, validation_loss=A_loss, validation_accuracy=A_accuracy))
        dump(RUN / 'selection_A/validation_history_progress.json', history_A)
        if training_module._is_better_checkpoint(A_accuracy, A_loss, best_A_accuracy, best_A_loss):
            best_A_accuracy, best_A_loss = A_accuracy, A_loss
            torch.save(dict(model_state=model.state_dict(), class_names=names, model_name='mobilenet_v3_small', image_size=224,
                            epoch=epoch, validation_accuracy=A_accuracy, validation_loss=A_loss, class_loss_weights=None,
                            incremental=None, distillation=None, preprocessing=dict(resize_mode='letterbox', mean=[.485,.456,.406], std=[.229,.224,.225]),
                            config=asdict(config), selection=dict(protocol='A_second_batch_half', validation_manifest=str(RUN/'validation_A.csv'),
                                                                test_manifest=str(RUN/'test_A.csv'), shared_training=True)),
                       RUN / 'selection_A/best.pt')
        print(json.dumps(dict(protocol='A', epoch=epoch, validation_accuracy=A_accuracy, validation_loss=A_loss)), flush=True)
        return B_loss, B_accuracy

    FoodImageDataset.__getitem__ = tracked_getitem
    training_module._weighted_sampler = scoped_sampler
    training_module._validate_epoch = dual_validate
    started = time.perf_counter()
    try:
        checkpoint_B = training_module.train_experiment(config)
        assert len(history_A) == config.epochs == 20
        history_B = read(RUN / 'selection_B/history.json')
        for a, b in zip(history_A, history_B):
            assert a['epoch'] == b['epoch']
            a['train_loss'] = b['train_loss']
        dump(RUN / 'selection_A/history.json', history_A)
        dump(RUN / 'selection_A/summary.json', dict(protocol='A', checkpoint=str(RUN/'selection_A/best.pt'),
                                                  best_validation_accuracy=best_A_accuracy, best_validation_loss=best_A_loss))
        dump(RUN / 'execution.json', dict(status='completed', epochs=20, seconds=time.perf_counter()-started,
                                          checkpoint_A=str(RUN/'selection_A/best.pt'), checkpoint_B=str(checkpoint_B),
                                          shared_training_trajectory=True))
        dump(RUN / 'sample_counts.json', dict(counts))
        assert set(counts) == {r['actual_input'] for r in rows}
        assert min(counts.values()) >= 1 and sum(counts.values()) == len(rows)*20
        dump(RUN / 'sampling_execution.json', dict(total_reads=sum(counts.values()), unique_images=len(counts), minimum_reads=min(counts.values()),
                                                  maximum_reads=max(counts.values()), first_epoch_full_coverage=True, replacement_from_epoch2=True))
    except Exception as error:
        dump(RUN / 'execution.json', dict(status='failed', error=str(error), seconds=time.perf_counter()-started))
        dump(RUN / 'partial_sample_counts.json', dict(counts))
        raise


if __name__ == '__main__':
    main()

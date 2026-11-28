"""复用原训练入口，测试集不进入本进程；只用验证集选模。"""
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from prepare_data import RUN, ROOT, load, dump, sha

sys.path.insert(0, str(RUN / 'source_snapshot'))
from oven_food.config import load_experiment_config
import oven_food.train as training
from family_sampler import FamilyCoverageSampler
from validation_monitor import validate_with_class_metrics


def main():
    assert torch.cuda.is_available(), 'GPU不可用，停止而非偷偷改用CPU'
    independent = load(RUN / 'source_isolation_independent.json')
    assert not independent['errors'] and not any(independent['mismatches'].values())
    assert Path(training.__file__).resolve().is_relative_to(RUN/'source_snapshot')
    # 独立来源审计已完成；具体字段另由启动前检查确认，不读取测试预测。
    assert (RUN / 'prepared_data_audit.json').is_file()
    assert sha(RUN / 'official_initialization.pt') == '5562ddb1fe6d07123b9d0a57fd49c150216f3fd66fa2dffecf0e51487797ed9d'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_experiment_config(RUN / 'train_config.json')
    rows = load(RUN / 'training_integrated.json')
    names = load(RUN / 'official_class_names.json')
    weights = load(RUN / 'sampling_weights.json')
    counts = Counter()
    history = []
    original = training.FoodImageDataset.__getitem__

    def tracked(dataset, index):
        result = original(dataset, index)
        if dataset.augmentation_profile == 'oven_geometry':
            counts[dataset.records[index].image_path] += 1
        return result

    def sampler(records, target_fraction, torch_module, sampler_type):
        assert [r.image_path for r in records] == [r['actual_input'] for r in rows]
        return FamilyCoverageSampler(rows, weights, seed=config.seed, coverage_interval=4)

    def validate(model, loader, criterion, device, torch_module):
        loss, accuracy, stats = validate_with_class_metrics(model, loader, criterion, device, names)
        epoch = len(history)+1
        history.append({'epoch': epoch, 'seconds': time.perf_counter()-started, **stats})
        dump(RUN / 'validation_class_history.json', history)
        dump(RUN / 'training_progress.json', {'epoch': epoch, 'validation_loss': loss,
              'validation_accuracy': accuracy, 'elapsed_seconds': time.perf_counter()-started})
        print('VALIDATION', epoch, stats['correct'], '/', stats['total'], flush=True)
        return loss, accuracy

    training.FoodImageDataset.__getitem__ = tracked
    training._weighted_sampler = sampler
    training._validate_epoch = validate
    started = time.perf_counter()
    dump(RUN / 'execution.json', {'status': 'running', 'epochs_requested': 20, 'training_started': True})
    try:
        checkpoint = training.train_experiment(config)
        assert len(history) == config.epochs == 20
        dump(RUN / 'sample_counts.json', dict(counts))
        assert set(counts) == {r['actual_input'] for r in rows}
        assert min(counts.values()) >= 1
        assert dict(counts) == load(RUN/'expected_sampling_counts.json')
        dump(RUN / 'execution.json', {'status': 'completed', 'epochs': 20,
             'seconds': time.perf_counter()-started, 'checkpoint': str(checkpoint),
             'checkpoint_sha256': sha(checkpoint), 'training_started': True,
             'test_used_for_selection': False, 'initialization_sha256': sha(RUN/'official_initialization.pt')})
    except Exception as exc:
        dump(RUN / 'execution.json', {'status': 'failed', 'error': repr(exc), 'seconds': time.perf_counter()-started})
        dump(RUN / 'partial_sample_counts.json', dict(counts))
        raise


if __name__ == '__main__':
    main()

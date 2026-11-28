"""在新进程严格重载已固定权重，只在训练完成后评测完整原图。"""
import sys
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from prepare_data import RUN, ROOT, load, dump, sha

sys.path.insert(0, str(RUN / 'source_snapshot'))
from oven_food.dataset import FoodImageDataset
from oven_food.manifest import read_manifest
from oven_food.model import build_model
from oven_food.metrics import classification_metrics


def main():
    execution = load(RUN / 'execution.json')
    assert execution['status'] == 'completed'
    assert not (RUN / 'final_evaluation.json').exists(), '禁止根据测试表现重复调参再评分'
    torch.set_num_threads(4)
    checkpoint_path = RUN / 'training/best.pt'
    assert sha(checkpoint_path) == execution['checkpoint_sha256']
    for filename, digest in load(RUN/'integration_preflight_review.json')['file_sha256'].items():
        assert sha(RUN/filename) == digest, '预审冻结文件发生变化：' + filename
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    names = load(RUN / 'official_class_names.json')
    assert checkpoint['class_names'] == names
    assert checkpoint['config']['checkpoint'] == str(RUN / 'official_initialization.pt')
    model = build_model('mobilenet_v3_small', num_classes=len(names), pretrained=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    device = torch.device('cuda:0')
    model.to(device).eval()
    outputs = {}
    for split, manifest_file, inventory_file in [
        ('validation', 'train_validation.csv', 'validation_base.json'),
        ('test', 'test_only.csv', 'test_base.json')]:
        inventory = load(RUN / inventory_file)
        for row in inventory:
            assert sha(row['actual_input']) == row['prepared_sha256']
        records = [r for r in read_manifest(RUN / manifest_file) if r.split == ('val' if split == 'validation' else 'test')]
        assert [r.image_path for r in records] == [r['actual_input'] for r in inventory]
        dataset = FoodImageDataset(records, data_root=ROOT, class_names=names,
                augmentation_profile='baseline', image_size=224, resize_mode='letterbox')
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        truth, predicted, confidences = [], [], []
        with torch.inference_mode():
            for images, labels in loader:
                probabilities = model(images.to(device)).softmax(1)
                assert torch.isfinite(probabilities).all(), '非有限测试输出，停止'
                scores, indices = probabilities.max(1)
                truth.extend(labels.tolist())
                predicted.extend(indices.cpu().tolist())
                confidences.extend(scores.cpu().tolist())
        metrics = classification_metrics(truth, predicted, class_names=names)
        matrix = metrics['confusion_matrix']
        per_class = {}
        for i, name in enumerate(names):
            total = sum(matrix[i])
            per_class[name] = {'correct': matrix[i][i], 'total': total,
                    'accuracy': matrix[i][i]/total if total else None,
                    'misclassified_as': {names[j]: n for j, n in enumerate(matrix[i]) if j != i and n}}
        worst = sorted([n for n in names if per_class[n]['total']],
                       key=lambda n: (per_class[n]['accuracy'], names.index(n)))[:3]
        predictions = [{'source': row['source'], 'source_sha256': row['family_id'],
                'true_label': names[y], 'predicted_label': names[p], 'confidence': score,
                'correct': y == p} for row, y, p, score in zip(inventory, truth, predicted, confidences)]
        outputs[split] = {**metrics, 'per_class': per_class, 'worst_three': worst,
                          'full_original_images_only': True}
        dump(RUN / (split + '_predictions.json'), predictions)
    # 新进程内再用一张未裁整图确认CPU/GPU预测一致；不再次计算测试排名。
    example = load(RUN / 'validation_base.json')[0]
    records = [r for r in read_manifest(RUN / 'train_validation.csv') if r.split == 'val']
    item = FoodImageDataset(records[:1], data_root=ROOT, class_names=names,
            augmentation_profile='baseline', image_size=224, resize_mode='letterbox')[0][0][None]
    with torch.inference_mode():
        gpu = model(item.to(device)).cpu()
        model.cpu()
        cpu = model(item)
    assert int(gpu.argmax(1)) == int(cpu.argmax(1))
    dump(RUN / 'final_evaluation.json', {'checkpoint_sha256': sha(checkpoint_path),
         'best_epoch': checkpoint['epoch'], 'class_names': names, **outputs,
         'reload_strict': True, 'cpu_gpu_example_prediction_agrees': True,
         'cpu_gpu_max_logit_difference': float((cpu-gpu).abs().max()),
         'test_evaluated_after_checkpoint_fixed': True,
         'test_is_new_blind_set': False, 'physical_individual_independence_proven': False})
    print('已固定模型评测完成', {k: v['overall_accuracy'] for k,v in outputs.items()}, flush=True)


if __name__ == '__main__':
    main()

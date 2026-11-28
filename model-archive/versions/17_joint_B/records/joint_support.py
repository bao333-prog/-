"""本轮局部采样工具，不改变项目的公共训练接口。"""
from collections import Counter, defaultdict
import math
import torch


def hierarchical_weights(rows, *, target_share=0.75):
    """类别→域→新旧实拍→采集组→图片，各层分别分配概率质量。

    域同时存在时分配75%实拍、25%网图；缺某域的类别全部给现有域。
    同一类实拍若有新、旧两个批次，各得一半，避免只复习最近一次摆盘。
    """
    if not rows or not 0 < target_share < 1 or any(r['split'] != 'train' for r in rows):
        raise ValueError('weights require nonempty training records and target_share in (0,1)')
    envs = defaultdict(set)
    eras = defaultdict(set)
    groups = defaultdict(set)
    sizes = Counter()
    for r in rows:
        label, env, era, group = r['label'], r['environment'], r['era'], r['sampling_group']
        if env not in {'target', 'proxy'}:
            raise ValueError('unknown environment')
        envs[label].add(env)
        eras[label, env].add(era)
        groups[label, env, era].add(group)
        sizes[label, env, era, group] += 1
    weights = []
    for r in rows:
        label, env, era, group = r['label'], r['environment'], r['era'], r['sampling_group']
        domain = (target_share if env == 'target' else 1-target_share) if len(envs[label]) == 2 else 1.0
        weights.append(domain / len(envs) / len(eras[label,env]) / len(groups[label,env,era]) / sizes[label,env,era,group])
    assert all(math.isfinite(w) and w > 0 for w in weights)
    assert math.isclose(sum(weights), 1.0, abs_tol=1e-10)
    return weights


class CoverageThenWeightedSampler:
    """首轮遍历全部训练图；以后按分层权重有放回采样，每轮仍N次。

    首轮保证每张合格图实际参与训练。后续各图次数可不同，记录真实抽样数。
    独立随机数生成器使额外验证不影响采样顺序。
    """
    def __init__(self, weights, *, seed):
        self.weights = torch.tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or not len(self.weights) or not torch.isfinite(self.weights).all() or not (self.weights > 0).all():
            raise ValueError('sampling weights must be finite positive 1-D values')
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        self.epoch += 1
        if self.epoch == 1:
            indices = torch.randperm(len(self), generator=self.generator)
        else:
            indices = torch.multinomial(self.weights, len(self), replacement=True, generator=self.generator)
        return iter(indices.tolist())

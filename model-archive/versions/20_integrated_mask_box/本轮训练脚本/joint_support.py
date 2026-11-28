"""本轮局部采样工具，不改变项目的公共训练接口。"""
from collections import Counter, defaultdict
import math
import torch


def hierarchical_weights(rows, *, target_share=0.75, max_expected_draws=4.0):
    """类别→域→新旧实拍→采集组→图片，各层分别分配概率质量。

    域同时存在时分配75%实拍、25%网图；缺某域的类别全部给现有域。
    同一类实拍若有新、旧两个批次，各得一半。重复图片分摊概率；
    每个类别/域/批次分支内部限幅并重新分配，保持该分支总概率不变。
    上限是随机抽样的期望频率，并非实际读取次数硬上限。
    """
    if not rows or not 0 < target_share < 1 or any(r['split'] != 'train' for r in rows):
        raise ValueError('weights require nonempty training records and target_share in (0,1)')
    if not math.isfinite(max_expected_draws) or max_expected_draws < 1:
        raise ValueError('max_expected_draws must be finite and at least 1')
    envs = defaultdict(set)
    eras = defaultdict(set)
    groups = defaultdict(set)
    fingerprints = defaultdict(Counter)
    branches = defaultdict(list)
    keys = []
    for index, r in enumerate(rows):
        label, env, era, group = r['label'], r['environment'], r['era'], r['sampling_group']
        if env not in {'target', 'proxy'}:
            raise ValueError('unknown environment')
        envs[label].add(env)
        eras[label, env].add(era)
        groups[label, env, era].add(group)
        fingerprint = r.get('pixel_sha256') or r.get('input_sha256') or r.get('raw_sha256') or r.get('actual_input') or f'record:{index}'
        fingerprints[label,env,era,group][fingerprint] += 1
        branches[label,env,era].append(index)
        keys.append(fingerprint)
    weights = []
    for index, r in enumerate(rows):
        label, env, era, group = r['label'], r['environment'], r['era'], r['sampling_group']
        domain = (target_share if env == 'target' else 1-target_share) if len(envs[label]) == 2 else 1.0
        group_fingerprints = fingerprints[label,env,era,group]
        weights.append(domain / len(envs) / len(eras[label,env]) / len(groups[label,env,era])
                       / len(group_fingerprints) / group_fingerprints[keys[index]])
    # 逐分支执行“超限项固定，其余按原比例分配剩余质量”，避免直接截断后
    # 全局归一化破坏类别和新旧批次比例。稀少类别无法满足全局上限时，
    # 仅放宽到该分支的均匀概率；实际池的全部分支都可满足4/N。
    for indices in branches.values():
        mass = sum(weights[i] for i in indices)
        cap = max(max_expected_draws/len(rows), mass/len(indices))
        original = {i: weights[i] for i in indices}
        remaining, active = mass, list(indices)
        while active:
            scale = remaining / sum(original[i] for i in active)
            saturated = [i for i in active if original[i]*scale > cap+1e-15]
            if not saturated:
                for i in active: weights[i] = original[i]*scale
                break
            for i in saturated:
                weights[i] = cap
            remaining -= cap*len(saturated)
            occupied = set(saturated)
            active = [i for i in active if i not in occupied]
    assert all(math.isfinite(w) and w > 0 for w in weights)
    assert math.isclose(sum(weights), 1.0, abs_tol=1e-10)
    return weights


class CoverageThenWeightedSampler:
    """每4轮安排一次全量遍历，其余轮次按分层权重采样，每轮N次。

    20轮中第1/5/9/13/17轮全覆盖，保证每图至少5次。其余次数可不同。
    独立随机数生成器使额外验证不影响采样顺序。
    """
    def __init__(self, weights, *, seed, coverage_interval=4):
        self.weights = torch.tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or not len(self.weights) or not torch.isfinite(self.weights).all() or not (self.weights > 0).all():
            raise ValueError('sampling weights must be finite positive 1-D values')
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        if not isinstance(coverage_interval,int) or isinstance(coverage_interval,bool) or coverage_interval < 1:
            raise ValueError('coverage_interval must be a positive integer')
        self.coverage_interval = coverage_interval
        self.coverage_epochs = []
        self.weighted_epochs = []

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        self.epoch += 1
        if (self.epoch-1) % self.coverage_interval == 0:
            self.coverage_epochs.append(self.epoch)
            indices = torch.randperm(len(self), generator=self.generator)
        else:
            self.weighted_epochs.append(self.epoch)
            indices = torch.multinomial(self.weights, len(self), replacement=True, generator=self.generator)
        return iter(indices.tolist())

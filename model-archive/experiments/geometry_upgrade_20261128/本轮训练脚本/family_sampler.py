"""按原图来源家族抽样：派生数量不增加家族在覆盖轮或加权轮的贡献。"""
import bisect
import math
from collections import defaultdict
import torch


class FamilyCoverageSampler:
    def __init__(self, rows, weights, *, seed, coverage_interval=4):
        if not rows or len(rows) != len(weights) or coverage_interval < 1:
            raise ValueError('invalid rows/weights/interval')
        families = defaultdict(dict)
        labels = defaultdict(set)
        for index, (row, weight) in enumerate(zip(rows, weights)):
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('weights must be finite and positive')
            key, path = row['family_id'], row['actual_input']
            labels[key].add(row['label'])
            if len(labels[key]) != 1:
                raise ValueError('同一来源家族出现不同类别')
            if path not in families[key]:
                families[key][path] = [index, 0.]
            families[key][path][1] += weight
        self.keys = list(families)
        self.indices, self.cdfs, masses = [], [], []
        for views in families.values():
            entries = list(views.values())
            mass = sum(item[1] for item in entries)
            cdf, acc = [], 0.
            for _, weight in entries:
                acc += weight/mass
                cdf.append(acc)
            cdf[-1] = 1.
            self.indices.append([item[0] for item in entries])
            self.cdfs.append(cdf)
            masses.append(mass)
        self.mass = torch.tensor(masses, dtype=torch.double)
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.coverage_interval = coverage_interval

    def __len__(self):
        return len(self.keys)

    def __iter__(self):
        self.epoch += 1
        if (self.epoch-1) % self.coverage_interval == 0:
            # 每个来源仅一次，五次覆盖轮轮流访问该来源的不同视图。
            cycle = (self.epoch-1)//self.coverage_interval
            order = torch.randperm(len(self), generator=self.generator).tolist()
            result = [self.indices[f][cycle % len(self.indices[f])] for f in order]
        else:
            chosen = torch.multinomial(self.mass, len(self), replacement=True,
                                       generator=self.generator).tolist()
            coins = torch.rand(len(self), generator=self.generator).tolist()
            result = [self.indices[f][min(bisect.bisect_left(self.cdfs[f], u), len(self.indices[f])-1)]
                      for f, u in zip(chosen, coins)]
        return iter(result)

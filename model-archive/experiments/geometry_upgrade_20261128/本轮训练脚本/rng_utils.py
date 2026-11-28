"""额外验证仅隔离CPU DataLoader随机状态；eval网络不启用随机层。"""
from contextlib import contextmanager
import torch

@contextmanager
def isolated_validation_rng():
    with torch.random.fork_rng(devices=[]):
        yield

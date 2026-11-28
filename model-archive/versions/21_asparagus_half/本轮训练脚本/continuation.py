"""仅依据验证集决定是否按用户要求延长训练。"""

def should_stop(history, nominal=20, maximum=60, patience=3):
    n=len(history)
    if n < nominal: return False
    if n >= maximum: return True
    scores=[r['validation_accuracy'] for r in history]
    if n == nominal:
        # 最近三轮有净提升，且最后一轮没有低于前一轮，才视为仍在上升。
        return not (scores[-1] > scores[-3] and scores[-1] >= scores[-2])
    best=max(scores)
    last_high=max(i for i,s in enumerate(scores) if s==best and (i==0 or s>max(scores[:i])))
    # 若更早有高点，仍需从第20轮后给予完整的三轮观察窗口。
    return n-1-max(nominal-1,last_high) >= patience

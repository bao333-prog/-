"""逐轮每类验证统计；不改变普通交叉熵或原有最佳轮次判断规则。"""
import torch


def validate_with_class_metrics(model, loader, criterion, device, names):
    model.eval()
    matrix=torch.zeros((len(names),len(names)),dtype=torch.long)
    loss_sum=0.0
    total=0
    with torch.inference_mode():
        for images,labels in loader:
            labels=labels.to(device,non_blocking=True)
            logits=model(images.to(device,non_blocking=True))
            loss=criterion(logits,labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError('non-finite validation output')
            loss_sum+=loss.item()*labels.size(0)
            total+=labels.size(0)
            actual=labels.cpu()
            predicted=logits.argmax(1).cpu()
            matrix+=torch.bincount(actual*len(names)+predicted,minlength=len(names)**2).reshape(len(names),len(names))
    if total==0:
        raise ValueError('validation has no samples')
    per={n:dict(correct=int(matrix[i,i]),total=int(matrix[i].sum())) for i,n in enumerate(names)}
    recalls=[v['correct']/v['total'] for v in per.values() if v['total']]
    accuracy=float(matrix.diag().sum())/total
    return loss_sum/total,accuracy,dict(correct=int(matrix.diag().sum()),total=total,per_class=per,
                                       evaluated_classes=len(recalls),macro_accuracy=sum(recalls)/len(recalls),
                                       confusion_matrix=matrix.tolist())

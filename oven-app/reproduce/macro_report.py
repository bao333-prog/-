"""从逐类成绩生成宏平均报告；缺失基线时不编造下降值。"""
import argparse
import json
from pathlib import Path

def macro(per_class, names):
    values=[]
    for name in names:
        item=per_class[name]
        value=item['correct']/item['total'] if isinstance(item,dict) else item
        if value is None: raise ValueError('类别缺少评测样本: '+name)
        values.append(value)
    return sum(values)/len(values)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--baseline-metrics',type=Path,help='同一评测集父模型的 metrics.json，含 per_class_accuracy')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); report=json.loads(args.report.read_text(encoding='utf-8'))
    old=report['distillation']['teacher_class_names']; new=[n for n in report['class_names'] if n not in old]
    baseline=None
    if args.baseline_metrics:
        baseline=macro(json.loads(args.baseline_metrics.read_text(encoding='utf-8'))['per_class_accuracy'],old)
    lines=['# 宏平均评测报告','', '口径：现有验证集；非独立测试或烤箱实拍验收。','', '| 模型 | 旧类宏平均 | 新类宏平均 | 相对父B旧类下降（百分点） |','|---|---:|---:|---:|']
    for name,result in report['evaluations']['clean'].items():
        previous=macro(result['per_class'],old); added=macro(result['per_class'],new)
        drop=f'{100*(baseline-previous):.4f}' if baseline is not None else '待补父B逐类成绩'
        lines.append(f'| {name} | {previous:.4%} | {added:.4%} | {drop} |')
    lines += ['', '父B历史总体成绩292/293不能直接代替宏平均。提供同一集合的父B逐类指标后用 --baseline-metrics 补算。', '下降 =（父B旧类宏平均 − 当前模型旧类宏平均）×100。', '本报告仅重算历史指标，未重新运行图片评测。']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()

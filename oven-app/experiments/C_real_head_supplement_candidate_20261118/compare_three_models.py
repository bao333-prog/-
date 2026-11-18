"""同一167张候选图上比较原模型、第一候选和补充候选，不调参。"""
from pathlib import Path
import argparse,json,csv,hashlib,sys
import torch
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['source-code','data','base','previous','run']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.source_code.resolve()),str((a.source_code/'src').resolve())]
    from predict import load_classifier,image_tensor
    torch.set_num_threads(4)
    with (a.data/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['usage']=='comparison_candidate']
    assert len(rows)==167
    results={};predictions={};names=None
    models={'baseline':a.base,'previous_candidate':a.previous,'supplement_candidate':a.run/'model_C_real_head_candidate.pt'}
    def metrics(selected,pred):
        per={};correct=0
        for name in names:
            indices=[i for i in selected if rows[i]['food_label']==name];n=len(indices);right=sum(pred[i]==name for i in indices);correct+=right
            per[name]={'support':n,'correct':right,'accuracy':right/n if n else None}
        vals=[x['accuracy'] for x in per.values() if x['support']]
        return {'total':len(selected),'correct':correct,'overall_accuracy':correct/len(selected),'macro_accuracy_observed_classes':sum(vals)/len(vals),'observed_class_count':len(vals),'per_class':per}
    for key,path in models.items():
        model,cn,checkpoint=load_classifier(path)
        if names is None:names=cn
        assert names==cn;pred=[]
        for offset in range(0,len(rows),16):
            tensors=[]
            for r in rows[offset:offset+16]:
                with Image.open(a.data/r['image_path']) as im:tensors.append(image_tensor(im,checkpoint))
            with torch.inference_mode():logits=model(torch.cat(tensors))
            assert torch.isfinite(logits).all();pred.extend(names[i] for i in logits.argmax(1).tolist())
        predictions[key]=pred
        results[key]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_167':metrics(list(range(len(rows))),pred),'original_143':metrics([i for i,r in enumerate(rows) if r['food_label'] not in ['南瓜','可颂']],pred),'supplement_24':metrics([i for i,r in enumerate(rows) if r['food_label'] in ['南瓜','可颂']],pred)}
        print(key,results[key]['all_167']['correct'],'/167',flush=True)
    assert results['baseline']['original_143']['correct']==122
    assert results['previous_candidate']['original_143']['correct']==127
    report={'class_names':names,'results':results,'independent_test':False,'pumpkin_same_source_diagnostic':True,'selection':'fixed epoch; no tuning from comparison scores'}
    (a.run/'three_way_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.run/'three_way_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['image_file','true_label','sha256',*models])
        for i,r in enumerate(rows):w.writerow([Path(r['image_path']).name,r['food_label'],r['sha256'],*[predictions[k][i] for k in models]])
if __name__=='__main__':main()

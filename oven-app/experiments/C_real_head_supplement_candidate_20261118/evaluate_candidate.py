"""重新加载父模型与候选，完整网络预测同一比较集合；不进行训练或选模。"""
from pathlib import Path
import argparse,csv,json,hashlib,sys
import torch
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-code','data','base','run']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.source_code.resolve()),str((a.source_code/'src').resolve())]
    from predict import load_classifier,image_tensor
    torch.set_num_threads(4)
    parent,names,pc=load_classifier(a.base);candidate,cn,cc=load_classifier(a.run/'model_C_real_head_candidate.pt');assert cn==names
    with (a.data/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:all_rows=list(csv.DictReader(f))
    rows=[r for r in all_rows if r['usage']=='comparison_candidate'];assert len(rows)==167
    def evaluate(model,checkpoint):
        truth=[];pred=[]
        for offset in range(0,len(rows),16):
            tensors=[];batch=rows[offset:offset+16]
            for r in batch:
                with Image.open(a.data/r['image_path']) as im:tensors.append(image_tensor(im,checkpoint))
            with torch.inference_mode():logits=model(torch.cat(tensors))
            assert torch.isfinite(logits).all();pred.extend(logits.argmax(1).tolist());truth.extend(names.index(r['food_label']) for r in batch)
        matrix=[[0]*len(names) for _ in names]
        for y,h in zip(truth,pred):matrix[y][h]+=1
        per={}
        for i,n in enumerate(names):
            total=sum(matrix[i]);per[n]=dict(support=total,correct=matrix[i][i],accuracy=matrix[i][i]/total if total else None)
        accs=[v['accuracy'] for v in per.values() if v['support']]
        return dict(total=len(rows),correct=sum(y==h for y,h in zip(truth,pred)),overall_accuracy=sum(y==h for y,h in zip(truth,pred))/len(rows),macro_accuracy_observed_classes=sum(accs)/len(accs),observed_class_count=len(accs),per_class=per,confusion_matrix=matrix),pred
    bm,bp=evaluate(parent,pc);cm,cp=evaluate(candidate,cc)
    frozen=[k for k in parent.state_dict() if not k.startswith('classifier.3.')]
    assert all(torch.equal(parent.state_dict()[k],candidate.state_dict()[k]) for k in frozen)
    for name in cc['adaptation']['not_trained_classes']:
        i=names.index(name)
        for key in ['classifier.3.weight','classifier.3.bias']:assert torch.equal(parent.state_dict()[key][i],candidate.state_dict()[key][i])
    path=a.run/'report.json';report=json.loads(path.read_text(encoding='utf-8'))
    report.update(baseline=bm,candidate=cm,macro_change_pp=100*(cm['macro_accuracy_observed_classes']-bm['macro_accuracy_observed_classes']),evaluation_method='fresh parent and candidate full-network inference on all 167 images',frozen_parameters_unchanged=True,absent_class_output_rows_unchanged=True)
    assert report['base_sha256']==hashlib.sha256(a.base.read_bytes()).hexdigest()
    assert report['candidate_sha256']==hashlib.sha256((a.run/'model_C_real_head_candidate.pt').read_bytes()).hexdigest()
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.run/'comparison_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['image_file','true_label','base_prediction','candidate_prediction','sha256'])
        for r,b,c in zip(rows,bp,cp):w.writerow([Path(r['image_path']).name,r['food_label'],names[b],names[c],r['sha256']])
    print(json.dumps({'base':bm,'candidate':cm,'macro_change_pp':report['macro_change_pp']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

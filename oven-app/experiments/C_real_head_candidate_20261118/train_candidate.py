"""同十四类分类头微调；固定轮数，比较候选不参与训练或选模。"""
import argparse,copy,csv,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-code',type=Path,required=True,help='包含predict.py和src/的模型代码目录')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.source_code.resolve()),str((args.source_code/'src').resolve())]
    from predict import load_classifier,image_tensor
    from oven_food.model import build_model
    config=dict(seed=42,epochs=24,learning_rate=0.0003,batch_size=128,feature_batch_size=16,distillation_weight=0.5,temperature=2.0,threads=4,augmentation='none',trainable='classifier final linear layer only',selection='fixed final epoch; comparison set never used for selection')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    before_hash=hashlib.sha256(args.base.read_bytes()).hexdigest()
    assert before_hash=='d66f52f1d26d75364751f913c5e76d1002d71f1742aa10cb2e9ad02528d4548c'
    model,names,cp=load_classifier(args.base)
    with (args.data/'manifest.csv').open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
    assert len(names)==14
    train=[r for r in records if r['usage']=='train'];comparison=[r for r in records if r['usage']=='comparison_candidate']
    assert len(train)==2676 and len(comparison)==143
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in comparison}
    for record in records:
        assert record['food_label'] in names
        path=(args.data/record['image_path']).resolve();assert path.is_relative_to(args.data.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
    print('CHECKED: base hash, classes, image hashes and split isolation',flush=True)
    start=time.perf_counter()
    def features(rows):
        fs=[];ys=[];logits=[]
        for offset in range(0,len(rows),config['feature_batch_size']):
            batch=rows[offset:offset+config['feature_batch_size']];images=[]
            for record in batch:
                with Image.open(args.data/record['image_path']) as im:images.append(image_tensor(im,cp))
            tensor=torch.cat(images)
            with torch.inference_mode():
                x=model.features(tensor);x=model.avgpool(x).flatten(1)
                for layer in list(model.classifier.children())[:-1]:x=layer(x)
                reference=model.classifier[-1](x)
            fs.append(x.clone());logits.append(reference.clone());ys.extend(names.index(r['food_label']) for r in batch)
            if offset%256==0:print(f'FEATURES {offset+len(batch)}/{len(rows)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
        return torch.cat(fs),torch.tensor(ys),torch.cat(logits)
    x,y,teacher=features(train)
    # 比较候选只缓存输入特征；训练过程不读取其标签或结果。
    head=copy.deepcopy(model.classifier[-1]);head.train()
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    counts=torch.bincount(y,minlength=len(names));active=counts>0;absent=~active
    weights=torch.ones(len(names));weights[active]=len(y)/(int(active.sum())*counts[active].float())
    optimizer=torch.optim.Adam(head.parameters(),lr=config['learning_rate'])
    history=[]
    for epoch in range(1,config['epochs']+1):
        total=0.;correct=0;order=torch.randperm(len(y))
        for indices in order.split(config['batch_size']):
            pred=head(x[indices]);target=y[indices]
            ce=torch.nn.functional.cross_entropy(pred,target,weight=weights)
            t=config['temperature']
            kl=torch.nn.functional.kl_div((pred/t).log_softmax(1),(teacher[indices]/t).softmax(1),reduction='batchmean')*t*t
            loss=ce+config['distillation_weight']*kl
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():
                head.weight[absent]=initial['weight'][absent];head.bias[absent]=initial['bias'][absent]
            total+=float(loss.detach())*len(indices);correct+=int((pred.argmax(1)==target).sum())
        history.append(dict(epoch=epoch,train_loss=total/len(y),train_accuracy=correct/len(y)))
        print(json.dumps(history[-1]),flush=True)
    assert all(torch.equal(head.state_dict()[k][absent],v[absent]) for k,v in initial.items())
    frozen_before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('classifier.3.')}
    model.classifier[-1].load_state_dict(head.state_dict());model.eval()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen_before.items())
    output=copy.deepcopy(cp);output['model_state']=model.state_dict();output['epoch']=config['epochs']
    output['adaptation']={'id':'C_real_head_candidate_20261118','parent_sha256':before_hash,'configuration':config,'trained_classes':[names[i] for i in active.nonzero().flatten().tolist()],'not_trained_classes':[names[i] for i in absent.nonzero().flatten().tolist()],'train_manifest_sha256':hashlib.sha256((args.data/'manifest.csv').read_bytes()).hexdigest(),'independent_test':False}
    # 原检查点的旧验证/优化器记录不作为本轮记录继续保存。
    for key in list(output):
        if key not in {'model_state','model_name','class_names','image_size','preprocessing','epoch','adaptation'}:del output[key]
    destination=args.output/'model_C_real_head_candidate.pt';torch.save(output,destination)
    fresh,fresh_names,fresh_cp=load_classifier(destination);assert names==fresh_names
    cx,cy,_=features(comparison)
    with torch.inference_mode():
        base_logits=torch.nn.functional.linear(cx,initial['weight'],initial['bias'])
    with torch.inference_mode():new_logits=fresh.classifier[-1](cx)
    # 新进程结构重建后，逐图预处理+完整网络输出须与缓存特征通路一致。
    for i in range(min(8,len(comparison))):
        with Image.open(args.data/comparison[i]['image_path']) as im:tensor=image_tensor(im,fresh_cp)
        with torch.inference_mode():actual=fresh(tensor)[0]
        torch.testing.assert_close(actual,new_logits[i],atol=1e-4,rtol=1e-4)
    def metrics(logits):
        pred=logits.argmax(1);per={}
        matrix=torch.zeros(len(names),len(names),dtype=torch.int64)
        for a,b in zip(cy.tolist(),pred.tolist()):matrix[a,b]+=1
        for i,name in enumerate(names):
            mask=cy==i;n=int(mask.sum());correct=int((pred[mask]==i).sum())
            per[name]=dict(support=n,correct=correct,accuracy=correct/n if n else None)
        values=[v['accuracy'] for v in per.values() if v['support']]
        return dict(total=len(cy),correct=int((pred==cy).sum()),overall_accuracy=float((pred==cy).float().mean()),macro_accuracy_observed_classes=sum(values)/len(values),observed_class_count=len(values),per_class=per,confusion_matrix=matrix.tolist())
    baseline=metrics(base_logits);candidate=metrics(new_logits)
    report=dict(base_sha256=before_hash,candidate_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),base_model='C_real_20261105',candidate_model=output['adaptation']['id'],class_names=names,configuration=config,train_count=len(train),comparison_count=len(comparison),baseline=baseline,candidate=candidate,macro_change_pp=100*(candidate['macro_accuracy_observed_classes']-baseline['macro_accuracy_observed_classes']),frozen_parameters_unchanged=True,absent_class_output_rows_unchanged=True,full_reload_prediction_check=True,independent_test=False,missing_comparison_classes=[n for n,v in baseline['per_class'].items() if not v['support']],elapsed_seconds=time.perf_counter()-start)
    assert hashlib.sha256(args.base.read_bytes()).hexdigest()==before_hash
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    with (args.output/'comparison_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['image_file','true_label','base_prediction','candidate_prediction','sha256'])
        for r,a,b in zip(comparison,base_logits.argmax(1).tolist(),new_logits.argmax(1).tolist()):writer.writerow([Path(r['image_path']).name,r['food_label'],names[a],names[b],r['sha256']])
    print('RESULT',json.dumps({k:report[k] for k in ['macro_change_pp','elapsed_seconds','candidate_sha256']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

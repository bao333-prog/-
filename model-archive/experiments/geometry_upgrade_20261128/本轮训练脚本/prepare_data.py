"""先冻结全量来源，再分别重建224与320缓存；不修改原图/旧模型。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from support import RUN, ROOT, PREVIOUS, INTEGRATED, load, dump, sha
from data_ops import cache_row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int,required=True,choices=[224,320]);args=parser.parse_args()
    size=args.size;inventories={}
    for split in ['training','validation','test']:
        rows=load(PREVIOUS/f'{split}_base.json')
        inventories[split]=rows
    inventories['integrated']=load(PREVIOUS/'training_integrated.json')
    checks=load(PREVIOUS/'frozen_external_files.json')
    # 此前冻结清单包含旧模型；另外记录最新半截芦笋模型也必须不变。
    checks[str(PREVIOUS/'training/best.pt')]=sha(PREVIOUS/'training/best.pt')
    dump(RUN/'frozen_external_files.json',checks)
    for a,b in [('training','validation'),('training','test'),('validation','test')]:
        for key in ['family_id','prepared_pixel_sha256']:
            assert not ({r[key] for r in inventories[a]} & {r[key] for r in inventories[b]})
        sessions=lambda rows:{r['source_session'] for r in rows if r['environment']=='target'}
        assert not sessions(inventories[a]) & sessions(inventories[b])
    # 实际输入去重后再并发写入，避免原图同内容别名相互覆盖。
    unique={r['actual_input']:r for key in ['integrated','validation','test'] for r in inventories[key]}
    with ThreadPoolExecutor(max_workers=4) as pool:cached=list(pool.map(lambda r:cache_row(r,size),unique.values()))
    by_old={r['baseline_input']:r for r in cached}
    for key,rows in inventories.items():
        results=[]
        for row in rows:
            c=by_old[row['actual_input']]
            extra={k:c[k] for k in ['cache_source','cache_source_sha256','cache_source_size','cache_crop_xyxy',
                  'source_already_low_resolution','prepared_pixel_sha256','baseline_input','actual_input','prepared_sha256','prepared_resolution']}
            results.append({**row,**extra})
        dump(RUN/'data'/str(size)/(key+'.json'),results)
    current={key:load(RUN/'data'/str(size)/(key+'.json')) for key in inventories}
    for split in ['validation','test']:
        assert not ({r['prepared_pixel_sha256'] for r in current['integrated']} & {r['prepared_pixel_sha256'] for r in current[split]})
    report=dict(resolution=size,unique_cached_views=len(cached),original_rows=len(current['training']),
        integrated_rows=len(current['integrated']),validation_rows=len(current['validation']),test_rows=len(current['test']),
        raw_per_class=dict(Counter(r['label'] for r in current['training'])),
        original_proxy_low_resolution=sum(r['source_already_low_resolution'] for r in current['training'] if r['environment']=='proxy'),
        real_sources_rebuilt_from_raw=True,old_224_input_pixels_matched=(size==224),
        source_file_pixel_known_session_cross_split_overlap=0)
    dump(RUN/'data'/str(size)/'audit.json',report);print(report,flush=True)


if __name__=='__main__':main()

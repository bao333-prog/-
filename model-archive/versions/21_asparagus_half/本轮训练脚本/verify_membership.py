"""回归核验：用户提供且已通过审查的121张必须全部进入训练原图池。"""
import argparse
from inspect_new import RUN, load

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('manifest'); args=parser.parse_args()
    supplied=load(RUN/'new_inventory_pending.json')
    present={r['family_id'] for r in load(args.manifest)}
    missing=[r['source'] for r in supplied if r['sha256'] not in present]
    assert not missing, f'漏掉{len(missing)}张半截芦笋训练图片，首张：{missing[0]}'
    print('PASS：121张半截芦笋均已进入训练原图池')

"""复制记录并按用户提供的最长路径前缀映射迁移 JSON/CSV；不修改历史记录。"""
import argparse
import json
import shutil
from pathlib import Path

def replace_paths(value, mapping):
    if isinstance(value, str):
        normalized = value.replace('\\', '/')
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            old = old.replace('\\', '/').rstrip('/')
            if normalized == old or normalized.startswith(old + '/'):
                return new.rstrip('/\\') + normalized[len(old):]
        return value
    if isinstance(value, list): return [replace_paths(x, mapping) for x in value]
    if isinstance(value, dict): return {replace_paths(k,mapping):replace_paths(v,mapping) for k,v in value.items()}
    return value

def main():
    import csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--path-map', type=Path, required=True)
    args=parser.parse_args()
    mapping=json.loads(args.path_map.read_text(encoding='utf-8'))
    if args.destination.resolve().is_relative_to(args.source.resolve()):
        parser.error('destination must be outside source')
    shutil.copytree(args.source,args.destination)
    for path in args.destination.rglob('*'):
        if path.suffix=='.json':
            value=json.loads(path.read_text(encoding='utf-8'))
            path.write_text(json.dumps(replace_paths(value,mapping),ensure_ascii=False,indent=2),encoding='utf-8')
        elif path.suffix=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.reader(f))
            with path.open('w',encoding='utf-8-sig',newline='') as f: csv.writer(f).writerows(replace_paths(rows,mapping))
if __name__=='__main__': main()

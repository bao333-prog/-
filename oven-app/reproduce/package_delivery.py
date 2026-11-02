"""打包当前统一项目；默认不包括数据、权重、历史记录与环境。"""
import argparse
from pathlib import Path
import zipfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('输出已存在，请换一个文件名')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['frontend','backend','src','tests','reproduce','docs']:
            for path in (args.project/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc','.log'}:
                    archive.write(path,path.relative_to(args.project))
        for name in ['README.md','.gitignore','pyproject.toml','predict.py','requirements-validated.txt']:
            archive.write(args.project/name,name)
    with zipfile.ZipFile(args.output) as archive: assert archive.testzip() is None
    print(args.output)
if __name__=='__main__': main()

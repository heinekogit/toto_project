"""Refresh only v2.5 shadow from saved predictions; archive replaced shadow files."""
import argparse,json,hashlib,shutil,tempfile,os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from shadow_v25_operational import build,write,FILES,ROOT


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outdir',type=Path,default=ROOT/'data/purchase_reference');args=parser.parse_args();out=args.outdir
    manifest_path=out/'manifest.json';manifest=json.loads(manifest_path.read_text());rid=manifest['round_id']
    for name in ['next_predictions.csv','buyplan.csv']:
        if manifest['output_sha256'][name]!=digest(out/name):raise ValueError(f'{name}: saved manifest mismatch')
    protected={str(p):digest(p) for p in out.glob('*') if p.is_file() and p.name not in (*FILES,'manifest.json')}
    bundle=build(pd.read_csv(out/'next_predictions.csv'),pd.read_csv(out/'buyplan.csv'))
    now=datetime.now(ZoneInfo('Asia/Tokyo'));archive=out/'shadow_archive'/('v25_before_c95_'+now.strftime('%Y%m%dT%H%M%S%f'))
    with tempfile.TemporaryDirectory(prefix='.v25-c95-',dir=out) as temp:
        temp=Path(temp);write(bundle,temp,rid)
        archive.mkdir(parents=True)
        for name in (*FILES,'manifest.json'):
            if (out/name).exists():shutil.copy2(out/name,archive/name)
        manifest['shadow_v25_observation']=bundle[3]
        manifest['shadow_v25_refreshed_at']=now.isoformat()
        manifest['shadow_v25_previous_archive']=str(archive)
        for name in FILES:manifest['output_sha256'][name]=digest(temp/name)
        (temp/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        for name in FILES:os.replace(temp/name,out/name)
        os.replace(temp/'manifest.json',manifest_path)
    assert all(digest(Path(path))==h for path,h in protected.items())
    print(json.dumps({'round':rid,'mode':'C95','p13':bundle[3]['p13'],'retention':bundle[3]['retention'],'D_coverage':bundle[3]['D_count_coverage'],'archive':str(archive),'official_files_unchanged':True},ensure_ascii=False,indent=2))
    print(out/'shadow_buyplan_v25.html')

if __name__=='__main__':main()

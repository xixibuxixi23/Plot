"""Summarize expanded M4 runs using only their own experiment artifacts."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--write',action='store_true')
    args=p.parse_args()
    profiles=('villager_peaceful','zombie_melee','skeleton_swordsman','villager_defender')
    result={}
    for node,profile in zip(('rcz3','rcz4','rcz5','rcz6'),profiles):
        output=args.root/'models'/profile
        log=args.root/'logs'/f'{node}.log'
        item={'node':node,'profile':profile,'complete':(output/'COMPLETE.json').exists()}
        cache=args.root/'cache'/profile/'summary.json'
        if cache.exists(): item['data']=json.loads(cache.read_text())['splits']
        baseline=output/'baseline.json'
        if baseline.exists(): item['baseline']=json.loads(baseline.read_text())
        validation=output/'validation.jsonl'
        if validation.exists():
            rows=[json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
            def compact(row):
                return {key:({k:v for k,v in value.items() if k != 'horizon_loss'}
                             if isinstance(value,dict) else value) for key,value in row.items()}
            if rows:
                item['latest']=compact(rows[-1])
                item['best_loss']=compact(min(rows,key=lambda r:r['val']['loss']))
                item['best_attack']=compact(max(rows,key=lambda r:r['val']['attack_f1']))
        if log.exists(): item['log_tail']=[line[:600] for line in log.read_text().splitlines()[-2:]]
        if item['complete']: item['completion']=json.loads((output/'COMPLETE.json').read_text())
        result[profile]=item
        print(json.dumps(item),flush=True)
    if args.write:
        (args.root/'expanded-summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

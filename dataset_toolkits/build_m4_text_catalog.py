#!/usr/bin/env python3
"""Build deterministic shared/current text IDs for integrated M4."""
import argparse, hashlib, json
from pathlib import Path


def canonical(value): return " ".join(str(value or "").strip().split())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("indices",nargs="+")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    texts={""}
    for index in a.indices:
        for line in open(index):
            row=json.loads(line)
            if row["family"] == "language_builder":
                shared=canonical(row.get("shared_task_text",row.get("task_text","")))
                texts.update((shared,canonical(row.get("current_task_text",shared))))
    ordered=sorted(texts,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    payload={"schema_version":"plot-m4-text-v1","encoder":{
        "name":"google/t5-v1_1-base","max_length":64,"status":"embedding-cache-pending"},
        "texts":[{"text_id":i,"sha256":hashlib.sha256(text.encode()).hexdigest(),"text":text}
                 for i,text in enumerate(ordered)]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({"output":str(a.output),"texts":len(ordered)}))
if __name__=="__main__":main()

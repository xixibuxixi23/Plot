#!/usr/bin/env python3
"""Encode the M4 text catalog once with the original frozen T5 encoder."""
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, T5EncoderModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("catalog",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--model",default="google/t5-v1_1-base")
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--max-length",type=int,default=64);a=p.parse_args()
    catalog=json.loads(a.catalog.read_text());texts=[row["text"] for row in catalog["texts"]]
    tokenizer=AutoTokenizer.from_pretrained(a.model);model=T5EncoderModel.from_pretrained(a.model)
    device=torch.device(a.device);model=model.eval().to(device)
    hidden=[];masks=[]
    with torch.no_grad():
        for start in range(0,len(texts),a.batch_size):
            token=tokenizer(texts[start:start+a.batch_size],padding="max_length",truncation=True,
                            max_length=a.max_length,return_tensors="pt")
            mask=token["attention_mask"];token={k:v.to(device) for k,v in token.items()}
            hidden.append(model(**token).last_hidden_state.bfloat16().cpu())
            masks.append(mask.byte())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    save_file({"encoder_hidden":torch.cat(hidden).contiguous(),
               "attention_mask":torch.cat(masks).contiguous()},str(a.output))
    catalog["encoder"].update(status="complete",cache_file=a.output.name,
                              hidden_size=int(model.config.d_model))
    a.catalog.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n')
if __name__=="__main__":main()

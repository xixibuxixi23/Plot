#!/usr/bin/env python3
"""Build an item vocabulary from training metadata without reading validation labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LEGACY_ITEMS = [
    '', 'mcl_core:brick_block', 'mcl_core:cobble', 'mcl_core:dirt',
    'mcl_core:glass', 'mcl_core:stonebrick', 'mcl_core:tree', 'mcl_core:wood',
    'mcl_ocean:sea_lantern', 'mcl_tools:pick_iron',
    'textagent_task:npc_axe', 'textagent_task:npc_sword',
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();index=json.loads(Path(args.index).read_text())
    root=Path(index['dataset_root']);names=set(LEGACY_ITEMS)
    for record in index['splits']['train']:
        episode=root/record['path']
        metadata=json.loads((episode/'training_metadata.json').read_text())
        names.update(str(name) for name in metadata.get('item_vocabulary',{}))
        manifest=json.loads((episode/'manifest.json').read_text())
        for loadout in (manifest.get('agent_weapons') or {}).values():
            if loadout.get('weapon_item'):names.add(str(loadout['weapon_item']))
    ordered=LEGACY_ITEMS+[name for name in sorted(names) if name not in LEGACY_ITEMS]
    payload={'schema_version':'plot-item-vocabulary-v1','source_split':'train',
             'items':{name:index for index,name in enumerate(ordered)}}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'items={len(ordered)} output={output}')


if __name__=='__main__':main()

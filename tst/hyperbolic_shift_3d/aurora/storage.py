#!/usr/bin/env python3
"""Conservative campaign-only footprint accounting; never deletes files."""
import os
from pathlib import Path

LIMIT_BYTES = 5_000_000_000


def policy(plan):
    values=plan.get('storage')
    if not isinstance(values,dict) or values.get('limit_bytes')!=LIMIT_BYTES:
        raise ValueError('Plan must declare the fixed 5,000,000,000-byte home campaign cap')
    for key in ['planned_new_bytes','atomic_checkpoint_reserve_bytes']:
        if type(values.get(key)) is not int or values[key]<0:
            raise ValueError('Plan needs an explicit nonnegative storage estimate: '+key)
    if any(c.get('kind')!='check' for c in plan['commands']) and values['atomic_checkpoint_reserve_bytes']==0:
        raise ValueError('Evolution requires reserved space for an atomic checkpoint copy')
    if values['planned_new_bytes']+values['atomic_checkpoint_reserve_bytes']>LIMIT_BYTES:
        raise ValueError('Planned outputs and checkpoint reserve exceed the campaign cap')
    return values


def footprint(root):
    root=Path(root)
    seen=set()
    total=0
    files=0
    for directory,children,names in os.walk(root,followlinks=False):
        for path in [Path(directory),*[Path(directory)/name for name in names],
                     *[Path(directory)/name for name in children if (Path(directory)/name).is_symlink()]]:
            try:
                info=path.lstat()
            except FileNotFoundError:
                continue
            identity=(info.st_dev,info.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            # Count logical bytes too: sparse files cannot hide planned growth.
            total+=max(info.st_size,getattr(info,'st_blocks',0)*512)
            files+=1
    return dict(bytes=total,entries=files)


def require_capacity(root,plan,prospective=False):
    values=policy(plan)
    measured=footprint(root)
    reserved=values['atomic_checkpoint_reserve_bytes']
    planned=values['planned_new_bytes'] if prospective else 0
    result=dict(measured,limit_bytes=LIMIT_BYTES,reserve_bytes=reserved,planned_bytes=planned,
                headroom_bytes=LIMIT_BYTES-measured['bytes']-reserved-planned)
    if result['headroom_bytes']<0:
        raise RuntimeError('Campaign storage cap/reserve exceeded: '+str(result))
    return result

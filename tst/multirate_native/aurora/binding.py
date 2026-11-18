"""Compact MPI rank binding to Aurora GPU tiles or complete devices."""
import re
import shutil


def launcher(ranks, whole_gpu=False):
    masks = ('1-16:17-32:33-48:53-68:69-84:85-100' if whole_gpu else
             '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100')
    cores = masks.split(':')
    ppn = min(ranks,len(cores))
    wrapper = 'gpu_dev_compact.sh' if whole_gpu else 'gpu_tile_compact.sh'
    command = ['mpiexec','-n',str(ranks),'-ppn',str(ppn),'--cpu-bind=list:'+':'.join(cores[:ppn]),
               shutil.which(wrapper)]
    if whole_gpu:
        command += ['env','EnableImplicitScaling=1']
    return command


def rank_devices(text, ranks, whole_gpu=False):
    mask = r'(\d+)' if whole_gpu else r'(\d+\.\d+)'
    rows=re.findall(r'^([\w.-]+) tile='+mask+r' rank=(\d+)$',text,re.M)
    binding={int(rank):(host,device) for host,device,rank in rows}
    if (len(rows)!=ranks or set(binding)!=set(range(ranks)) or
            len(set(binding.values()))!=ranks):
        raise ValueError('incomplete or duplicate Aurora rank/device binding')
    devices={(host,device.split('.')[0]) for host,device in binding.values()}
    if len(devices)!=(ranks if whole_gpu else ranks//2):
        raise ValueError('unexpected physical GPU count')
    if not whole_gpu:
        for rank in range(0,ranks,2):
            host,tile=binding[rank]
            device=tile.split('.')[0]
            if tile!=device+'.0' or binding[rank+1]!=(host,device+'.1'):
                raise ValueError('adjacent ranks are not bound to sibling tiles')
    return binding

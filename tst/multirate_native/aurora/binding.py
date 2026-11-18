"""Compact binding for one MPI process per Aurora GPU tile."""
import shutil


def launcher(ranks):
    ppn = min(ranks, 12)
    cores = '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'.split(':')
    return ['mpiexec', '-n', str(ranks), '-ppn', str(ppn), '--cpu-bind=list:'+':'.join(cores[:ppn]),
            shutil.which('gpu_tile_compact.sh')]

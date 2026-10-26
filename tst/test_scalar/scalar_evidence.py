"""Read immutable scalar runs and checkpoint continuations as one lineage."""
import csv,json,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np

@lru_cache(maxsize=64)
def _checkpoint_digest(path,size,mtime):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def parent(path):
 record=Path(path)/'restart-invocation.json'
 if not record.exists():return None
 invocation=json.loads(record.read_text());original=Path(invocation['parent'])
 if original.exists():return original
 mapping=Path(path)/'analysis-parent.json'
 if not mapping.exists():raise FileNotFoundError('Collected continuation needs a verified local parent mapping: '+str(path))
 link=json.loads(mapping.read_text())
 if link['native_parent']!=str(original) or link['restart_invocation_sha256']!=hashlib.sha256(record.read_bytes()).hexdigest():raise RuntimeError('Parent mapping does not bind this native continuation')
 resolved=Path(link['local_parent']);checkpoint=resolved/'checkpoint.f64';stat=checkpoint.stat()
 if _checkpoint_digest(str(checkpoint),stat.st_size,stat.st_mtime_ns)!=invocation['checkpoint_sha256']:raise RuntimeError('Mapped parent checkpoint differs from native restart input')
 return resolved
def snapshot(path,t):
 file=Path(path)/f'fields-t{t:.3f}.f64'
 if file.exists():return np.fromfile(file).reshape(-1,12)
 previous=parent(path)
 if previous is None:raise FileNotFoundError(file)
 return snapshot(previous,t)
def history(path):
 previous=parent(path);rows=history(previous) if previous is not None else {}
 rows.update({(float(r['time']),r['region']):r for r in csv.DictReader((Path(path)/'history.csv').open())})
 return rows
def waveform(path):
 file=Path(path)/'waveform.csv';header=file.open().readline().strip();names=('time','real22','imag22')
 # Initial checkpoint drivers omitted the header on a continuation. Preserve
 # those raw files and interpret their three numeric columns explicitly.
 rows=np.atleast_1d(np.genfromtxt(file,delimiter=',',names=True) if header.startswith('time,') else np.genfromtxt(file,delimiter=',',names=names))
 previous=parent(path)
 if previous is not None:
  old=waveform(previous);rows=np.concatenate((old[old['time']<rows['time'][0]-1e-12],rows))
 return rows

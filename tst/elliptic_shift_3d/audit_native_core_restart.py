#!/usr/bin/env python3
"""Bounded negative controls of actual native static-core restart topology.
Only new copies of a completed checkpoint are altered; source evidence is read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import struct
import subprocess
import time
from analyze_native_wormhole import checksum


def main():
 p=argparse.ArgumentParser();p.add_argument('--controls',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 binary=root/'build-shift-3d/src/athena';source=a.controls.resolve()/'full/rst/native_wormhole.00001.rst';sidecar=Path(str(source)+'.composite');original=sidecar.read_bytes();offset=0
 def skip_string():
  nonlocal offset
  n=struct.unpack_from('=Q',original,offset)[0];offset+=8+n
 skip_string();skip_string();offset+=4+4+8;n=struct.unpack_from('=Q',original,offset)[0];offset+=8+8*n
 bulk_offset=offset;record_offset=offset+4+4+8
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 manifest={'scope':'Actual static-core checkpoint count/map/bounds binding negative controls only; no new evolution or scientific qualification','binary_sha256':sha(binary),'source_native_RST_sha256':sha(source),'source_sidecar_sha256':sha(sidecar),'cases':[]}
 for name,position,fmt,value,expected in [('count',bulk_offset+4,'i',7,'core topology/count mismatch'),('map',record_offset+4,'i',99,'core pack map mismatch'),('bounds',record_offset+12,'d',-.24,'core bounds mismatch')]:
  folder=out/name;folder.mkdir();rst=folder/source.name;shutil.copyfile(source,rst);data=bytearray(original);struct.pack_into('='+fmt,data,position,value);struct.pack_into('=Q',data,len(data)-8,checksum(data[:-8]));Path(str(rst)+'.composite').write_bytes(data)
  command=[str(binary),'-r',str(rst),'z4c/composite_restart_file='+str(rst)+'.composite','-d',str(folder/'output')];start=time.monotonic()
  with (folder/'stdout').open('w') as stdout,(folder/'stderr').open('w') as stderr:
   process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr,start_new_session=True)
   try:code=process.wait(timeout=180)
   except subprocess.TimeoutExpired:
    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    code=124
  text=(folder/'stdout').read_text()+(folder/'stderr').read_text();row={'name':name,'command':command,'maximum_seconds':180,'owned_process_group':process.pid,'seconds':time.monotonic()-start,'exit_code':code,'expected_diagnostic':expected,'rejected_for_expected_reason':code!=0 and code!=124 and expected in text};manifest['cases'].append(row);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');assert row['rejected_for_expected_reason'];print(name,code,row['seconds'],flush=True)
 manifest['source_unchanged']=sha(source)==manifest['source_native_RST_sha256'] and sha(sidecar)==manifest['source_sidecar_sha256'];manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256'];assert manifest['source_unchanged'] and manifest['binary_unchanged'];manifest['script_sha256']=sha(Path(__file__));(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()

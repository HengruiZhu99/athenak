#!/usr/bin/env python3
"""Export committed scientific sources for a new, separately built snapshot."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
paths = ['src', 'tst', 'CMakeLists.txt', 'config.hpp.in', 'LICENSE', 'README.md']
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root)
def digest(data):
    return hashlib.sha256(data).hexdigest()

revision = git('rev-parse', 'HEAD').decode().strip()
if git('diff', '--name-only', 'HEAD', '--', *paths).strip():
    raise RuntimeError('Scientific source has uncommitted changes')
if git('ls-files', '--others', '--exclude-standard', '--', *paths).strip():
    raise RuntimeError('Scientific source has untracked files')
archive = out / 'hyperbolic-aurora-source-v23.tar.gz'
manifest = out / 'source-v23-non-kokkos-manifest.json'
provenance = out / 'source-v23-provenance.json'
if any(p.exists() for p in (archive, manifest, provenance)):
    raise RuntimeError('Refusing to overwrite a frozen export')
with archive.open('xb') as stream:
    subprocess.run(['git', 'archive', '--format=tar.gz', '--prefix=source-v23/',
                    revision, *paths], cwd=root, stdout=stream, check=True)
mapping = {}
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    for member in members:
        if member.isdir():
            continue
        if not member.isfile() or '..' in Path(member.name).parts:
            raise RuntimeError('Unexpected non-regular export member')
        mapping[member.name] = digest(stream.extractfile(member).read())
# The remote verified extractor accepts only the manifest's regular members.
# Repack from verified committed bytes to omit archive directory entries.
temporary = archive.with_suffix('.repack')
with tarfile.open(archive, 'r:gz') as source, tarfile.open(temporary, 'w:gz') as target:
    for member in source.getmembers():
        if member.isfile():
            target.addfile(member, source.extractfile(member))
temporary.replace(archive)
manifest.write_text(json.dumps(mapping, sort_keys=True, indent=2) + '\n')
provenance.write_text(json.dumps(dict(
    source_commit=revision, files=len(mapping), archive_sha256=digest(archive.read_bytes()),
    kokkos_pin='6739bc623081648af9e752b616d9671527922cbf',
    scope='V11 sparse stage exchange with fine boundary/error-synchronization profiling; unchanged physical equations, interpolation arithmetic, and checkpoint schema. Experimental performance candidate.'
), indent=2) + '\n')
print(provenance.read_text())

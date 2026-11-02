"""Packaging/freezing tests; synthetic files are not GPU validation evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import control
import native_transition_campaign as c
import native_transition_prepare as p


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        launcher=self.root/p.LAUNCHER;launcher.mkdir(parents=True)
        source=self.root/p.SOURCE_DIR/'src/example.hpp';source.parent.mkdir(parents=True)
        source.write_text('synthetic source')
        control.atomic(self.root/c.MANIFEST,{str(source.relative_to(self.root)):control.digest(source)})
        for name in p.FILES:(launcher/name).write_text('synthetic launcher '+name)
        control.atomic(launcher/'package-provenance.json',dict(source_commit=c.SOURCE_COMMIT,
            source_manifest_sha256=control.digest(self.root/c.MANIFEST),
            files={name:p.identity(launcher/name) for name in p.FILES}))
        logs=self.root/'build-logs/native-transition-v1';logs.mkdir(parents=True)
        control.atomic(logs/'result.json',dict(commands=[dict(returncode=0,timed_out=False,storage_error=None)]*2))
        self.cache=self.root/p.BUILD_DIR/'CMakeCache.txt';self.cache.parent.mkdir()
        self.cache.write_text('\n'.join(k+':STRING='+v for k,v in dict(
            Athena_ENABLE_MPI='ON',Athena_SINGLE_PRECISION='OFF',Athena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS='ON',
            Kokkos_ENABLE_SYCL='ON',Kokkos_ENABLE_SERIAL='ON',Kokkos_ARCH_INTEL_PVC='ON',
            CMAKE_BUILD_TYPE='Release',CMAKE_CXX_COMPILER='/opt/bin/icpx',
            CMAKE_CXX_FLAGS='-fp-model=precise -fno-sycl-id-queries-fit-in-int',CMAKE_CXX_FLAGS_RELEASE='-O2').items()))
        self.flags=[]
        for v in c.VARIANTS.values():
            for kind in ['mpi','checkpoint_constraints']:
                target='hyperbolic_shift_3d_'+kind+'_native_'+v['suffix']
                binary=self.root/p.BUILD_DIR/'tst/hyperbolic_shift_3d'/target
                binary.parent.mkdir(parents=True,exist_ok=True);binary.write_text('synthetic '+target)
                flags=binary.parent/'CMakeFiles'/(target+'.dir')/'flags.make'
                flags.parent.mkdir(parents=True)
                defines=['NATIVE_WORMHOLE','DIRECT_CHI','DIRECT_LAPSE']
                defines += [k for k,b in [('SHARED_RESTORATION',v['shared']),('CONSISTENT_KO',v['ko10']),('SCRI_FIRST6',v['scri6'])] if b]
                flags.write_text('CXX_DEFINES = '+' '.join('-DHYPERBOLIC_'+k+'=1' for k in defines))
                self.flags.append(flags)

    def test_freeze_all_ten_targets_and_no_overwrite(self):
        result=p.freeze(self.root)
        self.assertEqual(len(result['binaries']),5);self.assertEqual(len(result['diagnostic_binaries']),5)
        self.assertEqual(len(result['flags']),10)
        provenance=json.loads((self.root/p.LAUNCHER/'provenance.json').read_text())
        self.assertEqual(provenance['files'][Path(c.SCIENCE).name],p.identity(self.root/c.SCIENCE))
        with self.assertRaises(ValueError):p.freeze(self.root)

    def test_reject_wrong_compilation_and_failed_build(self):
        original=self.cache.read_text()
        for old,new in [('MPI:STRING=ON','MPI:STRING=OFF'),('SYCL:STRING=ON','SYCL:STRING=OFF'),
                        ('PVC:STRING=ON','PVC:STRING=OFF'),('RELEASE:STRING=-O2','RELEASE:STRING=-Ofast'),
                        ('/opt/bin/icpx','/usr/bin/c++')]:
            self.cache.write_text(original.replace(old,new))
            with self.assertRaises(ValueError):p.freeze(self.root)
            self.assertFalse((self.root/c.SCIENCE).exists())
        self.cache.write_text(original)
        for flags in self.flags:
            original_flags=flags.read_text()
            for mutation in [original_flags.replace('NATIVE_WORMHOLE=1','NATIVE_WORMHOLE=0'),
                             original_flags.replace('DIRECT_CHI=1','DIRECT_CHI=0'),
                             original_flags.replace('DIRECT_LAPSE=1','DIRECT_LAPSE=0'),
                             original_flags+' -DHYPERBOLIC_SCRI_FIRST6='+('0' if 'SCRI_FIRST6' in original_flags else '1')]:
                flags.write_text(mutation)
                with self.assertRaises(ValueError):p.freeze(self.root)
            flags.write_text(original_flags)
        record=self.root/'build-logs/native-transition-v1/result.json'
        healthy=json.loads(record.read_text())
        for key,value in [('returncode',1),('timed_out',True),('storage_error','cap')]:
            bad=copy.deepcopy(healthy);bad['commands'][1][key]=value;control.atomic(record,bad)
            with self.assertRaises(ValueError):p.freeze(self.root)

    def test_package_change_detected_and_build_stops_before_commands(self):
        for file in [self.root/p.SOURCE_DIR/'src/example.hpp',self.root/p.LAUNCHER/'control.py']:
            original=file.read_text();file.write_text(original+' changed')
            with self.assertRaises(ValueError):p.verify_package(self.root)
            file.write_text(original)
        # An existing build is preserved, not silently retried or overwritten.
        with patch.object(Path,'home',return_value=self.root),patch.object(p.worker,'run_bounded') as run:
            with self.assertRaises(ValueError):p.build(self.root)
            run.assert_not_called()


if __name__=='__main__':unittest.main()

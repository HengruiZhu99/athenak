import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import smooth_campaign as c


class CheckpointTests(unittest.TestCase):
    def test_requires_admitted_gw1_configuration_and_verified_finite_payload(self):
        n=16;nc=2;ns=6*n*n*31
        data=bytearray(struct.pack('<16s4i3Q2d',b'HYP3D_MPI_GW1',n,31,2,1,nc,ns,2001,1.,.0005)+c.DT_CONFIG)
        data+=struct.pack('<d',1.)*(2*(31*nc+ns))
        with tempfile.TemporaryDirectory() as tmp, patch.object(c,'cells',return_value=nc):
            p=Path(tmp)/'input'
            p.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
            self.assertEqual(c.inspect_checkpoint(p,n,sha)['time'],1.)
            with self.assertRaises(ValueError):c.inspect_checkpoint(p,n,'0'*64)
            for offset,fmt,value in [(0,'<16s',b'HYP3D_MPI_DT1'),(28,'<i',0),(56,'<d',.9),
                    (64,'<d',.001),(144,'<i',3),(152,'<d',-1.),(160,'<d',float('nan'))]:
                bad=bytearray(data);struct.pack_into(fmt,bad,offset,value);p.write_bytes(bad)
                with self.assertRaises(ValueError):c.inspect_checkpoint(p,n,hashlib.sha256(bad).hexdigest())


if __name__=='__main__':unittest.main()

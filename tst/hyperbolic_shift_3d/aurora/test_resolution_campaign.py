import hashlib,struct,unittest
from array import array
from unittest.mock import patch
import resolution_campaign as r
class MigrationTests(unittest.TestCase):
    def fixture(self):
        n=16;nc=2;ns=6*n*n*31
        data=bytearray(struct.pack('<16s4i3Q2d',b'HYP3D_MPI_DT1',n,31,2,1,nc,ns,2001,1.,.0005)+r.DT_CONFIG)
        values=array('d',(i*.0001 for i in range(2*(31*nc+ns))));data+=values.tobytes()
        return data,nc,ns
    def test_compensation_units_only(self):
        data,nc,ns=self.fixture()
        with patch.object(r,'cells',return_value=nc):out=r.convert(data,hashlib.sha256(data).hexdigest(),16)
        before=array('d');before.frombytes(data[152:]);after=array('d');after.frombytes(out[168:]);reset=set(range(31*nc,62*nc,31))|set(range(31*nc+18,62*nc,31))|set(range(62*nc+ns,62*nc+2*ns,31))
        self.assertEqual(len(before),len(after))
        for i,(a,b) in enumerate(zip(before,after)):self.assertEqual(b,0 if i in reset else a)
        self.assertEqual(out[72:168],r.config('AC1_to20M'))
    def test_rejects_bad_checkpoint_identity(self):
        for offset,fmt,value in [(28,'<i',0),(56,'<d',.9),(64,'<d',.001),(144,'<i',3)]:
            data,nc,ns=self.fixture();struct.pack_into(fmt,data,offset,value)
            with patch.object(r,'cells',return_value=nc),self.assertRaises(ValueError):r.convert(data,hashlib.sha256(data).hexdigest(),16)
    def test_requires_verified_bytes(self):
        data,_,_=self.fixture()
        with self.assertRaises(ValueError):r.convert(data,'0'*64,16)
if __name__=='__main__':unittest.main()

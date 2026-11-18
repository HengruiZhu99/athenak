"""Build a mesh-only audit against the same agreement stencil builder as Scheme A."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multirate_ownership'))
from qualify_migration import link_main


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--cxx',required=True)
    parser.add_argument('input',type=Path)
    args=parser.parse_args()
    work=args.build.resolve()/'tst/multirate_native/agreement'
    exe=link_main(args.build,args.cxx,Path(__file__).with_name('agreement_membership.cpp'),work)
    subprocess.run([str(exe),str(args.input.resolve())],check=True,cwd=work)


if __name__=='__main__':
    main()

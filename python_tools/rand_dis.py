import numpy as np
from ase.io import read ,write
from ase.build import make_supercell
import os
import sys
in_files=sys.argv[1:]
out_file="rand"
os.system("rm "+out_file+".xyz")
sc=[[4,0,0],[0,4,0],[0,0,1]]
for f in in_files:
    aa=read(f)
    a=make_supercell(aa,sc)
    spos=a.get_scaled_positions()
    cell=a.get_cell()
    strain=[1.05,1.02,1.0,0.98,0.95]
    stru=[]
    for s in strain:
        _a=a.copy()
        _a.set_cell(cell*s)
        _a.set_scaled_positions(spos)
        stru.append(_a)
    for a in stru:
        write(out_file+".xyz",a,append=True,format="extxyz")
        for i in range(8):
            a_=a.copy()
            a_.rattle((i+1)*0.01,rng=np.random)
            write(out_file+".xyz",a_,append=True,format="extxyz")
    


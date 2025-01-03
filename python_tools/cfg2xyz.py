#!/usr/bin/env python
import os
import numpy as np
import sys
from ase.data import atomic_numbers
##import sys

os.system("rm in.xyz")


map_dic={}
ele=[]

if len(sys.argv)>2:
    print (f"num of ele: {int(len(sys.argv)-2)}")
    ele=sys.argv[2:]

order= True
if (order):
    ele_order = {}
    for i in ele:
        ele_order.update({i: atomic_numbers[i]})
    ele = sorted(ele, key=lambda x: ele_order[x])


for i in range(len(ele)):
    map_dic.update({i:ele[i]})

cfgs=sys.argv[1]
#cfgs="./trainset.cfg"

with open(cfgs) as f:
    lines = f.readlines()
    cfgcnt = 0
    for line in lines:
        if line == ' Size\n':
            cfgcnt += 1

    cntr=1
    for i in range(len(lines)):
        if lines[i] != 'BEGIN_CFG\n':
            continue
#        else:
#            print("reading cfg#"+str(cntr+1))
        size = int(lines[i+2].split()[0])
        energy=float(lines[i+9+size].split()[0])
        lat=lines[i+4].split()+lines[i+5].split()+lines[i+6].split()
        for k in range(len(lat)):
            lat[k]=float(lat[k])
        stress=lines[i+11+size].split()
        for l in range(len(stress)):
            stress[l]=float(stress[l])
        _stress=[stress[0],stress[5],stress[4],stress[5],stress[1],stress[3],stress[4],stress[3],stress[2]]


        nruter = []
        for j in range(size):
            tmp = []
            words = lines[i+8+j].split()
            tmp.append(map_dic[int(words[1])])
            tmp.append(float(words[2]))
            tmp.append(float(words[3]))
            tmp.append(float(words[4]))
            tmp.append(float(words[5]))
            tmp.append(float(words[6]))
            tmp.append(float(words[7]))
            nruter.append(tmp)


        with open('in.xyz','a') as ff:
            ff.write(str(size)+'\n')
            ff.write("""Lattice="{: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} " Properties=species:S:1:pos:R:3:forces:R:3 energy={: 12.8f} virial= "{: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f}" pbc="T T T" \n""".format(*lat,energy,*_stress))
            for k in range (size):
                ff.write('{} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f} {: 12.8f}\n'.format(*nruter[k]))
        print(cntr)
        cntr += 1
print(map_dic)

import sys
import os
from ase.io import iread, write
from ase.data import atomic_numbers
inxyz=sys.argv[1]
if len(sys.argv)>2:
    print (f"num of ele: {int(len(sys.argv)-2)}")
    sub_ele=sys.argv[2:]
sub_set=set(sub_ele)
outfile="_".join(sub_ele)+".xyz"
os.system("rm "+ outfile)
n=0
fin=iread(inxyz)
map_dic={}
#ele=['Na', 'Pb', 'K', 'Nb', 'La', 'Tm', 'Co', 'Ca', 'Sc', 'Tl', 'Lu', 'As', 'N', 'Au', 'Ho', 'Ti', 'Cd', 'W', 'Gd', 'Mg', 'Dy', 'Bi', 'Ni', 'B', 'Sn', 'V', 'Ba', 'Pd', 'Tb', 'In', 'Ta', 'P', 'Zn', 'Y', 'Zr', 'Pt', 'Si', 'Al', 'Sb', 'Ru', 'Ce', 'Ge', 'Th', 'Er', 'Te', 'Rh', 'Li', 'Ga', 'Sr', 'Ag', 'Hf', 'Os', 'Se', 'Fe']
#ele=["Al","Ga","In","As"]

b=list(fin)
ele_set=set()

for i in b:
    ele_set = ele_set|set(i.get_chemical_symbols())
    if sub_set>=set(i.get_chemical_symbols()):
            write(outfile,i,format="extxyz",append="True")
            n+=1
ele=list(ele_set)


order= True
if (order):
    ele_order = {}
    for i in ele:
        ele_order.update({i: atomic_numbers[i]})
    ele = sorted(ele, key=lambda x: ele_order[x])


for i in range(len(ele)):
    map_dic.update({ele[i]:i})

print(f"num_sub:{n}")

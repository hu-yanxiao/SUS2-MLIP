import sys
from ase.io import iread
from ase.data import atomic_numbers
inxyz=sys.argv[1]
fin=iread(inxyz)
map_dic={}
#ele=['Na', 'Pb', 'K', 'Nb', 'La', 'Tm', 'Co', 'Ca', 'Sc', 'Tl', 'Lu', 'As', 'N', 'Au', 'Ho', 'Ti', 'Cd', 'W', 'Gd', 'Mg', 'Dy', 'Bi', 'Ni', 'B', 'Sn', 'V', 'Ba', 'Pd', 'Tb', 'In', 'Ta', 'P', 'Zn', 'Y', 'Zr', 'Pt', 'Si', 'Al', 'Sb', 'Ru', 'Ce', 'Ge', 'Th', 'Er', 'Te', 'Rh', 'Li', 'Ga', 'Sr', 'Ag', 'Hf', 'Os', 'Se', 'Fe']
#ele=["Al","Ga","In","As"]

b=list(fin)
ele_set=set()
for i in b:
    ele_set = ele_set|set(i.get_chemical_symbols())

ele=list(ele_set)


order= True
if (order):
    ele_order = {}
    for i in ele:
        ele_order.update({i: atomic_numbers[i]})
    ele = sorted(ele, key=lambda x: ele_order[x])


for i in range(len(ele)):
    map_dic.update({ele[i]:i})
#b=list(fin)
ff = open("INPUT.cfgs","w")
for i in range(len(b)):
    print(i)
    atoms=b[i]
    eles=atoms.get_chemical_symbols()
    nat=len(eles)
    cell = atoms.get_cell()
    pos = atoms.get_positions()
    force= atoms.get_forces()
    en=atoms.get_potential_energy()
    virial=atoms.info['virial']
    ff.write("""BEGIN_CFG\n""")
    ff.write(""" Size\n""")
    ff.write("""  {:6} \n""".format(nat))
    ff.write(""" Supercell \n""")
    ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[0, 0], cell[0, 1], cell[0, 2]))
    ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[1, 0], cell[1, 1], cell[1, 2]))
    ff.write("""{:15.10f} {:15.10f} {:15.10f}\n""".format(cell[2, 0], cell[2, 1], cell[2, 2]))
    ff.write("""AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n""")
    for i in range(nat):
        ff.write(
            """ {:6} {:6} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n""".format(i + 1, map_dic[eles[i]], pos[i, 0], pos[i, 1], pos[i, 2],
                                                                                         force[i,0], force[i,1], force[i,2]))
    ff.write("""Energy \n""")
    ff.write(f"""\t{en} \n""")
    ff.write("""PlusStress:  xx          yy          zz          yz          xz          xy \n""")
    ff.write(f"\t{virial[0,0]}  \t{virial[1,1]}  \t{virial[2,2]}  \t{virial[1,2]}  \t{virial[0,2]}  \t{virial[0,1]} \n")
    ff.write("""END_CFG \n""")
ff.close()
print(map_dic)

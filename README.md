# SUS2-MLIP
SUS2-MLIP:**S**uper-Linear Machine Learning Interatomic Potentials with Physics-Informed **U**niversal **S**caling and **U**ltra-**S**mall Parameterization. 
This model is realized through the modified mlip-2 package (https://gitlab.com/ashapeev/mlip-2/-/tree/master?ref_type=heads).
![image](https://github.com/user-attachments/assets/0aaaa76f-b4f8-459e-b8ec-1ddc08849693)

# Installation
You can install SUS2-MLIP by running:
```bash
 ./configuration  
 make mlp  ## get bin/sus2mlip
 make libinterface ## get lib/libinterface.a for external tool e.g. LAMMPS and pysus2mlip
```
# training datasets
Like the original mlip-2 package, SUS2-MLIP reads material structures and their properties from .cfg files.  
Format of cfg file: 
```bash
BEGIN_CFG  
 Size  
     192  
 Supercell  
  16.7849720000    0.0000000000    0.0000040000  
   0.0000000000   17.3600060000   -0.0000010000  
   0.0000030000   -0.0000000000   12.4404560000  
AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz  
      1      0     2.498644     3.809912     3.110113    -0.009924    -0.001443     0.000004  
      2      0     2.498644    12.489910     3.110112    -0.009924    -0.001443     0.000004  
      3      0    10.891125     3.809912     3.110115    -0.009924    -0.001443     0.000004  
      4      0    10.891125    12.489910     3.110114    -0.009924    -0.001443     0.000004  
                                              ...

```

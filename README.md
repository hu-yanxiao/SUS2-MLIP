# SUS2-MLIP
SUS2-MLIP:**S**uper-Linear Machine Learning Interatomic Potentials with Physics-Informed **U**niversal **S**caling and **U**ltra-**S**mall Parameterization. 
This model is realized through the modified [MLIP-2](https://gitlab.com/ashapeev/mlip-2/-/tree/master?ref_type=heads) package.
![image](https://github.com/user-attachments/assets/0aaaa76f-b4f8-459e-b8ec-1ddc08849693)

# Installation
You can install SUS2-MLIP by running:
```bash
 ./configuration  
 make mlp  ## get bin/sus2mlip
 make libinterface ## get lib/libinterface.a for external tool e.g. LAMMPS and pysus2mlip
```
# Format of Datasets
Like the original MLIP-2 package, SUS2-MLIP reads material structures and their properties from `cfg` files.  
  
Format of `cfg` file: 
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
    190      2     8.206723     9.099676     3.110114     0.016692     0.018104    -0.000001
    191      2    16.599205     0.419678     3.110116     0.016692     0.018104    -0.000001
    192      2    16.599205     9.099676     3.110116     0.016692     0.018104    -0.000001
Energy
        -165049.45992
PlusStress:  xx          yy          zz          yz          xz          xy
        1.1012082011734257      -0.7401052299730059     0.2790165512621857      0.0012779654132595323   -0.0002538647260112984          -2.4887947553777763e-10

```
The scripts for converting formats between `cfg` and `ase readable files (e.g. extxyz)` can be found in `./python_tool/`
# Untrained Models
**:red_circle: PLEASE NOTE :red_circle:**: Although the implementation of SUS2-MLIP is based on MLIP-2, there are significant and fundamental differences between the two. Consequently, the original models in `untrained_mtps/` cannot be utilized within the current framework.

Format of `.mtp` files for SUS2-MLIP:
```bash
MTP
version = 1.1.0
potential_name = sus2mlip_l2k2
scaling = 0.01
L = 2
scaling_map = LK
species_count = 2
potential_tag =
radial_basis_type = RBChebyshev_sss
        min_dist = 0.0
        max_dist = 6.0
        ...
```
There are two new hyperparameters `L` and `scaling_map`.   
`L` reffers to the max level of moment tensor.  (**DON'T CHANGE**)  
`scaling_map = L or K or LK` corresponds to 𝜂=(𝐿,𝐾), 𝜂=𝐿 and 𝜂=𝐾 respectively. η determines the dimensions on which the global scaling are applied:  
 $$r_{Ij,{\color{red}\eta}}^{*}=\alpha_{Z_{I}Z_{j},{\color{red}\eta}}\left(r_{Ij}-r_{0}^{Z_{I}Z_{j},{\color{red}\eta}}\right)$$  
  
**Note**: `min_dist` in our model do not affect the mapping from pair distance *r* to *x∈[-1,1]* due to the nonlinearity-embedded universal radial fuction, but it determines the inintialization of scaling factor. Setting `min_dist = 0.0` is usually a good choice.  
  


At `untrained_sus2mlip/`, we prepared 6 sets of untrained basis corresponding to `L∈{2,3} & *k*∈{2,3}`. In both model, the interactions are considered up to the 5-body. Further details regarding the scalar basis in each model are provided in the table below.
![QQ_1735905101839](https://github.com/user-attachments/assets/c2c17d17-81ab-4d2d-ab61-8eb3f0e9d882)  

More technical details about unvirsal scaling and super-linear radial function can be found in our paper: 
> Super-Linear Machine Learning Interatomic Potentials with Physics-Informed Universal Scaling and Ultra-Small Parameterization https://doi.org/xxxx

# Usage
SUS2-MLIP models are trained and evaluated using `mlp-sus2` command, highly similar to `mlp` in MLIP-2.
## model training
```bash
mlp-sus2 train untrained_sus2mlip trainset.cfg -curr-pot-name=current.mtp
```

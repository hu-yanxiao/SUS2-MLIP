# SUS2-MLIP
SUS2-MLIP:**S**uper-Linear Machine Learning Interatomic Potentials with Physics-Informed **U**niversal **S**caling and **U**ltra-**S**mall Parameterization. 
This project adopts MLIP-2-compatible file conventions, such as `.cfg` datasets and `.mtp`-parsable model files.
![image](https://github.com/user-attachments/assets/0aaaa76f-b4f8-459e-b8ec-1ddc08849693)

# Project Family

SUS2 is maintained as a small set of focused repositories:

| Repository | Role |
| --- | --- |
| `SUS2-MLIP` | Core SUS2 v1.1 moment-tensor trainer, model format, and reference CPU workflow. |
| `SUS2-SH` | Spherical-harmonic variant of SUS2. It keeps the SUS2 training logic while replacing moment-tensor angular channels with real SH channels. |
| `SUS2-SH-GPU` | CUDA training evaluator for SUS2-SH, including GPU objective/gradient evaluation, GPU-built linear systems, and multi-GPU streaming training. |
| `GPUMD-SUS2` | GPUMD interface for SUS2 v1.1 moment-tensor models. |
| `GPUMD-SUS2-SH` | GPUMD interface for SUS2-SH models, with SH-specific product and radial paths. |
| `PySUS2` | Python/ASE/phonon workflow package for SUS2 v1.1 moment-tensor models. |
| `PySUS2SH` | Python/ASE/phonon workflow package for SUS2-SH models. |

Use this repository when you need the SUS2-MLIP moment-tensor trainer or model
core. Use `SUS2-SH` and `SUS2-SH-GPU` for the spherical-harmonic line.

# Installation
You can install SUS2-MLIP by running:
```bash
 ./configure
 make mlp  ## get bin/mlp-sus2
 make libinterface ## get lib/libinterface.a for external tool e.g. PySUS2 (in progress)
```
# Format of Datasets
Like the original MLIP package, SUS2-MLIP reads material structures and their properties from `cfg` files.  
  
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
**:red_circle: PLEASE NOTE :red_circle:**: While the implementation of SUS2-MLIP is built upon the MLIP package, there are notable and fundamental differences between them. As a result, the  models in the `untrained_mtps/` folder cannot be used within our current framework.

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
        min_dist = 1.5
        max_dist = 6.0
        ...
```
There are two new hyperparameters `L` and `scaling_map`.   
`L` reffers to the max level of moment tensor.  (**DON'T CHANGE**)  
`scaling_map = L` or `K` or `LK` corresponds to 𝜂=(𝐿,𝐾), 𝜂=𝐿 and 𝜂=𝐾 respectively. η determines the dimensions on which the global scaling are applied:  
 $$r_{Ij,{\color{red}\eta}}^{*}=\alpha_{Z_{I}Z_{j},{\color{red}\eta}}\left(r_{Ij}-r_{0}^{Z_{I}Z_{j},{\color{red}\eta}}\right)$$  
  
**Note**: `min_dist` in our model do not affect the mapping from pair distance *r* to *x∈[-1,1]* due to the nonlinearity-embedded universal radial fuction, but it determines the inintialization of scaling factor. Setting `min_dist = 0.0` is usually a good choice.  
  


At `untrained_sus2mlip/`, we prepared 6 sets of untrained basis corresponding to `L∈{2,3} & k∈{1,3}`. In both model, the interactions are considered up to the 5-body. Further details regarding the scalar basis in each model are provided in the table below.
![QQ_1735905101839](https://github.com/user-attachments/assets/c2c17d17-81ab-4d2d-ab61-8eb3f0e9d882)  

More technical details about universal scaling and super-linear radial function can be found in our paper:
> Super-Linear Machine Learning Interatomic Potentials with Physics-Informed Universal Scaling and Ultra-Small Parameterization [https://doi/10.1073/pnas.2503439122](https://www.pnas.org/doi/10.1073/pnas.2503439122)

# Usage
SUS2-MLIP models are trained and evaluated using `mlp-sus2` command, similar to `mlp` of MLIP. See the [user manual of MLIP](https://gitlab.com/ashapeev/mlip-2/-/blob/master/doc/manual/manual.pdf?ref_type=heads) for details. **The following are some frequently used commands:**  
* **Basic model training**  
To train a model, you should run `mlp-sus2 train` with prepared dataset `trainset.cfg`, untrained model `untrained_sus2mlip` and training options:
```bash
mlp-sus2 train untrained_sus2mlip trainset.cfg --curr-pot-name=current.mtp
```
Commonly used training tags:

```bash
--energy-weight=<double>
--force-weight=<double>
--stress-weight=<double>
--distribution-weight=<double>
--std-weight=<double>
--stdd-weight=<double>
--valid-cfgs=<string>
--max-iter=<int>
--curr-pot-name=<string>
--trained-pot-name=<string>
--bfgs-conv-tol=<double>
--weighting=<string>
--init-params=<random|same>
--do-lin
--skip-preinit
--update-mindist
```

Notes:

- `--do-lin` enables the linear pre-fitting path inside nonlinear training.
- `--std-weight` and `--stdd-weight` control the two std-based regularization terms.
* **Evaluating trained models**  
To evaluate a trained model `trained_sus2mlip` on a specified dataset `target.cfg`, you run:
```bash
mlp-sus2 calc-errors trained_sus2mlip target.cfg
```
# External Tools
## LAMMPS
The current LAMMPS interface for SUS2-MLIP moment-tensor models is provided in
the `sus2-interface-20260410.tar.gz` archive included in this release snapshot.
**Note**: Setting `radial_basis_type = RBChebyshev_sss_lmp` enables the list-based treatment of radial functions in the LAMMPS interface.
For spherical-harmonic models with `potential_tag = SUS2-SH`, use the SH
LAMMPS interface maintained in the `SUS2-SH` repository.
## GPUMD-SUS2
Use [GPUMD-SUS2](https://github.com/hu-yanxiao/GPUMD-SUS2) for GPUMD inference
with SUS2 v1.1 moment-tensor models. Use
[GPUMD-SUS2-SH](https://github.com/hu-yanxiao/GPUMD-SUS2-SH) for SH models.
## PySUS2
PySUS2 is a Python/ASE workflow package for SUS2 v1.1 moment-tensor models. It
supports structure relaxation, phonon dispersion, lattice thermal conductivity,
elastic constants, equation-of-state workflows, and related post-processing.
For SH models, use
[PySUS2SH](https://github.com/hu-yanxiao/PySUS2SH).
## CSO-AES
[CSO-AES](https://github.com/hu-yanxiao/CSO-AES/tree/main) (Covering Set Optimization driven Atomic Environment Sampling) is a Python tool designed to facilitate active learning in SUS2-MLIP modeling. It also helps with the integration and optimization of the database, making it easier for researchers and developers to work with.  
(in progress)

-----------------------------------------------

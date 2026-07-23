# SUS2-SH Binary Source Map

Updated: 2026-07-23

SUS2-SH is a separate project from SUS2-MLIP. Do not use the SUS2-MLIP
developer binary as a substitute for any SUS2-SH binary listed here.

## Formal Release 3

- Repository: `hu-yanxiao/SUS2-SH`
- Branch: `release3`
- Commit: `f1127cd7bd90957af714d0b131313452f2ad8dd6`
- Source archive:
  `/work/phy-weigw/20260321_Test/SUS2-SH-release3-final-candidate-20260723/SUS2-SH-release3-f1127cd.tar.gz`
- Source archive SHA256:
  `510f7d05570afe608467874a44ac22b1b44b54d4253d40c11c61422153be5962`
- Canonical source/runtime root:
  `/work/phy-weigw/20260321_Test/SUS2-SH-release3-work-codex`

Canonical binaries:

| Runtime | Canonical path | SHA256 |
| --- | --- | --- |
| Standalone trainer/evaluator | `/work/phy-weigw/20260321_Test/SUS2-SH-release3-work-codex/bin/mlp-sus2` | `f42a99ed46d60083a7717a854bc27545c3824254c61d157d48e9147cb6d804a8` |
| CPU LAMMPS | `/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi` | `ed1f447d39c473b1c2f5b429e991ce15f640c90ee026768d4cb8922611ec668c` |
| GPU LAMMPS | `/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1` | `e1c6a3e20d541aba7a99b804701cc529b1247d7d7c7076854cde6cbd654fe320` |

The canonical LAMMPS source interface is installed in:

```text
/work/phy-weigw/apps/lammps-10Dec2025
```

The CPU binary is the Intel old-make `ml_sus2_avx2` build with
`gcc/11.2.0` prepended. The GPU binary is the all-double Kokkos CUDA build for
Ampere 80 using CUDA 12.4, NVHPC 22.11, GCC 11.2.0 host code, and CUDA-aware
OpenMPI.

## Release 3 Validation

| Gate | LSF job | Result |
| --- | ---: | --- |
| Standalone release suite | `3972815` | `STANDALONE_TESTS_ALL: ok` |
| CPU formal build | `3972781` | candidate SHA verified |
| CPU K4/K3, MPI1/MPI2 parity | `3972792` | `CPU_PARITY_ALL: ok` |
| GPU formal build | `3972793` | candidate SHA verified |
| GPU K4/K3 device/host parity | `3973010` | `GPU_PARITY_ALL: ok` |
| Expanded GPU compute-sanitizer | `3973020` | four cases, each `ERROR SUMMARY: 0 errors` |
| Four-GPU 1,054,560-atom full-chunk correctness/memory | `3973636` | passed; peak 8,158 MiB/GPU |
| Two-replica 8-run 1,000-step ABBA | `3973637`, `3973638` | final mean 1,591.2425 s; `+0.0253%` |
| Transactional canonical install | `3975307` | installed with rollback backup and exact SHA checks |
| Post-install canonical CPU parity | `3975308` | `POSTINSTALL_CPU_SMOKE_ALL: ok` |
| Post-install canonical GPU parity | `3975309` | `POSTINSTALL_GPU_SMOKE_ALL: ok` |

The four-GPU test used the `_lmp` tabulated radial model with
`chunksize=1054560`. Against the prior stride-guard binary, the final binary
had maximum differences of `8.8314e-16 eV/atom` in energy,
`2.0262e-14` in sampled force components, and `1.4522e-10` in stress.

The ABBA test used full `chunksize=1054560`, 20 warmup steps, and 1,000 timed
steps for every run. The four prior-binary timings averaged `1590.8400 s`; the
four formal Release 3 timings averaged `1591.2425 s`, a `0.0253%` time
increase, below the predeclared `1.0%` acceptance limit.

## Main/Developer Reference

The separate main/developer reference checkout remains:

```text
/work/phy-weigw/20260321_Test/SUS2-SH-work-codex
```

Do not silently treat that checkout as the formal `release3` deployment. Use
the formal paths and hashes above when the request explicitly concerns
Release 3.

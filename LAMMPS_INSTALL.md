# ML-SUS2 LAMMPS Install Standard

This file records the current server-side LAMMPS build convention for `ML-SUS2`.

## Naming

Use `ML-SUS2` as the active package name.

Do not treat `ML-SUS2-MTP` as the active name in new work except when removing old build residue.

## Canonical Remote Paths

- Shared source tree:

```bash
/work/phy-weigw/apps/lammps-10Dec2025
```

- CPU binary:

```bash
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

- GPU binary:

```bash
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

- Interface tarball:

```bash
/work/phy-weigw/20260321_Test/SUS2-MLIP-release-work-codex/sus2-interface-20260410.tar.gz
```

## CPU Rule

CPU LAMMPS uses the old `make` workflow with the Intel compiler toolchain.

Treat that route as the canonical CPU install flow.

## GPU Rule

GPU LAMMPS uses CMake from the same source tree, but only after cleaning up the residue left by the CPU `make` route.

Before GPU configure/build, run:

```bash
make -C /work/phy-weigw/apps/lammps-10Dec2025/src no-all purge
```

Then remove stale root-level SUS2 files from `src/` if they exist:

```bash
pair_sus2_mtp.cpp
pair_sus2_mtp.h
sus2_mtp_*.cpp
sus2_mtp_*.h
```

Use a clean CMake build directory and submit the GPU rebuild on a GPU node through LSF.

## Workflow

1. Update interface sources in the shared source tree.
2. Rebuild CPU with `make` when needed.
3. Purge CPU `make` residue before any GPU CMake rebuild.
4. Reconfigure GPU in a clean build directory.
5. Submit GPU compile on a GPU queue.
6. Replace the canonical installed GPU binary in place.
7. Validate in:

```bash
/work/phy-weigw/hyx/ma/l3k3/jacobi/lmp
```

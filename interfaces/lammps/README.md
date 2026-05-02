# ML-SUS2 LAMMPS Interface

This directory is the maintained LAMMPS interface snapshot for SUS2-MLIP.
It was refreshed from the server runtime tree:

```text
/work/phy-weigw/apps/lammps-10Dec2025
```

The runtime pair styles remain:

```text
sus2mtp
sus2mtp/kk
sus2mtp/kk/device
sus2mtp/kk/host
```

Use `ML-SUS2` as the package name. Do not use the old `ML-SUS2-MTP`
name for new builds.

## Layout

```text
ML-SUS2/                 CPU base interface sources
ML-SUS2-KK/              Kokkos/CUDA override sources
MAKE/OPTIONS/            Intel CPU make targets used on the server
scripts/install_into_lammps.sh
```

The top-level `sus2-interface-20260410.tar.gz` archive contains the same
files for users who prefer the historical single-package distribution.

## Install Into A LAMMPS Tree

Start from a clean LAMMPS source tree, then run:

```bash
cd /path/to/SUS2-MLIP/interfaces/lammps
scripts/install_into_lammps.sh /path/to/lammps
```

The script copies:

- `ML-SUS2/*.cpp` and `ML-SUS2/*.h` into both `LAMMPS_ROOT/src/ML-SUS2/`
  and `LAMMPS_ROOT/src/`.
- `ML-SUS2-KK/pair_sus2_mtp_kokkos.*` into `LAMMPS_ROOT/src/KOKKOS/`.
- `MAKE/OPTIONS/Makefile.ml_sus2_avx2` and
  `MAKE/OPTIONS/Makefile.ml_sus2_avx512_skx` into
  `LAMMPS_ROOT/src/MAKE/OPTIONS/`.
- `ML-SUS2` into the standard package list in
  `LAMMPS_ROOT/cmake/CMakeLists.txt`, if it is not already present.

The direct copy into `LAMMPS_ROOT/src/` is intentional: it makes the old
LAMMPS make workflow reliable even if the downstream tree does not have custom
package install glue for `ML-SUS2`.

## CPU Build

On the maintained server, the CPU binary is selected through this wrapper:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

The wrapper dispatches to AVX2 or AVX512 builds. For a fresh tree after
running the install script:

```bash
cd "$LAMMPS_ROOT/src"
make ml_sus2_avx2 -j 16
make ml_sus2_avx512_skx -j 16
```

The Intel make targets expect Intel MPI and oneAPI MKL variables such as
`MKLROOT` to be available.

## GPU / Kokkos Build

The maintained server GPU binary is:

```text
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

Preferred A100 build modules on the server:

```bash
module load cuda/12.4 nvhpc/22.11 gcc/11.2.0 cmake/3.25.2
export OMPI_CC=gcc
export OMPI_CXX=g++
```

Recommended CMake shape:

```bash
cd "$LAMMPS_ROOT"
mkdir -p build-sus2kk
cd build-sus2kk
cmake \
  -C ../cmake/presets/basic.cmake \
  -D BUILD_MPI=on \
  -D BUILD_OMP=off \
  -D PKG_ML-SUS2=on \
  -D PKG_KOKKOS=on \
  -D Kokkos_ENABLE_SERIAL=on \
  -D Kokkos_ENABLE_CUDA=on \
  -D Kokkos_ENABLE_CUDA_LAMBDA=on \
  -D Kokkos_ARCH_AMPERE80=on \
  -D KOKKOS_PREC=double \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D WITH_JPEG=OFF \
  -D WITH_PNG=OFF \
  ../cmake
cmake --build . -j 16
```

Before reusing a LAMMPS source tree for GPU CMake after an old make build,
clean stale make artifacts first:

```bash
make -C "$LAMMPS_ROOT/src" no-all purge
```

## Runtime Syntax

CPU:

```lammps
pair_style sus2mtp p.mtp tabstep 1.0e-4
pair_coeff * *
```

GPU:

```lammps
pair_style sus2mtp/kk p.mtp chunksize 129600 tabstep 1.0e-4
pair_coeff * *
```

`tabstep` is optional and defaults to `1.0e-4 A`. The table is built only for
species pairs that appear in the active simulation atom types.

The current table path supports:

```text
RBChebyshev_sss_lmp
RBLaguerre_log1p_lmp
RBLaguerre_log1p_pos_lmp
RBJacobi_sss_lmp
```

Other radial basis names are parsed by the CPU side when implemented in the
SUS2 interface sources, but only the `_lmp` variants above use the
preinterpolation table.

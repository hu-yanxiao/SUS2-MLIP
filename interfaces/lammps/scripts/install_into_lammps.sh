#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/lammps" >&2
  exit 2
fi

LAMMPS_ROOT="$(cd "$1" && pwd)"
LAMMPS_SRC="$LAMMPS_ROOT/src"
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$LAMMPS_SRC" ]; then
  echo "LAMMPS source directory not found: $LAMMPS_SRC" >&2
  exit 2
fi

install -d "$LAMMPS_SRC/ML-SUS2"
install -d "$LAMMPS_SRC/KOKKOS"
install -d "$LAMMPS_SRC/MAKE/OPTIONS"

cp "$PKG_ROOT"/ML-SUS2/*.cpp "$PKG_ROOT"/ML-SUS2/*.h "$LAMMPS_SRC/ML-SUS2/"
cp "$PKG_ROOT"/ML-SUS2/*.cpp "$PKG_ROOT"/ML-SUS2/*.h "$LAMMPS_SRC/"
cp "$PKG_ROOT"/ML-SUS2-KK/pair_sus2_mtp_kokkos.cpp \
   "$PKG_ROOT"/ML-SUS2-KK/pair_sus2_mtp_kokkos.h \
   "$LAMMPS_SRC/KOKKOS/"
cp "$PKG_ROOT"/MAKE/OPTIONS/Makefile.ml_sus2_avx2 \
   "$PKG_ROOT"/MAKE/OPTIONS/Makefile.ml_sus2_avx512_skx \
   "$LAMMPS_SRC/MAKE/OPTIONS/"

CMAKE_LIST="$LAMMPS_ROOT/cmake/CMakeLists.txt"
if [ -f "$CMAKE_LIST" ] && ! grep -Eq '^[[:space:]]*ML-SUS2[[:space:]]*$' "$CMAKE_LIST"; then
  tmp_file="$(mktemp)"
  awk '
    { print }
    /^[[:space:]]*ML-UF3[[:space:]]*$/ && ! inserted {
      print "  ML-SUS2"
      inserted = 1
    }
    END {
      if (! inserted) {
        exit 7
      }
    }
  ' "$CMAKE_LIST" > "$tmp_file" || {
    rm -f "$tmp_file"
    echo "Could not insert ML-SUS2 into $CMAKE_LIST; add it to the standard package list manually." >&2
    exit 7
  }
  cp "$CMAKE_LIST" "$CMAKE_LIST.bak_ml_sus2_$(date +%Y%m%d%H%M%S)"
  mv "$tmp_file" "$CMAKE_LIST"
fi

echo "Installed ML-SUS2 LAMMPS interface into $LAMMPS_ROOT"
echo "CPU source files: $LAMMPS_SRC/ML-SUS2 and $LAMMPS_SRC"
echo "Kokkos files: $LAMMPS_SRC/KOKKOS/pair_sus2_mtp_kokkos.*"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MLP_BIN="${MLP_BIN:-./bin/mlp-sus2}"
LAMMPS_BIN="${LAMMPS_BIN:-/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo}"
MPI_RUN="${MPI_RUN:-mpirun}"

tmp_dir=".codex_tmp/lammps_two_layer_gate_additive_mlp_check"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

base_model="$tmp_dir/gated_base.mtp"
direct_model="$tmp_dir/gated_additive.mtp"
lmp_model="$tmp_dir/gated_additive_lmp.mtp"
cfg="$tmp_dir/one.cfg"
mlp_pred="$tmp_dir/mlp_pred.cfg"

"$MLP_BIN" init-sh "$base_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-shared-radial >/dev/null

python3 - "$base_model" "$direct_model" "$lmp_model" <<'PY'
import re
import sys

base, direct, lmp = sys.argv[1:]
text = open(base).read()

species_count = int(re.search(r"species_count\s*=\s*(\d+)", text).group(1))
radial_func_count = int(re.search(r"radial_funcs_count\s*=\s*(\d+)", text).group(1))
alpha_scalar_count = int(re.search(r"alpha_scalar_moments\s*=\s*(\d+)", text).group(1))

def number_list(line_body):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line_body)]

def format_list(key, values):
    return f"{key} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"

def replace_list(model_text, key, values):
    replacement = format_list(key, values)
    pattern = rf"{re.escape(key)}\s*=\s*\{{[^}}]*\}}"
    replaced, count = re.subn(pattern, replacement, model_text, count=1, flags=re.S)
    if count:
        return replaced
    return model_text.rstrip() + "\n" + replacement + "\n"

def replace_required_list(model_text, key, values):
    replacement = f"{key} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    pattern = rf"{re.escape(key)}\s*=\s*\{{[^}}]*\}}"
    replaced, count = re.subn(pattern, replacement, model_text, count=1, flags=re.S)
    if not count:
        raise SystemExit(f"missing {key}")
    return replaced

weights_match = re.search(r"two_layer_gate_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not weights_match:
    raise SystemExit("missing two_layer_gate_weights")
weights = number_list(weights_match.group(1))
for idx in range(len(weights)):
    weights[idx] = 0.0
weights[0] = 0.38
if len(weights) > 1:
    weights[1] = -0.17
if len(weights) > 2:
    weights[2] = 0.09

gate_radial_match = re.search(r"two_layer_gate_radial_coeffs\s*=\s*\{([^}]*)\}", text, re.S)
if not gate_radial_match:
    raise SystemExit("missing two_layer_gate_radial_coeffs")
gate_radial = number_list(gate_radial_match.group(1))
for i in range(len(gate_radial)):
    gate_radial[i] = 0.020 + 0.003 * ((i % 4) - 1.5) + 0.002 * (i // 4)

additive = []
for species in range(species_count):
    for mu in range(radial_func_count):
        if species == 0:
            additive.append(0.35 + 0.04 * mu)
        else:
            additive.append(1.55 - 0.05 * mu)

species_coeffs = [0.12, -0.08][:species_count]
moment_coeffs = [0.0] * alpha_scalar_count
if alpha_scalar_count > 0:
    moment_coeffs[0] = 0.70
if alpha_scalar_count > 1:
    moment_coeffs[1] = -0.21
if alpha_scalar_count > 2:
    moment_coeffs[2] = 0.13
if alpha_scalar_count > 5:
    moment_coeffs[5] = -0.05

text = replace_required_list(text, "two_layer_gate_radial_coeffs", gate_radial)
text = replace_required_list(text, "two_layer_gate_additive_coeffs", additive)
text = replace_required_list(text, "two_layer_gate_weights", weights)
text = replace_list(text, "species_coeffs", species_coeffs)
text = replace_list(text, "moment_coeffs", moment_coeffs)

open(direct, "w").write(text)
open(lmp, "w").write(text.replace(
    "radial_basis_type = RBChebyshev_sss",
    "radial_basis_type = RBChebyshev_sss_lmp",
    1,
))
PY

cat > "$cfg" <<'CFG'
BEGIN_CFG
 Size
    8
 Supercell
    1.2000000000000000e+01 0.0000000000000000e+00 0.0000000000000000e+00
    0.0000000000000000e+00 1.2000000000000000e+01 0.0000000000000000e+00
    0.0000000000000000e+00 0.0000000000000000e+00 1.2000000000000000e+01
 AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz
    1 0  1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00  0 0 0
    2 0  2.3000000000000000e+00 1.2000000000000000e+00 1.1000000000000001e+00  0 0 0
    3 1  4.7999999999999998e+00 1.3999999999999999e+00 1.2000000000000000e+00  0 0 0
    4 1  5.7000000000000002e+00 1.6000000000000001e+00 1.3000000000000000e+00  0 0 0
    5 0  6.4000000000000004e+00 1.8000000000000000e+00 1.1000000000000001e+00  0 0 0
    6 1  7.2999999999999998e+00 2.0000000000000000e+00 1.2000000000000000e+00  0 0 0
    7 0  9.8000000000000007e+00 1.5000000000000000e+00 1.3999999999999999e+00  0 0 0
    8 1  1.0600000000000000e+01 1.3000000000000000e+00 1.0000000000000000e+00  0 0 0
 Energy
    0.0
 PlusStress:  xx          yy          zz          yz          xz          xy
     0 0 0 0 0 0
END_CFG
CFG

cat > "$tmp_dir/data.small" <<'DATA'
LAMMPS data

8 atoms
2 atom types

0.0 12.0 xlo xhi
0.0 12.0 ylo yhi
0.0 12.0 zlo zhi

Masses

1 1.0
2 1.0

Atoms # atomic

1 1 1.0 1.0 1.0
2 1 2.3 1.2 1.1
3 2 4.8 1.4 1.2
4 2 5.7 1.6 1.3
5 1 6.4 1.8 1.1
6 2 7.3 2.0 1.2
7 1 9.8 1.5 1.4
8 2 10.6 1.3 1.0
DATA

make_input() {
  local model="$1"
  local label="$2"
  cat > "$tmp_dir/in.$label" <<LAMMPS
units metal
atom_style atomic
boundary p p p
newton on
read_data $tmp_dir/data.small
pair_style sus2mtp $model tabstep 0.0005
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe
thermo 1
run 0 post no
write_dump all custom $tmp_dir/dump.$label id type x y z fx fy fz modify sort id
LAMMPS
}

run_lammps() {
  local ranks="$1"
  local label="$2"
  make_input "$lmp_model" "$label"
  "$MPI_RUN" -n "$ranks" "$LAMMPS_BIN" \
    -in "$tmp_dir/in.$label" \
    -log "$tmp_dir/log.$label" \
    > "$tmp_dir/stdout.$label" 2> "$tmp_dir/stderr.$label"
}

"$MLP_BIN" calc-efs "$lmp_model" "$cfg" "$mlp_pred" >/dev/null
run_lammps 1 lmp_1
run_lammps 2 lmp_2

python3 - "$tmp_dir" "$mlp_pred" <<'PY'
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
mlp_pred = pathlib.Path(sys.argv[2])

def read_mlp_values(path):
    energy = None
    forces = []
    force_columns = None
    expect_energy = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if expect_energy:
                energy = float(line.split()[0])
                expect_energy = False
            elif line == "Energy":
                expect_energy = True
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                force_columns = [header.index(name) for name in ("fx", "fy", "fz")]
            elif force_columns is not None and line[0].isdigit():
                parts = line.split()
                if len(parts) > max(force_columns):
                    forces.extend(float(parts[idx]) for idx in force_columns)
    if energy is None or not forces:
        raise SystemExit(f"failed to parse {path}")
    return [energy] + forces

def read_lammps_energy(label):
    lines = (root / f"log.{label}").read_text().splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^\s*Step\s+PotEng\s*$", line):
            parts = lines[idx + 1].split()
            if len(parts) >= 2 and parts[0] == "0":
                return float(parts[1])
    raise SystemExit(f"failed to parse PotEng from log.{label}")

def read_lammps_forces(label):
    lines = (root / f"dump.{label}").read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header = line.split()[2:]
            id_col = header.index("id")
            force_cols = [header.index(name) for name in ("fx", "fy", "fz")]
            rows = []
            for raw in lines[idx + 1:]:
                if raw.startswith("ITEM:"):
                    break
                parts = raw.split()
                if parts:
                    rows.append((int(parts[id_col]), [float(parts[col]) for col in force_cols]))
            rows.sort()
            return [value for _, force in rows for value in force]
    raise SystemExit(f"failed to parse forces from dump.{label}")

def lammps_values(label):
    return [read_lammps_energy(label)] + read_lammps_forces(label)

def compare(label, reference, candidate, tol):
    if len(reference) != len(candidate):
        raise SystemExit(f"{label} value count mismatch: {len(reference)} vs {len(candidate)}")
    max_err = max((abs(a - b) for a, b in zip(reference, candidate)), default=0.0)
    if not math.isfinite(max_err) or max_err > tol:
        raise SystemExit(f"{label} max abs error {max_err:.6e} > {tol:.6e}")
    print(f"{label}: max_abs={max_err:.6e}")

mlp = read_mlp_values(mlp_pred)
lmp_1 = lammps_values("lmp_1")
lmp_2 = lammps_values("lmp_2")
compare("mlp_calc_efs vs lammps_1", mlp, lmp_1, 5.0e-4)
compare("mlp_calc_efs vs lammps_2", mlp, lmp_2, 5.0e-4)
compare("lammps_1 vs lammps_2", lmp_1, lmp_2, 1.0e-8)
print("LAMMPS_TWO_LAYER_GATE_ADDITIVE_MLP_CHECK: ok")
PY

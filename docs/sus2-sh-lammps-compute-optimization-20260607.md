# SUS2-SH LAMMPS Compute Optimization Record - 2026-06-07

This file records accepted stage-best LAMMPS compute-side optimization states.
Temporary rejected candidates and raw scripts remain in `.codex_tmp/`.

## Stage-Best Version: Static-Fixed Gate Path + Mu-Grouped SH Edge + SH Eval Precompute

Status: accepted stage-best as of 2026-06-08 02:07 CST.

Git commit:

```text
7e7473600d2358c0d8e91d96b9462670520964ad
Optimize SUS2-SH LAMMPS gate static paths
```

Installed CPU LAMMPS binary:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
sha256 3c345dee54696fc7bcd13bd2499c311d9850c80b5b6c2a84ab44d142e82cbf7a
```

Previous installed binary backup:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.bak_20260608_before_sh_eval_precompute
sha256 77c3d914a4a03acdd98c703836e4ab46116546fb8d125464c59070d2839af13c
```

Accepted changes in this stage:

- `static_fixed_types` no-gate fixed-center/fixed-neighbor SH basic cache.
- Gate first-layer fixed-center/fixed-neighbor SH basic cache.
- Gate first-layer product-limit pruning from `two_layer_gate_scalar_indices`.
- Gate main-layer fixed-fixed separable cache for additive tanh gate models.
- Gate fixed-fixed dynamic edge-list pruning.
- Generic SH basic edge loop grouped by contiguous `mu=(l,k)` blocks, with fallback to the original loop if the model layout is not `mu` grouped.
- SH evaluation precomputes inverse powers and removes redundant zero-initialize
  stores for the supported `lmax <= 4` real-SH path.

Mathematical validation:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_mu_group_run0_1r_20260607
```

Same-rank old/new comparisons against `lmp_ml_sus2_avx2_noipo.gate_edge_prune_20260607`:

- gate force/base/static and pressure/base/static force dumps: `maxdf = 0`, `rmsdf = 0`
- gate pressure thermo values: max diff `0`
- no-gate force/base/static and pressure/base/static force dumps: `maxdf = 0`, `rmsdf = 0`
- no-gate pressure thermo values: max diff `0`

Formal speed validation:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_mu_group_ab_40c_20260607
```

40-rank same-node A/B on `b03u26a`, Intel Xeon Platinum 8375C, `_lmp` table mode:

| Case | Previous Pair avg | Stage-best Pair avg | Repeat Pair avg | Incremental speedup |
| --- | ---: | ---: | ---: | ---: |
| gate static-fixed | 5.1822 s | 5.0965 s | 5.0929 s | about 1.7% |
| no-gate static-fixed | 1.8599 s | 1.8539 s | 1.8534 s | about 0.3% |

Current compute-only bottleneck after this stage, excluding communication:

- first-layer edge radial/SH/basic: about 23.5%
- main-layer edge radial/SH/basic: about 23.9%
- first-layer product/backprop: about 23.6%
- main-layer product/backprop: about 23.3%
- force/ZBL/gate-chain/cache leftovers: small

Rejected or not-yet-accepted directions already checked:

- SH-basic force factorization: exact, slower in LAMMPS.
- consumer-gather backprop: exact in microbenchmark, slower.
- output-grouped product DAG: exact, slower.
- full generated forward+backprop product kernel: exact, only small Pair gain because generated backprop slowed integrated LAMMPS.
- forward-only hard-coded generated product kernel: exact and useful for no-gate on this one `l4/k4/body6` graph, but not accepted as production because it is hard-coded and not generic over all `lk` models.
- SH edge value/derivative cache between gate layers: exact, much slower due to extra memory traffic.
- fixed endpoint force-write skip: exact in run0, slower.
- conservative fixed-only gate center skip: exact in run0, slower.
- gate dynamic edge buffer AoS (`TwoLayerGateEdge`) conversion: exact in run0,
  slower in same-node 40-rank A/B; keep the current split-vector layout.

This stage-best version is the rollback baseline for later experiments.

## Direct Improvement Relative To Initial Optimization Baseline

Before the direct A/B finished, the stage-wise Pair-time estimate was:

- no-gate: about `1.16x`, or about `+16%` Pair compute speedup.
- gate: about `1.28x`, or about `+28%` Pair compute speedup.

The estimate is from multiplying validated stage increments:

- no-gate fixed-fixed basic cache: `1.4122 -> 1.2194 s`, `1.158x`.
- no-gate mu-grouped edge increment: `1.8599 -> 1.8534 s`, `1.004x`.
- gate first/static+prune stage: `4.6411 -> 4.1198 s`, `1.127x`.
- gate main fixed-fixed separable cache: `9.4841 -> 8.6455 s`, `1.097x`.
- gate fixed-fixed edge-list pruning: `5.2712 -> 5.1654 s`, `1.020x`.
- gate mu-grouped edge increment: `5.1822 -> 5.0929 s`, `1.018x`.

Because those measurements came from different stage A/B runs, the direct A/B
below is the authoritative stage-best-vs-initial evidence.

Run directory:

```text
jobid 3769467
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_stagebest_vs_initial_ab_40c_20260608
```

Node and CPU:

```text
b03u26a
Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
40 MPI ranks
```

Binaries:

```text
initial   c060e9103ae741b3334f44fb629b92f4e393766f406173982f8f3fe58fe599cf
          /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.codexbak_20260607_before_static_fixed_final
stagebest 77c3d914a4a03acdd98c703836e4ab46116546fb8d125464c59070d2839af13c
          /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

The initial binary was run without `static_fixed_types`; the stage-best binary
was run with `static_fixed_types 2,3,7,8`. This compares the same mathematical
model while enabling the accepted fixed/static caches.

| Case | Initial Pair avg | Stage-best Pair avg | Repeat Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate, before SH eval precompute | 6.0571 s | 5.0889 s | 5.0913 s | 1.190x | 1.187x |
| no-gate, before SH eval precompute | 2.1471 s | 1.8530 s | 1.8545 s | 1.158x | 1.198x |
| gate, current installed | 6.0571 s | 4.9891 s | 4.9891 s | 1.214x | 1.210x |
| no-gate, current installed | 2.1471 s | 1.8135 s | n/a | 1.184x | 1.226x |

Direct conclusion:

- gate Pair compute is now about `+21.4%` faster than the initial optimization baseline.
- no-gate Pair compute is now about `+18.4%` faster than the initial optimization baseline.
- no-gate loop time is now about `+22.6%` faster because communication/modify balance also improved in this run.

## Rejected Experiment: Gate Dynamic Edge AoS Buffer

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_edge_struct_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_edge_struct_ab_40c_20260608
jobids:      3769477, 3769479
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.edge_struct_20260608
sha256 c3724db8e1a57c9acb516113e70e79b674c28420c0908f500074f3a3a961f512
```

The candidate replaced the separate gate dynamic edge arrays with one
`TwoLayerGateEdge` vector. It preserved the math exactly in run0:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a` showed no benefit:

| Case | Stage-best Pair avg | Candidate Pair avg | Speedup |
| --- | ---: | ---: | ---: |
| gate | 5.08595 s | 5.09425 s | 0.998x |
| no-gate | 1.85350 s | 1.85770 s | 0.998x |

Decision: rejected. The extra AoS cache-line footprint outweighed any reduction
in vector bookkeeping. The local and remote source trees were restored to the
stage-best split-vector layout after the test.

## Build Note: Pair Factory Object Must Match the Pair Header

During candidate relinking on 2026-06-08, binaries that replaced only
`pair_sus2_mtp.o` crashed at pair-style initialization with:

```text
malloc(): invalid size (unsorted)
```

Root cause: LAMMPS compiles pair-style factory construction through
`style_pair.h` inside `force.o`. If `PairSUS2MTP` layout changes in
`pair_sus2_mtp.h`, a stale `force.o` can contain a stale
`sizeof(PairSUS2MTP)`, causing constructor memory corruption even when
`pair_sus2_mtp.o` itself is new.

Rule for future candidate builds: after any header change, and whenever
relinking a temporary candidate, rebuild and replace both objects in the static
archive:

```text
Obj_ml_sus2_avx2_noipo/force.o
Obj_ml_sus2_avx2_noipo/pair_sus2_mtp.o
```

This was verified by rebuilding both objects and relinking
`lmp_ml_sus2_avx2_noipo.stagebest_relink_force_20260608`, which initialized
`gate.mtp` correctly. Earlier `pair_sus2_mtp.o`-only relinks crashed.

## Rejected Experiment: Scalar Additive Gate Coefficients

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_scalar_additive_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_scalar_additive_ab_40c_20260608
jobids:      3769498, 3769499
```

Candidate idea: detect the special case where
`a_{Z,\mu}` is constant over all radial functions for a neighbor species `Z`.
Then the current stabilized gate multiplier

```text
s_{j,\mu} = v_Z * (1 + alpha * tanh(a_{Z,\mu} * f_j))
```

can be evaluated as a single scalar per atom/species instead of one value per
`mu`. The derivative factor can also use the same scalar. This would be exact
only when the loaded model really has species-scalar additive coefficients.

Correctness on run0 was exact for the tested model:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

The current `gate.mtp` does not satisfy the scalar-coefficient condition:

```text
160 coefficients = 8 species * 20 radial functions
all species: all_equal = false
example species 1: min 0.1293, max 1.6915, maxdiff 1.2239
```

Therefore the fast path did not trigger and only added a branch. Same-node
40-rank A/B on `b03u26a` showed a slight slowdown:

| Case | Stage-best Pair avg | Candidate Pair avg | Speedup |
| --- | ---: | ---: | ---: |
| gate | 5.08540 s | 5.10275 s | 0.998x |
| no-gate | 1.85360 s | 1.85710 s | 0.998x |

Decision: rejected. It may be useful only for a deliberately constrained model
where `a_{Z,\mu}` is species-scalar. It is not useful for the current general
`a_{Z,\mu}` gate model.

## Rejected Experiment: Gate Static First-Layer Early Skip

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_static_skip_early_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_static_skip_early_ab_40c_20260608
jobids:      3769501, 3769502
```

Candidate idea: in the first-layer gate edge loop, when
`use_static_fixed_gate_cache` is active and both center and neighbor are fixed
types, skip before computing `rsq`, `sqrt`, and radial data. This preserves the
existing static fixed-fixed first-layer cache math and only changes loop order.

Correctness on run0 was exact for the tested model:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a` showed no measurable benefit:

| Case | Stage-best Pair avg | Candidate Pair avg | Speedup |
| --- | ---: | ---: | ---: |
| gate | 5.07980 s | 5.08390 s | 0.999x |
| no-gate | 1.85480 s | 1.86020 s | 0.997x |

Decision: rejected. The fixed-fixed early-skip work was already mostly removed
by the accepted static caches; moving the branch earlier did not materially
change the hot path.

## Fine Profile: Current Stage-Best Gate Hotspots

Run directory:

```text
jobid 3769503
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_fine_profile2_40c_20260608
```

Node and CPU:

```text
b03u26a
Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
40 MPI ranks, OMP_NUM_THREADS=1
```

Profile binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_fine_profile_20260608
sha256 4c406408abe1ec3493e3819a8114ae5baf20c7cad3aa42b2af34ad80703a1d13
```

This was a temporary instrumented build on top of the stage-best source. The
installed production binary was not replaced.

LAMMPS reported:

```text
Loop time of 2.15494 on 40 procs for 200 steps with 2223 atoms
Pair max time 2.136 s
```

The cumulative fine-profile line at 200 calls was:

```text
total=2.13748997
init=0.00463830307
first_edge=0.50445367
gate_forward=0.207353935
gate_deriv=0.290371869
forward_comm=0.425284304
main_edge=0.539707121
main_products=0.486487035
main_force=0.0619456246
reverse_comm=0.543721773
gate_chain=0.00300102681
first_edges=26457262
zbl_pairs=170414
main_edge_evals=26457262
main_static_skips=0
```

Percentages below divide each per-category MPI-rank maximum by the Pair total
maximum. They should be treated as hotspot ranking rather than an exactly
additive decomposition because each category is reduced independently with
`MPI_MAX`.

| Block | Time | Approx share |
| --- | ---: | ---: |
| main edge radial/basic accumulation | 0.5397 s | 25.3% |
| first-layer edge radial/basic accumulation | 0.5045 s | 23.6% |
| main products/backprop/weighted derivatives | 0.4865 s | 22.8% |
| reverse comm for gate adjoints | 0.5437 s | 25.4% |
| forward comm for gate values | 0.4253 s | 19.9% |
| first-layer gate derivative backprop/dot | 0.2904 s | 13.6% |
| first-layer gate forward scalar products | 0.2074 s | 9.7% |
| main force dot/scatter | 0.0619 s | 2.9% |
| final gate-chain force scatter | 0.0030 s | 0.1% |

Optimization implication:

- The remaining compute-side target is edge radial/basic accumulation plus
  product/backprop work.
- Final force scatter and the last gate-chain force loop are too small to be
  worth optimizing further.
- Communication is visible in the two-layer gate path, but the current
  optimization focus remains compute-only.

## Rejected Experiment: Non-Contiguous Mu-Indexed Basic Traversal

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_mu_indexed_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_mu_indexed_ab_40c_20260608
jobids:      3769506, 3769507
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.mu_indexed_20260608
sha256 7413bd8462737c2062407062c927030307c39e67e42a3fb61006d89f0b6e3039
```

Candidate idea: the current model's `alpha_index_basic` order is not
contiguously sorted by `mu`, so the existing `sh_basic_mu_grouped` fast path is
not active. The candidate built a `mu -> original basic index list` and used it
in SH basic accumulation and the static fixed gate main cache. Each original
basic index `k` was still written in its original slot, so the math and all
product definitions were unchanged.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a` showed a clear slowdown:

| Case | Stage-best Pair avg | Candidate Pair avg | Speedup |
| --- | ---: | ---: | ---: |
| gate | 5.08355 s | 5.47645 s | 0.928x |
| no-gate | 1.85420 s | 1.92510 s | 0.963x |

Decision: rejected. The indirect `mu -> k` traversal reduced some repeated
radial metadata reads, but it changed `moment_tensor_vals` and jacobian writes
from contiguous `k` order to a strided/original-index order. The cache and
vectorization loss outweighed the savings.

## Accepted Experiment: SH Eval Inverse-Power Precompute

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_eval_precompute_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_eval_precompute_ab_40c_20260608
jobids:      3769510, 3769511
```

Candidate and installed binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.sh_eval_precompute_20260608
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
sha256 3c345dee54696fc7bcd13bd2499c311d9850c80b5b6c2a84ab44d142e82cbf7a
```

Change:

- `eval_real_sh()` now computes `1/r`, `r^-2`, `r^-3`, and `r^-4` once per edge
  and passes the appropriate inverse power and derivative factor to each
  `Y_lm` component.
- The old implementation zero-initialized all requested SH values and
  derivatives before immediately overwriting every supported `lmax <= 4`
  component. The accepted version removes that redundant zero-fill.
- Formulas for each real-SH component and derivative are unchanged.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 5.07635 s | 4.98910 s | 1.017x | 1.017x |
| no-gate | 1.85270 s | 1.81350 s | 1.022x | 1.027x |

Decision: accepted and installed. This is a generic optimization for the
currently supported LAMMPS real-SH path (`lmax <= 4`) and benefits both gate and
single-layer SH modes.

## Rejected Experiment: First-Layer Sparse Needed Gate Moments

Run directories:

```text
first correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sparse_needed_run0_8c_20260608
first speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sparse_needed_ab_40c_20260608
first jobids:      3769513, 3769514
split correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sparse_needed_split_run0_8c_20260608
split speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sparse_needed_split_ab_40c_20260608
split jobids:      3769517, 3769518
```

Candidate binaries:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_sparse_needed_20260608
sha256 733fba8a9249046014b80ba7cf719c112538dab5a27f673a311b69f56b72c8b4

/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_sparse_needed_split_20260608
sha256 8590b4eaf74e8c5c9d6c4f73fcfedf3a653091c67c12fc4bdcd3bcf4ca3fc7ae
```

Candidate idea: precompute the two-layer gate's required moment dependency
graph, reduce first-layer gate moment clearing, and only enable product pruning
when enough products inside the required prefix can be skipped. For the current
model the gate weight mapping is dense in the prefix:

```text
alpha_moments = 3313
alpha_basic = 100
sh_product_count = 11489
gate_weight_count = 1503
gate_product_limit = 10097
needed_products_in_prefix = 10069
```

So only 28 products in the prefix could be skipped. The first candidate kept a
runtime pruning guard in the hot product/backprop loops and slowed down
substantially:

| Case | Stage-best Pair avg | Candidate Pair avg | Speedup |
| --- | ---: | ---: | ---: |
| gate | 5.03505 s | 5.28780 s | 0.952x |
| no-gate | 1.81580 s | 1.81690 s | 0.999x |

The split-loop candidate removed that hot-loop guard when product pruning was
inactive. Correctness on run0 was exact for both gate and no-gate:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a` for the split-loop candidate:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98995 s | 4.99185 s | 1.000x | 1.001x |
| no-gate | 1.81600 s | 1.81420 s | 1.001x | 1.008x |

Decision: rejected. The dependency graph is mathematically correct, but the
current model's gate dependency prefix is too dense for meaningful product
pruning, and reducing only the first-layer clear is not enough to move Pair
time. Source and the default LAMMPS build were restored to the accepted
`sh_eval_precompute` stage-best state after the test.

## Rejected Experiment: Gate Mu-Cache Precompute Before Main Edge

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_mu_cache_precompute_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_mu_cache_precompute_ab_40c_20260608
jobids:      3769519, 3769520
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_mu_cache_precompute_20260608
sha256 189628409f067911cb70c80f0c6331150a26a3b7c9c948fe547f2ee8872e94ca
```

Candidate idea: after the gate-value `forward_comm`, precompute the per-atom
per-`mu` tanh multiplier and derivative cache for all local plus ghost atoms,
then use a cache-only table interpolation path in the main edge loop. This
keeps the same mathematical inputs as the existing lazy cache path but avoids
filling tanh cache inside the edge loop.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.99040 s | 4.99445 s | 0.999x | 1.000x |
| no-gate | 1.81530 s | 1.81510 s | 1.000x | 1.001x |

Decision: rejected. The lazy per-atom tanh cache was already cheap enough; the
extra precompute pass and cache-only dispatch did not reduce Pair time. Source
and the default LAMMPS build were restored to the accepted `sh_eval_precompute`
stage-best state after the test.

## Rejected Experiment: Gate Edge SH Geometry Cache

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sh_cache_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_sh_cache_ab_40c_20260608
jobids:      3769521, 3769522
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_sh_cache_20260608
sha256 7b7485f1021ead99f074995e941939e3c9d315a582ad58acb62d1dbc2a786e09
```

Candidate idea: in the two-layer gate path, cache the first-layer active
edge's real-SH values and derivatives and reuse them in the main layer. This is
mathematically valid because both layers see the same edge geometry, while the
main layer still computes its own radial/table values and gate multipliers.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.99360 s | 5.30630 s | 0.941x | 0.942x |
| no-gate | 1.81560 s | 1.81490 s | 1.000x | 1.001x |

Decision: rejected. Avoiding the second `eval_real_sh()` did not compensate
for writing and rereading `(1 + 3) * (lmax + 1)^2` doubles per active edge. The
extra memory traffic made gate Pair time significantly worse. Source and the
default LAMMPS build were restored to the accepted `sh_eval_precompute`
stage-best state after the test.

## Rejected Experiment: Radial Table Delta Precompute

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_radial_delta_table_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_radial_delta_table_ab_40c_20260608
jobids:      3769523, 3769524
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.radial_delta_table_20260608
sha256 49faabdfee8107a567454176f64a96ae3ab52f65f47441ea6f20ec4f7e887122
```

Candidate idea: for `_lmp` radial tables, precompute per-grid deltas
`table[n+1] - table[n]` for main radial and two-layer gate radial tables, so
hot-path interpolation uses `row + ddr * delta` instead of computing the
subtraction per edge. This is mathematically identical for the table path.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98605 s | 5.21920 s | 0.955x | 0.955x |
| no-gate | 1.81650 s | 1.88450 s | 0.964x | 0.964x |

Decision: rejected. Moving the subtraction out of the hot loop was not worth
the extra table memory and cache pressure. Both gate and no-gate slowed down.
Source and the default LAMMPS build were restored to the accepted
`sh_eval_precompute` stage-best state after the test.

## Rejected Experiment: Packed Alpha-Times Index Rows

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_alpha_times_packed_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_alpha_times_packed_ab_40c_20260608
jobids:      3769525, 3769526
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.alpha_times_packed_20260608
sha256 8103a1a075faebc617c4579e0b833c0ac2b8759afd37bc4e1bdbbd4dc3a76092
```

Candidate idea: keep the original alpha-times topology/order and scalar
coefficient array, but pack the three integer indices `(a0, a1, out)` into one
contiguous row array. Product and backprop loops then read one packed row plus
one coefficient per product instead of three separate integer arrays. This is
generic over all `lk` models and does not change product order.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.99415 s | 5.02625 s | 0.994x | 0.992x |
| no-gate | 1.81590 s | 1.81780 s | 0.999x | 0.991x |

Decision: rejected. Packing the three integer streams did not improve the
integrated LAMMPS product/backprop loops. The extra row copy/load pattern was
slightly slower for gate and did not help no-gate. Source and the default
LAMMPS build were restored to the accepted `sh_eval_precompute` stage-best
state after the test.

## Rejected Experiment: Split Basic-Edge Accumulation Branches

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_branch_split_basic_edge_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_branch_split_basic_edge_ab_40c_20260608
jobids:      3769527, 3769529
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.branch_split_basic_edge_20260608
sha256 b9680b2a56c0925fd20de55c968b50c6473ac245b1fd2827a6ad9605501bab6c
```

Candidate idea: specialize the hot `accumulate_sh_basic_edge()` loops by call
mode, moving the `jj >= 0` and `store_raw` branches outside the per-basic
inner loop. This leaves the radial, SH, gate, force, and virial formulas
unchanged and is generic for all `lk` models.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98970 s | 5.10005 s | 0.978x | 0.978x |
| no-gate | 1.81710 s | 1.81610 s | 1.001x | 0.999x |

Decision: rejected. Removing the inner branches did not improve the integrated
gate path; the larger/specialized loop body was slower, likely from weaker
instruction-cache or vectorization behavior. Source and the default LAMMPS
build were restored to the accepted `sh_eval_precompute` stage-best state
after the test.

## Accepted Experiment: Reuse Inverse Distance in SH Evaluation

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_inv_dist_reuse_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_inv_dist_reuse_ab_40c_20260608
speed_check: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_inv_dist_reuse_ab2_40c_20260608
jobids:      3769530, 3769531, 3769532
```

Candidate/accepted binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.inv_dist_reuse_20260608
sha256 dd0699d088ce9643f9bebf10d355c84d5706c2a0177f768835409e11fcd3f06e
```

Candidate idea: pass the caller's already computed `1.0 / dist` into
`eval_real_sh()` so SH evaluation and the edge-Jacobian path reuse the same
inverse distance instead of doing separate divisions for the same edge. This
does not change radial interpolation, SH values, SH derivatives, gate formulas,
force accumulation, or virial formulas.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stage-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.99665 s | 4.97920 s | 1.004x | 1.004x |
| no-gate | 1.81720 s | 1.81020 s | 1.004x | 1.005x |

Same-node 40-rank A/B on `b03u26a`, candidate-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98860 s | 4.98615 s | 1.000x | 1.001x |
| no-gate | 1.81655 s | 1.81070 s | 1.003x | 1.003x |

Decision: accepted as a small, low-risk incremental improvement. The measured
gain is below 1% but is positive in both orderings and preserves bitwise
force/pressure/energy equality on the run0 check.

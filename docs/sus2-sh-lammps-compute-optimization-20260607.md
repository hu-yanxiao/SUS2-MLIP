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

## GPU Kokkos Gate Stage-Best - 2026-06-08

Status: accepted stage-best for the SUS2-SH Kokkos `/kk/device` interface.

Installed candidate before promotion:

```text
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_gatekk_candidate_20260608
sha256 bea9719ef8f520fb42ba4eb58c12c1c49aee3e7d85be6fa5eb2eeaf6d640d2a3
```

Build job:

```text
jobid 3770695
host a05u22g
queue gpu-phy-zhangwq
source hashes:
  pair_sus2_mtp_kokkos.cpp 474d794850284ac7f4251d58e8ff85ecd03d21050b1549da006d3e7c1923294d
  pair_sus2_mtp_kokkos.h   5a47a8b3ebc11fe1c0ba5f3a58be25da48906a0f14de4759602ac37c801cb700
```

Verification job:

```text
jobid 3770733
host a05u22g
queue gpu-phy-zhangwq
directory /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608
pair_style sus2mtp/kk/device gate.mtp chunksize 129600 tabstep 0.0005
```

CPU/GPU correctness against the CPU SUS2-SH LAMMPS implementation:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 5.076636e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.592310e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 5.076636e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.834484e-10 |

Performance on the same verification job:

| Case | Loop time | Throughput |
| --- | ---: | ---: |
| gate run100 | 43.1765 s | 329.513 katom-step/s |
| no-gate run100 | 19.2852 s | 737.726 katom-step/s |
| gate run500 | 215.463 s | 330.154 katom-step/s |
| no-gate run500 | 96.1945 s | 739.502 katom-step/s |

The best observed run for the same stage was job `3770529` on `b05u08g`:

```text
gate run500   336.277 katom-step/s
no-gate run500 766.659 katom-step/s
gate/no-gate 0.438626
```

Use the `3770529` numbers as the best observed throughput and the `3770733`
numbers as the final promotion verification on the latest rebuilt binary.

### Rejected Kokkos Experiment: Mu-Group ThreadVectorRange

The attempted CUDA `ThreadVectorRange(team, basic_mu_group_count)` reduction
over radial `mu=(l,k)` groups was exact but much slower. It increased the
gate first-derivative profile from about `0.114 s` to `0.250 s`, and reduced
the 500-step benchmark on `b05u08g` to:

```text
jobid 3770671
gate run500   202.066 katom-step/s
no-gate run500 514.904 katom-step/s
gate/no-gate 0.392434
```

Decision: reject and roll back. For this Kokkos path, keep serial `mu` loops
inside the team kernels and use device parallelism over atoms/edges instead of
vectorizing the small radial-group loop. The accepted code keeps the value-only
real-SH helper for basic-moment kernels, but does not use the rejected
`ThreadVectorRange` reduction over `basic_mu_group_count`.

### Rejected Kokkos Experiment: Team-Size Environment Sweep

Run directory:

```text
jobid 3770795
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_team_sweep_20260608
```

Candidate binary:

```text
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_gatekk_candidate_20260608
sha256 066c8e02b423164695849a6d5f2f95292d27842d580c2cb6a3f9df6ae9b9dbab
```

Candidate source hashes:

```text
pair_sus2_mtp_kokkos.cpp d83fb02fc36e9d6a0bc447868927299d337f5c009e081e944267c22d003b84a0
pair_sus2_mtp_kokkos.h   5a47a8b3ebc11fe1c0ba5f3a58be25da48906a0f14de4759602ac37c801cb700
```

Candidate idea: expose Kokkos team sizes through environment variables while
keeping the default launch sizes unchanged:

```text
SUS2_SH_KK_ALPHA_TEAM_SIZE
SUS2_SH_KK_FORCE_TEAM_SIZE
SUS2_SH_KK_GATE_DERIV_TEAM_SIZE
SUS2_SH_KK_GATE_CHAIN_TEAM_SIZE
```

The sweep used the same `2x2x2` replicated benchmark on one A100 with
`sus2mtp/kk/device ... chunksize 129600 tabstep 0.0005`.

| Variant | Gate run100 | No-gate run100 | Gate throughput | No-gate throughput | Gate/no-gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| default a16/f32/d32 | 43.1825 s | 19.1681 s | 329.467 katom-step/s | 742.233 katom-step/s | 0.443886 |
| force16 a16/f16/d32 | 46.9384 s | 23.7713 s | 303.104 katom-step/s | 598.503 katom-step/s | 0.506436 |
| force64 a16/f64/d32 | 43.2457 s | 19.5242 s | 328.985 katom-step/s | 728.696 katom-step/s | 0.451471 |
| deriv16 a16/f32/d16 | 46.4136 s | 19.2569 s | 306.531 katom-step/s | 738.811 katom-step/s | 0.414898 |
| deriv64 a16/f32/d64 | 42.8848 s | 19.1837 s | 331.754 katom-step/s | 741.630 katom-step/s | 0.447331 |
| alpha8 a8/f32/d32 | 45.2171 s | 19.7811 s | 314.642 katom-step/s | 719.232 katom-step/s | 0.437469 |
| alpha32 a32/f32/d32 | 42.3851 s | 19.7798 s | 335.665 katom-step/s | 719.279 katom-step/s | 0.466669 |
| all16 a16/f16/d16 | 50.1768 s | 24.2008 s | 283.541 katom-step/s | 587.881 katom-step/s | 0.482311 |

The best gate-only setting, `alpha32`, improved the 100-step gate throughput by
only about 1.7-1.9% over the default and simultaneously slowed no-gate by about
3.9%. The no-gate target of `800 katom-step/s` and gate target of
`400 katom-step/s` were still not reached.

Decision: rejected. Launch-size tuning does not address the remaining Kokkos
hot path. Keep the accepted stage-best code and binary:

```text
source pair_sus2_mtp_kokkos.cpp 474d794850284ac7f4251d58e8ff85ecd03d21050b1549da006d3e7c1923294d
binary /work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
sha256 bea9719ef8f520fb42ba4eb58c12c1c49aee3e7d85be6fa5eb2eeaf6d640d2a3
```

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

## Current Stage-Best vs Initial After Inverse-Distance Reuse

Direct A/B directory:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_current_inv_dist_vs_initial_ab_40c_20260608
jobid: 3769533
```

Compared binaries:

```text
initial: c060e9103ae741b3334f44fb629b92f4e393766f406173982f8f3fe58fe599cf
         /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.codexbak_20260607_before_static_fixed_final
current: dd0699d088ce9643f9bebf10d355c84d5706c2a0177f768835409e11fcd3f06e
         /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

Same-node 40-rank direct A/B, two repeats each:

| Case | Initial Pair avg | Current Pair avg | Pair speedup | Initial Loop avg | Current Loop avg | Loop speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate | 6.05255 s | 4.99210 s | 1.212x | 6.19152 s | 5.12243 s | 1.209x |
| no-gate | 2.14775 s | 1.81030 s | 1.186x | 3.01924 s | 2.43774 s | 1.239x |

Status: the no-gate single-layer SH path is near the 20% pair-time target and
above it by loop time; the gate path remains around 21% faster by pair/loop
time, still below the requested 40% target.

## Accepted Experiment: Remove Redundant Gate First-Local Edge Vector

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_no_first_local_vector_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_no_first_local_vector_ab_40c_20260608
speed_check: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_no_first_local_vector_ab2_40c_20260608
jobids:      3769535, 3769536, 3769537
```

Candidate/accepted binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.no_first_local_vector_20260608
sha256 46a2096aebdcb8d05030c3f7164227c402986334f942c558042d569888e9344a
```

Candidate idea: remove `two_layer_gate_edge_first_local_indices`. In the
current two-layer gate construction, an edge is pushed to the dynamic edge list
only after static first-layer fixed edges have already been skipped, and every
pushed edge immediately calls `accumulate_sh_basic_edge()` with the next local
edge index. Therefore, within each center,
`first_local == active_idx - active_begin` exactly. Storing a separate integer
edge vector is redundant.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stage-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98935 s | 4.97570 s | 1.003x | 1.002x |
| no-gate | 1.80955 s | 1.81445 s | 0.997x | 0.999x |

Same-node 40-rank A/B on `b03u26a`, candidate-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.99425 s | 4.97475 s | 1.004x | 1.004x |
| no-gate | 1.81330 s | 1.81360 s | 1.000x | 1.002x |

Decision: accepted. The gain is small but directionally stable for the gate
path in both ordering checks, and the no-gate pair-time difference is at the
noise level. The change removes one gate-only edge vector and one edge-vector
read without changing radial, SH, gate, force, or virial formulas.

## Current Stage-Best vs Initial After Removing First-Local Vector

Direct A/B directory:

```text
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_current_no_first_local_vs_initial_ab_40c_20260608
jobid: 3769538
```

Compared binaries:

```text
initial: c060e9103ae741b3334f44fb629b92f4e393766f406173982f8f3fe58fe599cf
         /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.codexbak_20260607_before_static_fixed_final
current: 46a2096aebdcb8d05030c3f7164227c402986334f942c558042d569888e9344a
         /work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
```

Same-node 40-rank direct A/B, two repeats each:

| Case | Initial Pair avg | Current Pair avg | Pair speedup | Initial Loop avg | Current Loop avg | Loop speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gate | 6.04470 s | 4.97235 s | 1.216x | 6.17710 s | 5.10171 s | 1.211x |
| no-gate | 2.14580 s | 1.81255 s | 1.184x | 3.01106 s | 2.43475 s | 1.237x |

Status: current gate is about `+21.6%` by Pair time and `+21.1%` by Loop
time relative to the initial optimization baseline. Current no-gate is about
`+18.4%` by Pair time and `+23.7%` by Loop time.

## Rejected Experiment: Internal SH Basic-Moment Reorder

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_basic_reorder_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_basic_reorder_ab_40c_20260608
jobids:      3769539, 3769540
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.sh_basic_reorder_20260608
sha256 8e125d0786b62b9a549238b18fbfae3718803418ecc69333966cac5edb948bec
```

Candidate idea: internally permute the first `alpha_index_basic_count` SH
basic moments by `(mu, sh_index)` after reading the model, then remap every
`sh_products` reference and `alpha_moment_mapping` entry that points into the
basic-moment range. This activates the existing `sh_basic_mu_grouped` edge
accumulation path without changing scalar order, linear coefficients, gate
scalar indices, gate weights, radial coefficients, or the model file format.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, candidate-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98155 s | 5.33015 s | 0.935x | 0.936x |
| no-gate | 1.81165 s | 1.81745 s | 0.997x | 0.997x |

Decision: rejected. The permutation is mathematically safe, but the grouped
edge accumulation path is slower for this model. The likely reason is that the
extra `mu` loop structure and changed low-index moment access pattern do not
pay back the removed `alpha_basic_mu[k]` lookup; the gate path pays this cost
in both first-layer and main-layer edge accumulation, so it regresses most.
The local and server source were restored to the previous stage-best after
this test.

## Rejected Experiment: Push Gate Derivatives Only After Backprop

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_deriv_push_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_deriv_push_ab_40c_20260608
jobids:      3769541, 3769542
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_deriv_push_20260608
sha256 a1ccf43d304fc82eaac8c77a4e03b8753c32af5853f8fa53aebc82e261198862
```

Candidate idea: do not append zero placeholders to
`two_layer_gate_edge_deriv_{x,y,z}` while building the first-layer gate edge
list. Instead, append the final `(gx, gy, gz)` values in the same edge order
after gate backprop computes them. This removes one redundant zero write per
component per active gate edge and leaves formulas and edge ordering unchanged.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, candidate-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98955 s | 5.01705 s | 0.995x | 0.996x |
| no-gate | 1.81465 s | 1.81535 s | 1.000x | 1.001x |

Decision: rejected. The removed zero writes did not translate into a speedup.
The extra `push_back` work moved into the gate derivative loop, and the gate
path was slightly slower by both Pair and Loop time. The no-gate difference
was noise-level, as expected.

## Rejected Experiment: Defer First-Layer Gate Jacobian Storage

Run directories:

```text
isolation correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_recompute_gate_deriv_localreduction_run0_8c_20260608
correctness:           /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_defer_gate_jac_localreduction_run0_8c_20260608
speed:                 /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_defer_gate_jac_vs_stagebest_ab_40c_20260608
jobids:                3769553, 3769554, 3769555
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.defer_gate_jac_localreduction_20260608
sha256 c141ca177b956f5baa942c42de8b8a5329af68f74d98bb0300d4280fbd3be125
```

Candidate idea: in the two-layer gate first pass, accumulate only first-layer
SH basic moment values and avoid storing the per-active-edge
`3 * alpha_index_basic_count` Jacobian slab. After gate-product backprop,
recompute the same first-layer radial values and real-SH derivatives for each
active edge, then dot them with `nbh_energy_ders_wrt_moments` to fill
`two_layer_gate_edge_deriv_{x,y,z}`. This is mathematically equivalent when
the recompute path uses the same radial table index/bin/fraction and the same
`two_layer_gate_shared_radial` setting.

Debug note: the first recompute implementation failed gate force consistency
with nonzero `x/z` force errors because it used an OpenMP SIMD reduction on
reference parameters. Changing the helper to accumulate into local
`local_gx/local_gy/local_gz` variables and then assign the references fixed the
issue. The isolated recompute-only run was exact after that change.

Correctness for the full candidate was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.97640 s | 5.27755 s | 0.943x | 0.943x |
| no-gate | 1.81400 s | 1.81495 s | 0.999x | 1.000x |

Decision: rejected. Removing the first-layer Jacobian slab reduces memory
traffic and memory footprint, but the later recomputation repeats radial table
interpolation and SH derivative evaluation for every active gate edge. For the
current model, that extra compute is more expensive than the saved memory
traffic, so gate Pair time slows by about 5.7%. The no-gate path is unaffected
within noise. Local and server source, plus the active default LAMMPS build,
were restored to the accepted first-local-vector stage-best after the test.

## Rejected Experiment: Basic-Input Pair-Reduced Backprop

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_basic_pair_backprop_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_basic_pair_backprop_vs_stagebest_ab_40c_20260608
jobids:      3769557, 3769558
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.basic_pair_backprop_20260608
sha256 d99b5f0fb1ef7548f74c6613930a36f7a24353e185855e546295c6e4e77134c5
```

Candidate idea: keep the forward product loop in its original order, but in
the full reverse product pass group repeated product rows with the same
ordered `(a0, a1)` when both inputs are basic moments. For each safe group,
compute `sum_r dE/dM[out_r] * coeff_r` once and update the two basic moment
derivatives once. The two-layer gate prefix backprop was deliberately left on
the original loop because otherwise the single cached plan would alternate
between prefix and full limits for every center.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.97790 s | 5.42525 s | 0.918x | 0.919x |
| no-gate | 1.81375 s | 2.07350 s | 0.875x | 0.874x |

Decision: rejected. The grouping is mathematically safe for basic-input rows,
but the generic execution plan adds an extra entry stream, group-row indirection,
and branch structure to a very tight reverse loop. Those costs are larger than
the saved basic-derivative updates, especially for no-gate where full product
backprop is directly on the hot path. Local and server source, plus the active
default LAMMPS build, were restored to the accepted first-local-vector
stage-best after the test.

## Rejected Experiment: Moment-Derivative Seed Copy

Run directories:

```text
full correctness:        /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_deriv_seed_run0_8c_20260608
full speed:              /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_deriv_seed_vs_stagebest_ab_40c_20260608
no-gate-only correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_deriv_seed_nogate_run0_8c_20260608
no-gate-only speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_deriv_seed_nogate_vs_stagebest_ab_40c_20260608
jobids:                  3769559, 3769560, 3769561, 3769562
```

Candidate binaries:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.deriv_seed_20260608
sha256 321d8dd84ffe3633ee88831f6ca36fc660bc9c5f245f66c79a607583e77ad09d

/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.deriv_seed_nogate_20260608
sha256 1bb2885521dd6ae1f4030cc98b715eb8c45b61317da46202f8e1b96cb5bfc802
```

Candidate idea: prebuild dense derivative seed arrays and replace
`fill(0) + scalar-index scatter` with one contiguous copy. The full candidate
used one seed for ordinary linear derivatives and one seed for two-layer gate
weights. The second candidate kept only the ordinary linear seed in the
single-layer/no-gate path and restored the original gate derivative
initialization.

Correctness on run0 was exact for both candidates:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stagebest-first order:

| Candidate | Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| full seed | gate | 4.97385 s | 5.00340 s | 0.994x | 0.994x |
| full seed | no-gate | 1.81355 s | 1.79010 s | 1.013x | 1.009x |
| no-gate-only seed | gate | 4.97410 s | 5.05400 s | 0.984x | 0.985x |
| no-gate-only seed | no-gate | 1.81525 s | 1.79220 s | 1.013x | 1.013x |

Decision: rejected. The seed copy is useful for no-gate, but both versions
caused a measurable gate regression in the same A/B setup. Since the active
objective requires improving gate and no-gate without trading one against the
other, neither version was installed. Local and server source, plus the active
default LAMMPS build, were restored to the accepted first-local-vector
stage-best after the test.

## Rejected Experiment: Gate Dynamic Edge Type Removal

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_edge_no_type_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_edge_no_type_vs_stagebest_ab_40c_20260608
jobids:      3769603, 3769604
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.edge_no_type_20260608
sha256 6d7a209739957261b63a3e7b270aac115dbc97f12b932ec16288dda70ac99875
```

Candidate idea: remove the cached `two_layer_gate_edge_types` vector from the
two-layer gate dynamic edge list and recover `jtype` as `type[j] - 1` from the
cached neighbor id when needed. This is mathematically equivalent because edge
order, neighbor ids, radial table indices, and all coordinate caches are
unchanged.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u26a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.97715 s | 5.03975 s | 0.988x | 0.988x |
| no-gate | 1.81305 s | 1.81585 s | 0.998x | 0.996x |

Decision: rejected. The removed integer vector saves one push/read stream, but
the replacement `type[j]` lookup is less favorable on the gate hot path for
this benchmark. The candidate regressed gate Pair time by about 1.2% and gave
no useful no-gate gain. Local and server source, plus the active default LAMMPS
build, were restored to the accepted first-local-vector stage-best after the
test.

## Rejected Experiment: Native Mu-Run SH Basic Fast Path

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_run_group_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_sh_run_group_vs_stagebest_ab_40c_20260608
jobids:      3769642, 3769644
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.sh_run_group_20260608
sha256 868cafb35f0996f6e0668c454f07fce0d2b3b439aed93d11eb3fbe8889b95169
```

Candidate idea: add a native-order contiguous-`mu` run fast path for SUS2-SH
basic moment accumulation. The existing `sh_basic_mu_grouped` path only works
when all basic moments are globally sorted by increasing `mu`. The current
model has repeated contiguous `mu` runs in its native order, so the candidate
hoisted `radial_vals[mu]` and `radial_ders[mu]` once per run while preserving
the original basic-moment indices, moment writes, Jacobian writes, and product
graph.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u22a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.96935 s | 5.41400 s | 0.918x | 0.918x |
| no-gate | 1.81305 s | 1.85290 s | 0.978x | 0.975x |

Decision: rejected. The native-run path is mathematically equivalent, but the
extra run loop, additional branch, and weaker compiler vectorization are more
expensive than the saved `radial_vals/radial_ders` loads. This is especially
bad for gate, where it regressed Pair time by about 8.2%. Local and server
source, plus the active default LAMMPS build, were restored to the accepted
first-local-vector stage-best after the test.

## Rejected Experiment: Recompute Gate Edge Table Metadata

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_recompute_table_info_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_recompute_table_info_vs_stagebest_ab_40c_20260608
jobids:      3769663, 3769664
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.recompute_table_info_20260608
sha256 64f901fa29233675b3cef99b7ba6c9ed8c84f944dee239c6c9c2ed24a0ac7f8f
```

Candidate idea: remove the cached two-layer gate edge table metadata
`table_index`, `table_bin`, and `table_frac`. The first-layer pass still uses
the table metadata immediately for gate radial values, but it no longer stores
three edge streams. The main additive pass recomputes table metadata from the
cached `dist`, current center type, and cached neighbor type before evaluating
the same additive table formula.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u34a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.98190 s | 5.01475 s | 0.993x | 0.993x |
| no-gate | 1.81220 s | 1.81190 s | 1.000x | 1.000x |

Decision: rejected. Recomputing table bin/frac is mathematically equivalent
but the extra arithmetic/function path is slightly more expensive than the
saved edge metadata streams on the gate benchmark. No-gate is effectively
unchanged because this candidate only touches the two-layer gate path. Local
and server source, plus the active default LAMMPS build, were restored to the
accepted first-local-vector stage-best after the test.

## Rejected Experiment: Fuse Gate Additive Table Interpolation With SH Basic Accumulation

Run directories:

```text
correctness: /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_table_sh_fusion_run0_8c_20260608
speed:       /work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gate_table_sh_fusion_vs_stagebest_ab_40c_20260608
jobids:      3769673, 3769674
```

Candidate binary:

```text
/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2_noipo.gate_table_sh_fusion_20260608
sha256 15c0f59f94869cd3da27e029ce85193fd6468f8a62924a8e57e70f2c913a708f
```

Candidate idea: for the main additive gate pass under `_lmp` radial tables,
fuse table interpolation, tanh-gate multiplier evaluation, real-SH evaluation,
moment accumulation, Jacobian write, and raw basic value write into one helper.
This avoids materializing and then rereading `radial_vals`, `radial_ders`, and
`two_layer_gate_residual_radial_vals` for the gate main-additive dynamic edge.
The fallback path kept the original `calc_pair_radial_values()` plus
`accumulate_sh_basic_edge()` behavior for unsupported basis/table cases.

Correctness on run0 was exact:

- no-gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- no-gate PE/Press/Pxx/Pyy/Pzz: diff `0`
- gate force and pressure dumps: `max_abs = 0`, `rms = 0`
- gate PE/Press/Pxx/Pyy/Pzz: diff `0`

Same-node 40-rank A/B on `b03u32a`, stagebest-first order:

| Case | Stage-best Pair avg | Candidate Pair avg | Pair speedup | Loop speedup |
| --- | ---: | ---: | ---: | ---: |
| gate | 4.96800 s | 5.53640 s | 0.897x | 0.897x |
| no-gate | 1.81280 s | 1.81350 s | 1.000x | 0.998x |

Decision: rejected. The fused path is mathematically identical in the run0
check, but it regresses gate Pair time by about 10.3%. The fused helper adds
more per-edge control flow and loses the simpler old hot-loop structure, so the
saved scratch writes do not pay for the extra work. No-gate is effectively
unchanged. Local and server source, plus the active default LAMMPS build, were
restored to the accepted first-local-vector stage-best after the test.

## Final Selection After Last Candidate Attempt

The last candidate considered was a gate main-additive fast-dispatch path:
prevalidate that additive coefficients and table ratios are complete, then let
the common `_lmp` additive-table edge skip repeated per-edge storage-size and
fallback checks while preserving the same table interpolation, tanh multiplier,
mu cache, SH accumulation, force, and virial formulas.

This candidate was not installed. During the final build attempt on
`/work/phy-weigw/apps/lammps-10Dec2025/src`, the Intel build reached the static
archive step and then stayed in uninterruptible I/O inside `ar` for several
minutes with an incomplete `liblammps_ml_sus2_avx2_noipo.a`. The build process
was killed, and the candidate never produced a verified binary, run0 result, or
40-rank A/B result. Because the user requested this to be the final attempt,
no further candidate was started.

The selected final best version is the accepted first-local-vector stage-best:

```text
installed binary:
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
sha256 46a2096aebdcb8d05030c3f7164227c402986334f942c558042d569888e9344a

LAMMPS source restored to:
/work/phy-weigw/apps/lammps-10Dec2025/src/pair_sus2_mtp.cpp
sha256 6d6fef9ffcad2f00f9a1c31129e3daf11bca3e82176d74d6fc5487b01a621e8e

/work/phy-weigw/apps/lammps-10Dec2025/src/pair_sus2_mtp.h
sha256 a6a22d610f31ff7c5614689b5ff617648a8d2215208b531a99008bd243816078
```

Final direct A/B against the initial optimization baseline remains:

| Case | Pair speedup | Loop speedup |
| --- | ---: | ---: |
| gate | 1.216x, about +21.6% | 1.211x, about +21.1% |
| no-gate single-layer SH | 1.184x, about +18.4% | 1.237x, about +23.7% |

Conclusion: current best is mathematically verified and installed. No later
candidate produced a verified improvement over it, so this version is the final
selected LAMMPS CPU build for the current optimization round.

## GPU Kokkos Stage-Best Update - 2026-06-09

Scope: SUS2-SH Kokkos `/kk/device` gate path in
`interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.{cpp,h}` and the installed
GPU LAMMPS binary:

```text
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

Rejected full scratch-adjoint candidate:

- Candidate binary:
  `/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_scratchadj_candidate_20260609`
- Correctness: gate max force diff `1.0e-8`; no-gate max force diff `1.0e-7`.
- Same-node `c04u01g` performance: gate `338.688 katom-step/s`, but no-gate
  dropped to `711.994 katom-step/s`.
- Decision: rejected because scratch caching in the ordinary main-force path
  damaged single-layer SH performance.

Accepted narrow candidate: cache only the per-center
`d_nbh_energy_ders_wrt_moments(ii,k)` adjoints in team scratch inside
`GateFirstDerivsTeam`. The main-force and no-gate force kernels remain on the
old path. This is mathematically identical because the cached values are copied
from the same per-center adjoint view before the edge loop and only reused in
the first-layer derivative contraction.

Build and verification:

```text
build job: 3770832 on c04u01g
verify job: 3770839 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/firstderiv_scratch_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_firstderiv_scratch_candidate_20260609
candidate sha256:
3dc2c35872bb84f3a4d816b5f12de189c0686fccdc0f4820f8437840cf356239
```

CPU/GPU correctness against the CPU SUS2-SH LAMMPS implementation:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.702725e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.343427e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Previous installed | Accepted candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 333.230 katom-step/s | 337.451 katom-step/s | +1.27% |
| no-gate | 751.174 katom-step/s | 751.071 katom-step/s | -0.01% |
| gate/no-gate | 0.443613 | 0.449293 | +1.28% |

Accepted candidate profile at evflag0:

```text
gate first_derivs = 0.110735228 s
gate total        = 0.488591147 s
no-gate total     = 0.269471024 s
```

Decision: accepted as the current GPU Kokkos stage-best because it gives a
small but clean gate speedup while preserving no-gate single-layer SH
performance and CPU/GPU correctness. The rejected full scratch-adjoint variant
must not be revived without a separate no-gate regression fix.

### Rejected `inv_dist` Reuse Candidate - 2026-06-09

Candidate idea: hoist `inv_dist = 1.0 / dist` once per active edge and reuse it
in SH derivative contractions instead of emitting several equivalent `1/dist`
expressions. This was mathematically identical to the accepted source: every
replacement stayed after the same `rsq > 0` and cutoff checks, and the ZBL force
path was not changed.

Build and verification:

```text
build job: 3770842 on c04u01g
verify job: 3770844 on c04u01g, manually stopped after candidate gate run500
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/invdist_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_invdist_candidate_20260609
candidate sha256:
8afc45d7cc79643010b8674d2ba50023e170a7297ae7bef41b445235e9ca4cfb
```

CPU/GPU correctness before stopping the job:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.827798e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.093205e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100:

| Case | Accepted installed | `inv_dist` candidate | Change |
| --- | ---: | ---: | ---: |
| gate run500 | 337.601 katom-step/s | 334.715 katom-step/s | -0.85% |
| no-gate run500 | 751.424 katom-step/s | not run | n/a |

Candidate profile at evflag0 before stopping:

```text
gate total    = 0.502485860 s
no-gate total = 0.268672617 s
```

Decision: rejected. The compiler/runtime did not turn the manual reciprocal
hoist into a speedup; gate run100 and run500 were both slower than the accepted
installed binary. The source on the local developer worktree and the remote
LAMMPS source tree were restored to the accepted stage-best hashes:

```text
pair_sus2_mtp_kokkos.cpp
d028e9be227d31d52deb56fdac0acf57b0840ee9a659ca56804d690c50e92aeb

pair_sus2_mtp_kokkos.h
0539ce97045843e05a7f085c963b72e20fa52edfe341ce6cfa3e7bcabb84f7f9

installed canonical binary
3dc2c35872bb84f3a4d816b5f12de189c0686fccdc0f4820f8437840cf356239
```

### Accepted Gate Main-Force Scratch Candidate - 2026-06-09

Candidate idea: cache only the per-center SH adjoint coefficients
`d_nbh_energy_ders_wrt_moments(ii,k)` in team scratch inside
`GateMainForceTeam`, then use the scratch values in the SH contraction for the
main two-layer gate force. This is mathematically identical: the cached values
are copied from the same per-center adjoint view after the product backprop and
before the neighbor loop, and no product order, radial value, SH value, force
formula, ZBL term, gate multiplier, or gate adjoint update is changed. The
ordinary no-gate `ComputeForceTeam` path is not modified.

Build and verification:

```text
build job: 3770850 on c04u01g
full verify job: 3770854 on c04u01g
repeat gate-only A/B job: 3770856 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/gateforce_scratch_verify_20260609
repeat directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/gateforce_scratch_repeat_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_gateforce_scratch_candidate_20260609
candidate sha256:
345770229e108abffe09f2910bde95337f63427b0aad8ffc65fbdab81b25ad10
```

CPU/GPU correctness against the CPU SUS2-SH LAMMPS implementation:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.687424e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.343501e-10 |

Same-node full A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Previous installed | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 337.688 katom-step/s | 338.803 katom-step/s | +0.33% |
| no-gate | 751.392 katom-step/s | 751.431 katom-step/s | +0.01% |
| gate/no-gate | 0.449417 | 0.450877 | +0.32% |

Gate-only repeat A/B on the same node:

| Case | Previous installed | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate repeat | 337.358 katom-step/s | 338.221 katom-step/s | +0.26% |

Candidate profile at evflag0:

```text
gate first_derivs = 0.110747877 s
gate main_force   = 0.159864817 s
gate total        = 0.488617221 s
no-gate main_force = 0.168865198 s
no-gate total      = 0.268226364 s
```

Decision: accepted as a small but repeatable gate speedup. The performance gain
is only about 0.3%, so this does not change the broader optimization picture:
the current GPU Kokkos stage-best is still below the 400 katom-step/s gate and
800 katom-step/s no-gate targets for the 2x2x2 benchmark. The installed
canonical binary and source hashes after acceptance are:

```text
pair_sus2_mtp_kokkos.cpp
fd020e545f6a1fad341be72637f28ef61f7878c2ed24b6a834ce9aa9ee7eb92b

pair_sus2_mtp_kokkos.h
0539ce97045843e05a7f085c963b72e20fa52edfe341ce6cfa3e7bcabb84f7f9

installed canonical binary
345770229e108abffe09f2910bde95337f63427b0aad8ffc65fbdab81b25ad10
```

### Accepted Gate SH-Only Alpha Scratch Candidate - 2026-06-09

Candidate idea: the `/kk/device` two-layer gate path already rejects non-SH
models at settings time, so `GateFirstLayer` and `GateMainAlphaBasic` do not
need the generic tensor-power scratch or the non-SH polynomial branch. This
candidate keeps exactly the old `is_sh_model` formula path:

```text
B_{i,k} += R_{Z_i Z_j,\mu(k)}(r_ij) Y_k(rhat_ij)
B^{main}_{i,k} += R^{main}_{Z_i Z_j,\mu(k)}(r_ij)
                 [1 + A tanh(a_{Z_j,\mu(k)} f_j)] Y_k(rhat_ij)
```

The only implementation changes are reducing the per-team scratch allocation
for those two gate alpha kernels to `team_size * basic_mu_group_count`, and
removing the unreachable non-SH branch inside those kernels. No product graph,
force formula, gate chain rule, ZBL term, no-gate alpha kernel, or no-gate force
kernel is changed.

Build and verification:

```text
build job: 3770858 on c04u01g
full verify job: 3770859 on c04u01g
repeat gate-only A/B job: 3770860 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/alpha_scratch_verify_20260609
repeat directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/alpha_scratch_repeat_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_alpha_scratch_candidate_20260609
candidate sha256:
4567450e46b63fe4cea1041f2209e85e68505c2eb5e1836f77938ee233110161
```

CPU/GPU correctness against the CPU SUS2-SH LAMMPS implementation:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.963439e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.694191e-10 |

Same-node full A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Previous installed | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 338.848 katom-step/s | 340.605 katom-step/s | +0.52% |
| no-gate | 751.489 katom-step/s | 751.563 katom-step/s | +0.01% |
| gate/no-gate | 0.450902 | 0.453196 | +0.51% |

Gate-only repeat A/B on the same node:

| Case | Previous installed | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate repeat | 338.772 katom-step/s | 339.808 katom-step/s | +0.31% |

Profile at evflag0:

```text
stable gate first_layer = 0.0521211935 s
candidate gate first_layer = 0.0504507413 s
stable gate main_basic = 0.0513359069 s
candidate gate main_basic = 0.0510981713 s
stable gate total = 0.488569380 s
candidate gate total = 0.486472460 s
candidate no-gate total = 0.268341438 s
```

Decision: accepted. The speedup is modest but repeatable, and the profile shows
the expected first-layer scratch reduction. The current GPU Kokkos stage-best
is still below the 400 katom-step/s gate and 800 katom-step/s no-gate targets,
so the next higher-headroom candidates remain SH basic-moment scratch
reduction and SH/table-only fast dispatch.

Installed canonical binary and source hashes after acceptance:

```text
pair_sus2_mtp_kokkos.cpp
569c167e428a9623b3f169d8feec78e79078d0557f2562801d7bf1be61d4a74e

pair_sus2_mtp_kokkos.h
0539ce97045843e05a7f085c963b72e20fa52edfe341ce6cfa3e7bcabb84f7f9

installed canonical binary
4567450e46b63fe4cea1041f2209e85e68505c2eb5e1836f77938ee233110161
```

### Rejected candidate: no-gate SH-only alpha dispatch

Date: 2026-06-09.

Candidate change: split the no-gate SH branch of `ComputeAlphaBasic` into a
dedicated `ComputeAlphaBasicSH` team kernel and reduce its scratch allocation to
`team_size * basic_mu_group_count`. The mathematical expression was unchanged:

```text
B_{i,k} += R_{Z_i Z_j,\mu(k)}(r_ij) Y_k(rhat_ij)
```

The candidate excluded env-gate and non-SH models from the fast path, so those
fallback paths stayed on the original generic kernel.

Build and verification:

```text
build job: 3770862 on c04u01g
failed first verify job: 3770863 on a05u22g
successful verify job: 3770864 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/nogate_alpha_sh_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_nogate_alpha_sh_candidate_20260609
candidate sha256:
ff5927d891cb6843a17c8cda117439b2b323b3ded07bac1c2196b9c0bd75cb19
```

Correctness against the CPU SUS2-SH LAMMPS implementation remained at the same
level as the installed binary:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.959568e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.829111e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Installed stage-best | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 340.372 katom-step/s | 340.512 katom-step/s | +0.04% |
| no-gate | 751.060 katom-step/s | 707.377 katom-step/s | -5.82% |
| gate/no-gate | 0.453188 | 0.481372 | ratio changes only because no-gate slowed |

Candidate profile at evflag0:

```text
profile_gate_evflag0 total = 0.486520613 s
profile_nogate_evflag0 main_basic = 0.0530149713 s
profile_nogate_evflag0 total = 0.281182376 s
```

Decision: rejected. The dedicated no-gate SH kernel preserved math but degraded
no-gate performance materially. Likely causes are worse compiler scheduling or
register/local-memory behavior after introducing another Kokkos tag; avoiding
the unused tensor scratch did not translate into speed. The current stage-best
remains the installed canonical binary with sha256
`4567450e46b63fe4cea1041f2209e85e68505c2eb5e1836f77938ee233110161`.

### Build-route candidate: GCC host with `nvcc_wrapper`

Date: 2026-06-09.

Candidate change: no source-code change. Rebuilt the current stage-best source
from an independent CMake directory using Kokkos `nvcc_wrapper` with GCC 11.2 as
the host compiler:

```text
CMAKE_CXX_COMPILER=/work/phy-weigw/apps/lammps-10Dec2025/lib/kokkos/bin/nvcc_wrapper
NVCC_WRAPPER_DEFAULT_COMPILER=g++
C++ Compiler Type: GNU
C++ Compiler Version: 11.2.0
```

This was tested because the installed NVHPC-host binary exited with signal 4 on
`a05u22g`, while the project build standard prefers GCC host code with NVCC for
GPU Kokkos builds.

Build and verification:

```text
build job: 3770867 on c04u01g
verify job: 3770868 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/gcc_host_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_gcc_host_candidate_20260609
candidate sha256:
3090f86d6bcae4e7a57414b2ac7d9873bd5208e9f79f4637bc9ae52dafff0c05
```

Correctness against the CPU SUS2-SH LAMMPS implementation:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 5.076636e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.594976e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 5.076636e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.137441e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Installed stage-best | GCC-host candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 340.271 katom-step/s | 340.152 katom-step/s | -0.03% |
| no-gate | 751.312 katom-step/s | 751.120 katom-step/s | -0.03% |
| gate/no-gate | 0.452902 | 0.452860 | -0.01% |

Candidate profile at evflag0:

```text
profile_gate_evflag0 total = 0.485523748 s
profile_nogate_evflag0 main_force = 0.169506524 s
profile_nogate_evflag0 total = 0.268052655 s
```

Decision: not accepted as a speed optimization. It is mathematically correct
and essentially performance-neutral, but it does not move toward the
400/800 katom-step/s targets and the binary is larger. A short `a05u22g`
compatibility smoke was submitted as job 3770870 but canceled while pending
because the specified host could not satisfy the requested affinity resources.
The current installed stage-best remains the NVHPC-host binary with sha256
`4567450e46b63fe4cea1041f2209e85e68505c2eb5e1836f77938ee233110161`.

Operational note: the first verify job landed on `a05u22g` and exited with
signal 4 before candidate execution; the installed stage-best binary also hit
the illegal instruction in the stable reference phase. The same script completed
on `c04u01g`. Until the GPU build is moved fully to a host-GCC path, speed
verification should avoid CPU-incompatible GPU nodes or explicitly pin to a
known-good host.

### Rejected candidate: no-gate force coeff scratch cache

Date: 2026-06-09.

Candidate change: mirror the gate main force team's coefficient scratch cache in
the no-gate `ComputeForceTeam` kernel. The intended math-preserving optimization
was to load

```text
d_nbh_energy_ders_wrt_moments(ii, k)
```

once per center atom into a per-team `s_basic_coeffs(k)` buffer, then reuse that
inside the neighbor and `mu_group` loops for the SH branch. No force formula,
neighbor traversal, radial value, ZBL, or virial path was changed.

Build and verification:

```text
build job: 3770865 on c04u01g
verify job: 3770866 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/force_coeff_scratch_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_force_coeff_scratch_candidate_20260609
candidate sha256:
b37a59f575df8237eb0cd13e4b8250d7a2907ac0009a51813f3962798a276ede
```

Correctness against the CPU SUS2-SH LAMMPS implementation remained unchanged:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.566638e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.823855e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Installed stage-best | Candidate | Change |
| --- | ---: | ---: | ---: |
| gate | 340.305 katom-step/s | 340.321 katom-step/s | +0.00% |
| no-gate | 750.980 katom-step/s | 713.254 katom-step/s | -5.02% |
| gate/no-gate | 0.453148 | 0.477139 | ratio changes only because no-gate slowed |

Candidate profile at evflag0:

```text
profile_gate_evflag0 total = 0.485955164 s
profile_nogate_evflag0 main_force = 0.169295617 s
profile_nogate_evflag0 total = 0.268110839 s
```

Decision: rejected. The scratch cache preserved math but did not reduce the
dominant no-gate force time, and run500 performance regressed materially. The
extra per-team scratch likely reduced occupancy or worsened backend scheduling
enough to overwhelm any saved global coefficient loads. The current stage-best
remains the installed canonical binary with sha256
`4567450e46b63fe4cea1041f2209e85e68505c2eb5e1836f77938ee233110161`.

### Accepted phase-best: packed main radial value/derivative table

Date: 2026-06-09.

Candidate change: keep the existing main radial value table and derivative table,
but add a flat interleaved force-path table:

```text
d_radial_vd_list[((table_index * list_grid_size + grid) * basic_mu_group_count + mu_group) * 2 + 0] = R
d_radial_vd_list[((table_index * list_grid_size + grid) * basic_mu_group_count + mu_group) * 2 + 1] = dR/dr
```

The table is generated in the same coefficient loop as the original
`d_radial_list` and `d_radial_der_list`. Value-only alpha/basic paths keep using
`d_radial_list`; only force kernels that need both `R` and `dR/dr` use the
interleaved table. The interpolation formula is unchanged:

```text
R(r) = R0 + ddr * (R1 - R0)
dR(r)/dr = D0 + ddr * (D1 - D0)
```

This is mathematically identical to the previous two-table read path. ZBL,
two-layer gate first-layer radial, tanh gate multiplier, force accumulation,
virial/stress, and communication logic are unchanged.

Build and verification:

```text
build job: 3770874 on c04u01g
verify job: 3770875 on c04u01g
repeat job: 3770876 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/packed_radial_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_packed_radial_candidate_20260609
candidate sha256:
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

Correctness against CPU SUS2-SH LAMMPS:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 5.076636e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.590896e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.829085e-10 |

Same-node A/B on `c04u01g`, `replicate 2 2 2`, one A100, `run 500`:

| Case | Previous stage-best | Packed main radial | Change |
| --- | ---: | ---: | ---: |
| gate | 340.39 katom-step/s | 358.45 katom-step/s | +5.30% |
| no-gate | 751.42 katom-step/s | 838.45 katom-step/s | +11.57% |

Candidate evflag0 profile:

```text
profile_gate_evflag0 total = 0.443920429 s
  first_layer = 0.0504187100 s
  first_finalize = 0.0460338779 s
  first_derivs = 0.1103781550 s
  main_basic = 0.0510071991 s
  products = 0.0169847533 s
  nbh_der = 0.0328772605 s
  main_force = 0.1177797930 s
profile_nogate_evflag0 total = 0.228269660 s
  main_basic = 0.0402022581 s
  products = 0.0170660143 s
  nbh_der = 0.0329649563 s
  main_force = 0.1294965370 s
```

Memory impact for the benchmark table shape
`64 species pairs x 12001 grid points x 20 mu groups`: the existing main
radial value+derivative tables use about `234.39 MB`; this candidate adds one
more interleaved `R,dR` table of about the same size. This is acceptable for the
A100 benchmark, but it should be kept visible when moving to much larger
species/radial grids.

Decision: accepted as the current phase-best. It passes the no-gate target
(`838.45 >= 800.00 katom-step/s`) and improves gate throughput, but gate remains
below the requested `400.00 katom-step/s` target.

### Rejected candidate: packed two-layer gate radial value/derivative table

Date: 2026-06-09.

Candidate change: add a second interleaved flat table for the first-layer gate
radial `R^{gate}` and `dR^{gate}/dr`, then use it in the first-layer derivative
force kernels. The math is the same as the accepted main radial packing: both
original tables and the packed table were generated from the same
`two_layer_gate_radial_coeffs` loop and the same linear interpolation.

Outcome: rejected before runtime verification. Build job `3770877` on `c04u01g`
was killed after the NVHPC CUDA host backend remained in `cpp2` for several
minutes while compiling `pair_sus2_mtp_kokkos.cpp.o`, with no candidate binary
produced. This resembles the earlier struct-view attempt: mathematically clean
data-layout changes can still push the single large Kokkos translation unit into
unacceptable NVHPC backend compile time. The source was reverted to the accepted
main radial packed version.

Operational lesson: do not add more Kokkos device views/helpers to this monolithic
translation unit unless the expected runtime gain is large enough to justify the
compiler risk. For future gate speed work, prefer reducing kernel work or data
traffic within already-compiled structures, or split the Kokkos translation unit
before adding more table variants.

### Rejected candidate: gate basic scratch accumulation

Date: 2026-06-09.

Candidate change: in `ComputeGateFirstLayer` and `ComputeGateMainAlphaBasic`,
replace per-neighbor `atomic_add` into `d_moment_tensor_vals(ii,k)` with
per-team scratch accumulation `s_basic_sums(team_rank,k)` and a final team
reduction over `team_size`. No formula was changed:

```text
B_i,k^gate += sum_j R_gate_ij,mu Y_k(rhat_ij)
B_i,k^main += sum_j R_main_ij,mu [1 + A tanh(a_Zj,mu f_j)] Y_k(rhat_ij)
```

Build and verification:

```text
build job: 3770878 on c04u01g
verify job: 3770879 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/basic_scratch_reduce_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_basic_scratch_reduce_candidate_20260609
candidate sha256:
cdd82183ad5dcef0f2eeaef71154b865e7fa09be31c6c37cde9be00817163ff0
```

Correctness against CPU SUS2-SH LAMMPS remained at the previous precision:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.835793e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 5.076636e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.338534e-10 |

Performance clearly regressed. Same-node stable baseline in the verification
directory gave:

```text
stable gate run500 = 358.31 katom-step/s
stable no-gate run500 = 838.80 katom-step/s
```

Candidate gate `run 100` dropped to `302.01 katom-step/s`; the `run 500` part
was killed early after the profile already showed a large regression. Candidate
evflag0 profile:

```text
profile_gate_evflag0 total = 0.518202974 s
  first_layer = 0.0842114234 s
  first_finalize = 0.0460249403 s
  first_derivs = 0.1104601050 s
  main_basic = 0.0916069150 s
  products = 0.0169574348 s
  nbh_der = 0.0328864601 s
  main_force = 0.1176114650 s
  chain_force = 0.0015211751 s
profile_nogate_evflag0 total = 0.231030747 s
```

Compared with the accepted packed-radial profile (`gate total = 0.443920429 s`,
`first_layer = 0.0504187100 s`, `main_basic = 0.0510071991 s`), the scratch
reduction made the two affected kernels much slower. The likely reason is that
the added scratch footprint and explicit team reduction reduce occupancy and
increase serial work more than the local atomic removal helps.

Decision: rejected. The active remote source was restored to the accepted
packed-radial version:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Rejected candidate: Kokkos team-size environment sweep

Date: 2026-06-09.

Candidate change: expose GPU Kokkos team sizes through environment variables
for the alpha/basic, first-layer gate derivative, main gate force, gate chain,
and no-gate force kernels. The default launch sizes were unchanged unless an
environment variable was set, so the candidate was mathematically identical to
the accepted packed-radial version at default settings.

Build and verification:

```text
build job: 3771303 on c04u01g
verification job: 3771304 on c04u01g
performance job: 3771305 on c04u01g
directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/teamenv_candidate_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_teamenv_candidate_20260609
candidate sha256:
61a1f713553c496cdcb933dc9a870ea416d76ea778ff8bcaee5f2933433591d5
```

Correctness against CPU SUS2-SH LAMMPS remained at the previous precision:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 5.076636e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.348417e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 5.076636e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.470373e-10 |

The best 100-step gate setting was:

```text
SUS2_SH_KK_ALPHA_TEAM_SIZE=32
SUS2_SH_KK_GATE_DERIV_TEAM_SIZE=64
```

Same-node `run 500` comparison at `chunksize 142272`:

| Version | gate katom-step/s | no-gate katom-step/s |
| --- | ---: | ---: |
| stable packed-radial | 371.084 | 857.649 |
| candidate default | 370.968 | 857.946 |
| candidate best gate env | 379.959 | 830.342 |

Decision: rejected as a source/default optimization. The best gate-only setting
improves gate throughput by about `2.4%`, but the same setting slows no-gate by
about `3.2%` and still leaves gate below the `400 katom-step/s` target. This is
useful only as an optional gate benchmark/runtime tuning knob, not as a generic
algorithmic improvement. The active remote source and canonical binary were
restored to the accepted packed-radial version:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Rejected candidate: first-deriv valid-edge zero-write removal

Date: 2026-06-09.

Candidate change: in the two first-layer derivative kernels, remove the
unconditional three-component zero write at the top of each neighbor edge and
write zeros only for invalid edges outside `max_cutoff_sq`. Valid edges still
write the final `d f_i / d r_ij` values before `chain_force` reads the buffer,
so the semantics were unchanged:

```text
invalid edge:
  d_two_layer_gate_first_derivs(i,j,:) = 0

valid edge:
  d_two_layer_gate_first_derivs(i,j,:) =
    d/dr_ij sum_mu R_gate_ij,mu(r_ij) dot_mu(Y)
```

Build and verification:

```text
build job: 3771027 on c04u01g
verify job: 3771102 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/firstderiv_zero_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_firstderiv_zero_candidate_20260609
candidate sha256:
d8043508dd6d083911c4838d24d14e618c9181db940a1b9322c6c51b3e881927
```

Correctness against CPU SUS2-SH LAMMPS stayed at the accepted precision:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.691267e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.955619e-10 |

Same-node speed comparison at `chunksize 142272`:

| Version | gate katom-step/s | no-gate katom-step/s |
| --- | ---: | ---: |
| stable packed-radial | 371.055 | 855.274 |
| first-deriv zero-write candidate | 370.871 | 855.157 |

The candidate also lost slightly in the per-step profile:

```text
stable gate profile total = 0.425523885 s
  first_derivs = 0.111598248 s
  main_force = 0.115538403 s
candidate gate profile total = 0.426252004 s
  first_derivs = 0.111877210 s
  main_force = 0.115810070 s
```

The saved invalid-edge writes are not in the hot path for this benchmark and
the changed store pattern did not reduce memory traffic enough to matter.

Decision: rejected. The active remote source and canonical binary were restored
to the accepted packed-radial version:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Rejected candidate: gate SH/table-only force tags

Date: 2026-06-09.

Candidate change: add two dedicated Kokkos tags for the current GPU gate
combination:

```text
TagPairSUS2MTPComputeGateFirstDerivsSHTable
TagPairSUS2MTPComputeGateMainForceSHTable
```

The dispatch was restricted to the mathematically identical fast path where the
model is SUS2-SH, the shared-radial gate table exists, the main radial packed
table exists, and all `species_count^2` pairs are tabulated. The fallback tags
were left in place for `/kk/host`, partial tables, and any non-current gate
combination. The formula was unchanged:

```text
d f_i / d r_ij =
  sum_mu d/dr [R_gate_ij,mu(r_ij) sum_k c_i,k Y_k(rhat_ij)]

main force and gate adjoint use:
  R_main_ij,mu(r_ij) [1 + A tanh(a_Zj,mu f_j)]
  d/df_j = R_main_ij,mu(r_ij) A a_Zj,mu sech^2(a_Zj,mu f_j)
```

Build and verification:

```text
build job: 3770899 on c04u01g
verify job: 3770900 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/shtable_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_shtable_candidate_20260609
candidate sha256:
3bac5d40375cf54ae94d0acaf8972abdccbb83159db1dc3ae37c7001a4771d16
```

Correctness against CPU SUS2-SH LAMMPS remained at the previous precision:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.829126e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.967319e-10 |

Same-node speed comparison at `chunksize 142272`:

| Version | gate katom-step/s | no-gate katom-step/s |
| --- | ---: | ---: |
| stable packed-radial | 371.014 | 855.589 |
| SH/table-only candidate | 363.252 | 855.495 |

The profile showed only a small `first_derivs` improvement and a small
`main_force` regression:

```text
stable gate profile total = 0.426452629 s
  first_derivs = 0.111686617 s
  main_force = 0.115211779 s
candidate gate profile total = 0.425553752 s
  first_derivs = 0.111240005 s
  main_force = 0.115843523 s
```

The 500-step throughput regressed despite the slightly lower profile total,
which indicates this fast-path specialization is within noise and may worsen
instruction/register behavior in the real run. It also increased the monolithic
Kokkos translation unit build wall time to `448 s`.

Decision: rejected. The active remote source and canonical binary were restored
to the accepted packed-radial version:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Accepted runtime setting: one chunk for the 2x2x2 benchmark

Date: 2026-06-09.

No source code changed. The benchmark system has `inum = 142272` after
`replicate 2 2 2`. The previous default test setting `chunksize 129600` split
both gate layers into two chunks, while `chunksize 142272` keeps each layer in
one chunk. This reduces fixed per-chunk overhead without changing the math.

Job and directory:

```text
chunk tune job: 3770880 on c04u01g
directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/chunk_tune_20260609
binary sha256:
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

Measured `run 500` performance:

| chunksize | gate katom-step/s | no-gate katom-step/s | gate chunks | main chunks |
| ---: | ---: | ---: | ---: | ---: |
| 129600 | 358.659 | 836.557 | 2 | 2 |
| 142272 | 370.853 | 855.228 | 1 | 1 |

Representative profile totals:

```text
chunksize 129600 gate profile total = 0.441479186 s
chunksize 142272 gate profile total = 0.426689608 s
chunksize 129600 no-gate profile total = 0.225758208 s
chunksize 142272 no-gate profile total = 0.219275195 s
```

Decision: accepted as the current best benchmark configuration. For this
benchmark, run with:

```text
pair_style sus2mtp/kk/device MODEL.mtp chunksize 142272 tabstep 0.0005
```

Equivalently, use a `chunksize` at least as large as `inum` if the goal is to
avoid chunk splitting on this single-GPU benchmark. This is a runtime setting,
not a new binary.

### Rejected candidate: grouped SH index fast path

Date: 2026-06-09.

Candidate change: detect whether each `mu` group is contiguous in both
`alpha_index_basic` and the SH component index, then use `first_k + offset` and
`first_sh + offset` in the two gate force hot loops. The fallback path remained
the original indexed loop, so the formula was unchanged for all `lk` models:

```text
dot_mu = sum_{k in group(mu)} coeff_k Y_k(rhat)
F terms use d/dx [R_mu(r) dot_mu]
gate adjoint uses R_mu(r) dgate_mu/df dot_mu
```

Build and verification:

```text
build job: 3770881 on c04u01g
verify job: 3770882 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/groupfast_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_groupfast_candidate_20260609
candidate sha256:
7e564aa7997b0ecd40fb9c27ca6f56d4180c11225bf1a129ab41b81f61a72bd5
```

Correctness against CPU SUS2-SH LAMMPS remained at the previous precision:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 8.277164e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 4.840382e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 8.956876e-10 |

Same-node speed comparison at `chunksize 142272`:

| Version | gate katom-step/s | no-gate katom-step/s |
| --- | ---: | ---: |
| stable packed-radial | 370.828 | 855.601 |
| grouped-index candidate | 372.469 | 855.441 |

The gate profile total moved only from `0.425761543 s` to `0.425490489 s`.
The candidate also increased build wall time to `438 s`. The measured gate
throughput increase is only about `0.44%`, which is within benchmark noise for
this setup and does not justify the extra branch and state in the monolithic
Kokkos translation unit.

Decision: rejected. The active remote source and canonical binary were restored
to the accepted packed-radial version:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Rejected final candidate: packed two-layer gate derivative table

Date: 2026-06-09.

Candidate change: use the existing gate derivative table view as a packed
`[value, derivative]` table for first-layer derivative kernels. The original
first-layer forward table was retained for the forward gate pass, while the
manual derivative hot loops loaded both `R_gate` and `dR_gate/dr` from the
packed derivative table. The mathematical formula was unchanged:

```text
f_i = sum_mu R_gate_ij,mu(r_ij) dot_mu[Y(rhat_ij)]
d f_i / d r_ij =
  sum_mu (dR_gate_ij,mu/dr dot_mu + R_gate_ij,mu d dot_mu/dr)
```

Build and verification:

```text
build job: 3771308 on c04u01g
verify job: 3771316 on c04u01g
performance job: 3771317 on c04u01g
verify directory:
/work/phy-weigw/hyx/5.28-mof-cl-h2o/gate/lammps_gate_vs_nogate/codex_gatekk_gate_team_verify_20260608/gatepackder_verify_20260609
candidate binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_gatepackder_candidate_20260609
candidate sha256:
abe14258b961efb06ed0f6e2e1a4bcae34f61f35a2b5c141460701394d13e8a2
```

Correctness against CPU SUS2-SH LAMMPS passed before the verify script hit its
known analysis-order bug:

| Case | dE | dPress | max force diff | RMS force diff |
| --- | ---: | ---: | ---: | ---: |
| gate run0 | 0 | 0 | 1.0e-8 | 4.840382e-11 |
| no-gate run0 | 0 | 0 | 1.0e-7 | 9.217308e-10 |
| gate force1 | n/a | n/a | 1.0e-8 | 5.076636e-11 |
| no-gate force1 | n/a | n/a | 1.0e-7 | 9.085433e-10 |

Same-node speed comparison at `chunksize 142272`:

| Version | gate katom-step/s | no-gate katom-step/s |
| --- | ---: | ---: |
| stable packed-radial | 371.128 | 857.351 |
| packed gate-derivative candidate | 370.641 | 857.361 |

The single-step profile moved in the right direction for gate but not enough
to survive the 500-step benchmark:

```text
stable gate profile total = 0.426541641 s
  first_derivs = 0.111409232 s
  nbh_der = 0.0367241576 s
  main_force = 0.115543306 s
candidate gate profile total = 0.425299760 s
  first_derivs = 0.111268688 s
  nbh_der = 0.0356480759 s
  main_force = 0.115321213 s
```

Decision: rejected. The candidate was numerically correct but gate throughput
regressed from `371.128` to `370.641` katom-step/s in the 500-step run. The
active remote source was restored to the accepted packed-radial version and
the canonical binary was left unchanged:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

### Final GPU Kokkos version after the last attempt

Date: 2026-06-09.

Final decision: keep the accepted packed-radial GPU Kokkos source and the
`chunksize 142272 tabstep 0.0005` benchmark setting. The last candidate did
not improve the 500-step gate throughput, so no further source candidate is
accepted in this round.

Final canonical files:

```text
remote LAMMPS source:
/work/phy-weigw/apps/lammps-10Dec2025/src/KOKKOS/pair_sus2_mtp_kokkos.cpp
/work/phy-weigw/apps/lammps-10Dec2025/src/KOKKOS/pair_sus2_mtp_kokkos.h

canonical GPU binary:
/work/phy-weigw/20260321_Test/lammps-sus2kk-v45-all-double-centroidstress/lmp.v45_all_double_centroidstress_tabstep_double_compute1
```

Final hashes:

```text
pair_sus2_mtp_kokkos.cpp sha256 =
f8b30d03590e2dcbd30d1d6ef8caa2ecb6a4e01e1796643489ee03fbdfe5eead
pair_sus2_mtp_kokkos.h sha256 =
e32286b970392a6f40992d3aa66b66fa19631db68e9ef9f1317562ff742ef8c2
canonical binary sha256 =
4a4fcf99b93cc71a83295d76c801672a3c7f32be9303a9b31a17b9a267750c31
```

Final measured `run 500` numbers on the benchmark system:

| Case | katom-step/s | Status |
| --- | ---: | --- |
| no-gate SH | 857.350-857.361 | above the 800 target |
| two-layer gate | 371.128 | below the 400 target |

The remaining gate bottleneck is still compute-side kernel work rather than
communication. In the accepted profile at `inum = 142272`, the largest gate
terms are `main_force ~= 0.116 s`, `first_derivs ~= 0.111 s`,
`main_basic ~= 0.051 s`, and `first_layer ~= 0.049 s`; forward/reverse comm is
only about `0.00035 s` combined.

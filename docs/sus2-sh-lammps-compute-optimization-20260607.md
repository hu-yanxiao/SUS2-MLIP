# SUS2-SH LAMMPS Compute Optimization Record - 2026-06-07

This file records accepted stage-best LAMMPS compute-side optimization states.
Temporary rejected candidates and raw scripts remain in `.codex_tmp/`.

## Stage-Best Version: Static-Fixed Gate Path + Mu-Grouped SH Edge

Status: accepted stage-best as of 2026-06-07 23:28 CST.

Git commit:

```text
7e7473600d2358c0d8e91d96b9462670520964ad
Optimize SUS2-SH LAMMPS gate static paths
```

Installed CPU LAMMPS binary:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi
sha256 77c3d914a4a03acdd98c703836e4ab46116546fb8d125464c59070d2839af13c
```

Previous installed binary backup:

```text
/work/phy-weigw/cpu-lammps/lmp.ml-sus2_tabstep_intelmpi.bak_20260607_before_sh_mu_group
sha256 938d1362dd1e08089e3d083f000408ac42ff7edd8855425dd021a35ea14b68cb
```

Accepted changes in this stage:

- `static_fixed_types` no-gate fixed-center/fixed-neighbor SH basic cache.
- Gate first-layer fixed-center/fixed-neighbor SH basic cache.
- Gate first-layer product-limit pruning from `two_layer_gate_scalar_indices`.
- Gate main-layer fixed-fixed separable cache for additive tanh gate models.
- Gate fixed-fixed dynamic edge-list pruning.
- Generic SH basic edge loop grouped by contiguous `mu=(l,k)` blocks, with fallback to the original loop if the model layout is not `mu` grouped.

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
| gate | 6.0571 s | 5.0889 s | 5.0913 s | 1.190x | 1.187x |
| no-gate | 2.1471 s | 1.8530 s | 1.8545 s | 1.158x | 1.198x |

Direct conclusion:

- gate Pair compute is about `+19.0%` faster than the initial optimization baseline.
- no-gate Pair compute is about `+15.8%` faster than the initial optimization baseline.
- no-gate loop time is about `+19.8%` faster because communication/modify balance also improved in this run.

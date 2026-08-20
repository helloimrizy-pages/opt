# RACE Stage 2 required tables

## Table 1 — Main frozen-suite comparison (unit expert transfers)

| Capacity | Stage1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 | 12,849,725 |
| 12 | 9,748,279 | 9,999,390 | 9,919,727 | 9,851,604 | 9,833,069 | 8,146,471 |
| 16 | 7,822,852 | 8,120,618 | 8,005,697 | 7,945,937 | 7,939,359 | 5,904,787 |
| 24 | 5,081,058 | 5,333,296 | 5,230,074 | 5,181,473 | 5,245,722 | 3,294,951 |
| 32 | 3,159,525 | 3,335,492 | 3,233,051 | 3,203,094 | 3,323,031 | 1,785,846 |

Normalized by the Stage 0 strongest-simple cost at the same capacity:

| Capacity | Stage1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 0.9839 | 1.0093 | 1.0012 | 0.9944 | 0.9925 | 0.8223 |
| 16 | 0.9715 | 1.0085 | 0.9942 | 0.9868 | 0.9860 | 0.7333 |
| 24 | 0.9581 | 1.0057 | 0.9862 | 0.9771 | 0.9892 | 0.6213 |
| 32 | 0.9575 | 1.0109 | 0.9798 | 0.9707 | 1.0071 | 0.5412 |

## Table 2 — Success metrics for the frozen primary variant `RACE_ONLINE`

| Capacity | Improvement vs Stage1 (95% CI) | Original oracle gap closed (95% CI) | Stage1 residual recovered (95% CI) |
| ---: | ---: | ---: | ---: |
| 12 | -1.06% [-1.15, -0.96] | 3.17% [2.90, 3.44] | -6.45% [-7.02, -5.83] |
| 16 | -1.57% [-1.71, -1.43] | 4.96% [4.50, 5.38] | -6.42% [-7.04, -5.84] |
| 24 | -1.98% [-2.23, -1.74] | 6.05% [5.30, 6.76] | -5.62% [-6.40, -4.96] |
| 32 | -1.38% [-1.75, -1.04] | 6.38% [5.42, 7.21] | -3.17% [-4.05, -2.38] |

## Table 3 — Frozen primary variant by workload regime

| Regime | Capacity | Stage0 simple | Stage1 | RACE | Oracle | Improvement vs Stage1 | Gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stationary | 8 | 2,992,620 | 2,992,620 | 2,992,620 | 2,992,620 | 0.00% | N/A |
| stationary | 12 | 2,305,638 | 2,264,721 | 2,279,645 | 1,896,388 | -0.66% | 6.35% |
| stationary | 16 | 1,872,290 | 1,814,668 | 1,836,079 | 1,375,283 | -1.18% | 7.29% |
| stationary | 24 | 1,230,577 | 1,175,318 | 1,201,464 | 768,001 | -2.22% | 6.29% |
| stationary | 32 | 763,662 | 728,401 | 742,036 | 416,358 | -1.87% | 6.23% |
| abrupt | 8 | 5,926,143 | 5,926,143 | 5,926,143 | 5,926,143 | 0.00% | N/A |
| abrupt | 12 | 4,552,306 | 4,488,773 | 4,535,807 | 3,752,707 | -1.05% | 2.06% |
| abrupt | 16 | 3,693,108 | 3,598,607 | 3,657,652 | 2,715,747 | -1.64% | 3.63% |
| abrupt | 24 | 2,424,479 | 2,335,230 | 2,377,466 | 1,511,098 | -1.81% | 5.15% |
| abrupt | 32 | 1,502,762 | 1,449,860 | 1,465,212 | 816,437 | -1.06% | 5.47% |
| repeated | 8 | 936,595 | 936,595 | 936,595 | 936,595 | 0.00% | N/A |
| repeated | 12 | 727,303 | 716,788 | 726,356 | 598,832 | -1.33% | 0.74% |
| repeated | 16 | 592,679 | 577,674 | 587,541 | 436,094 | -1.71% | 3.28% |
| repeated | 24 | 390,600 | 377,218 | 382,562 | 244,364 | -1.42% | 5.50% |
| repeated | 32 | 243,124 | 235,345 | 236,395 | 132,133 | -0.45% | 6.06% |
| mixed | 8 | 2,994,367 | 2,994,367 | 2,994,367 | 2,994,367 | 0.00% | N/A |
| mixed | 12 | 2,322,183 | 2,277,997 | 2,309,796 | 1,898,544 | -1.40% | 2.92% |
| mixed | 16 | 1,894,325 | 1,831,903 | 1,864,665 | 1,377,663 | -1.79% | 5.74% |
| mixed | 24 | 1,257,349 | 1,193,292 | 1,219,981 | 771,488 | -2.24% | 7.69% |
| mixed | 32 | 790,086 | 745,919 | 759,451 | 420,918 | -1.81% | 8.30% |

## Table 4 — Ablation decomposition (frozen ten-workload suite)

| Question | Comparison | Capacity | Before | After | Relative change |
| --- | --- | ---: | ---: | ---: | ---: |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 12 | 9,748,279 | 9,999,390 | -2.58% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 16 | 7,822,852 | 8,120,618 | -3.81% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 24 | 5,081,058 | 5,333,296 | -4.96% |
| A_multi_horizon | Stage 1 winner -> RACE_UNIFORM | 32 | 3,159,525 | 3,335,492 | -5.57% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 12 | 9,999,390 | 9,919,727 | +0.80% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 16 | 8,120,618 | 8,005,697 | +1.42% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 24 | 5,333,296 | 5,230,074 | +1.94% |
| B_static_weights | RACE_UNIFORM -> RACE_STATIC | 32 | 3,335,492 | 3,233,051 | +3.07% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 12 | 9,919,727 | 9,851,604 | +0.69% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 16 | 8,005,697 | 7,945,937 | +0.75% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 24 | 5,230,074 | 5,181,473 | +0.93% |
| C_online_adaptation | RACE_STATIC -> RACE_ONLINE | 32 | 3,233,051 | 3,203,094 | +0.93% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 12 | 9,851,604 | 9,833,069 | +0.19% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 16 | 7,945,937 | 7,939,359 | +0.08% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 24 | 5,181,473 | 5,245,722 | -1.24% |
| D_cost_sensitivity | RACE_ONLINE -> RACE_COST | 32 | 3,203,094 | 3,323,031 | -3.74% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 12 | 9,999,390 | 9,944,175 | +0.55% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 16 | 8,120,618 | 8,038,921 | +1.01% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 24 | 5,333,296 | 5,254,771 | +1.47% |
| E_adviser_diversity_uniform | RACE_UNIFORM -> RACE_UNIFORM_EXTENDED | 32 | 3,335,492 | 3,276,729 | +1.76% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 12 | 9,851,604 | 9,738,992 | +1.14% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 16 | 7,945,937 | 7,811,937 | +1.69% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 24 | 5,181,473 | 5,066,820 | +2.21% |
| F_adviser_diversity_online | RACE_ONLINE -> RACE_ONLINE_EXTENDED | 32 | 3,203,094 | 3,137,922 | +2.03% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 12 | 9,748,279 | 9,738,992 | +0.10% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 16 | 7,822,852 | 7,811,937 | +0.14% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 24 | 5,081,058 | 5,066,820 | +0.28% |
| G_extended_vs_stage1 | Stage 1 winner -> RACE_ONLINE_EXTENDED | 32 | 3,159,525 | 3,137,922 | +0.68% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 12 | 9,919,727 | 9,912,736 | +0.07% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 16 | 8,005,697 | 7,992,796 | +0.16% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 24 | 5,230,074 | 5,222,621 | +0.14% |
| H_per_layer_static | RACE_STATIC -> RACE_STATIC_PERLAYER | 32 | 3,233,051 | 3,234,006 | -0.03% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 12 | 9,851,604 | 9,887,855 | -0.37% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 16 | 7,945,937 | 8,009,648 | -0.80% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 24 | 5,181,473 | 5,274,423 | -1.79% |
| I_weight_scope | RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL | 32 | 3,203,094 | 3,302,906 | -3.12% |

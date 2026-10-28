# Free projective IEKF profile

scene `blade_103706`, first 5 frames, forward, NVIDIA GeForce RTX 4070 Laptop GPU

| factor | precision | N selected | frame wall ms | frame CUDA ms | peak alloc MB | rows | observed | not conv. | vs float64 position diff (median / max, m) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | float64 | 34369 | 203 | 192 | 208 | 31669 | 16986 | 3916 | — |
| 1 | float32 | 34369 | 213 | 198 | 143 | 31690 | 16964 | 4399 | 1.69e-07 / 1.36e+00 |
| 1 | mixed | 34369 | 221 | 208 | 190 | 31672 | 16970 | 4374 | 1.29e-07 / 1.06e+00 |
| 2 | float64 | 17185 | 199 | 187 | 143 | 27371 | 13106 | 4109 | — |
| 2 | float32 | 17185 | 178 | 166 | 101 | 27279 | 13087 | 4522 | 4.12e-07 / 1.42e+00 |
| 2 | mixed | 17185 | 191 | 177 | 132 | 27325 | 13079 | 4550 | 2.63e-07 / 1.25e+00 |
| 4 | float64 | 8593 | 185 | 173 | 109 | 18774 | 7541 | 3516 | — |
| 4 | float32 | 8593 | 198 | 172 | 81 | 18769 | 7559 | 3854 | 4.10e-05 / 1.75e+00 |
| 4 | mixed | 8593 | 194 | 182 | 99 | 18773 | 7564 | 3811 | 6.66e-06 / 1.65e+00 |
| 8 | float64 | 4297 | 146 | 134 | 86 | 11474 | 3946 | 2335 | — |
| 8 | float32 | 4297 | 137 | 124 | 69 | 11364 | 3932 | 2544 | 2.23e-03 / 1.68e+00 |
| 8 | mixed | 4297 | 142 | 129 | 81 | 11421 | 3945 | 2540 | 1.80e-03 / 1.65e+00 |

stage CUDA ms (median), first row:

| stage | ms |
|---|---|
| transfer_images | 95.9 |
| sampling_terms | 20.1 |
| visibility | 17.1 |
| solve | 16.1 |
| information_accumulation | 12.0 |
| prior_information | 5.3 |
| predict | 5.2 |
| projection | 5.0 |
| covariance | 3.6 |
| store | 3.5 |

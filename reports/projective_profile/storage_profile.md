# Free projective IEKF profile

scene `storage_45135`, first 4 frames, forward, NVIDIA GeForce RTX 4070 Laptop GPU

| factor | precision | N selected | frame wall ms | frame CUDA ms | peak alloc MB | rows | observed | not conv. | vs float64 position diff (median / max, m) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | float64 | 346437 | 1006 | 935 | 1612 | 330614 | 182668 | 89249 | — |
| 1 | mixed | 346437 | 963 | 892 | 1417 | 331214 | 183185 | 90364 | 2.14e-07 / 2.25e+00 |
| 2 | float64 | 173219 | 610 | 583 | 924 | 260774 | 132351 | 68378 | — |
| 2 | mixed | 173219 | 549 | 523 | 820 | 261272 | 132750 | 69376 | 2.52e-07 / 2.25e+00 |
| 4 | float64 | 86610 | 383 | 369 | 580 | 174024 | 77015 | 41922 | — |
| 4 | mixed | 86610 | 354 | 340 | 519 | 174188 | 77230 | 42568 | 2.51e-07 / 2.09e+00 |

stage CUDA ms (median), first row:

| stage | ms |
|---|---|
| visibility | 241.2 |
| solve | 193.1 |
| sampling_terms | 110.9 |
| information_accumulation | 96.6 |
| transfer_images | 83.7 |
| projection | 56.7 |
| predict | 48.6 |
| prior_information | 40.9 |
| covariance | 39.6 |
| store | 23.6 |

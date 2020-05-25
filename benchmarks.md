**u16 to u16 samples**

|        | Time for 100,000 Samples |
|--------|--------------------------|
| scalar | 245 μs                   |
| AVX    | 151 μs                   |
| AVX2   | 127 μs                   |

**f32 to s16 samples**

|                  | Time for 100,000 Samples |
|------------------|--------------------------|
| autovec disabled | 321 μ                    |
| SSE2             | 215 μs                   |
| AVX              | 245 μs                   |
| AV2              | 207 μs                   |
| AVX2 Manual      | 104 μs                   |

**f32 to s16 samples on Xeon**
```
Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz
```
```
scalar                  time:   [223.17 us 223.18 us 223.19 us]
                        change: [+9837.7% +9844.2% +9853.1%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 6 outliers among 100 measurements (6.00%)
  2 (2.00%) low severe
  2 (2.00%) low mild
  2 (2.00%) high severe

avx                     time:   [251.13 us 251.15 us 251.18 us]
                        change: [+9829.4% +9835.5% +9840.9%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 10 outliers among 100 measurements (10.00%)
  1 (1.00%) low severe
  5 (5.00%) high mild
  4 (4.00%) high severe

avx2                    time:   [253.39 us 253.45 us 253.51 us]
                        change: [+10139% +10151% +10160%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high severe

avx2 manual             time:   [191.99 us 192.01 us 192.03 us]
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) low severe
  2 (2.00%) high mild
  1 (1.00%) high severe
  ```
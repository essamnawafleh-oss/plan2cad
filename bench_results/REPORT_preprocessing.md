# Benchmark comparison

Files: `bench_results/pp_default.json`, `bench_results/pp_denoise_on.json`, `bench_results/pp_minimal.json`, `bench_results/pp_no_light.json`, `bench_results/pp_no_persp.json`

Runs with errors: pp_default: 0, pp_denoise_on: 0, pp_minimal: 0, pp_no_light: 0, pp_no_persp: 0

## photo drawings, calibrated

| Metric | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|
| runs | 6 | 6 | 6 | 6 | 6 |
| Wall F1 | 0.973 | 0.973 | 0.928 | 0.969 | 0.915 |
| Precision | 0.953 | 0.953 | 0.898 | 0.949 | 0.910 |
| Recall | 0.994 | 0.994 | 0.961 | 0.990 | 0.935 |
| IoU | 0.906 | 0.902 | 0.695 | 0.892 | 0.808 |
| Endpoint err (m) | 0.042 | 0.042 | 0.120 | 0.040 | 0.101 |
| Thickness err (m) | 0.000 | 0.000 | 0.015 | 0.009 | 0.004 |
| Junctions connected | 0.97 | 0.97 | 0.88 | 0.97 | 0.89 |
| Missing walls | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| False walls | 1.00 | 1.00 | 2.33 | 1.33 | 2.17 |
| Door F1 | 0.987 | 0.974 | 0.827 | 0.972 | 0.954 |
| Window F1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.937 |
| Pipeline time (s) | 9.0 | 8.9 | 33.5 | 24.6 | 13.1 |
| NN time (s) | 0.3 | 0.3 | 0.4 | 0.3 | 0.4 |
| Import time (s) | 1.04 | 1.12 | 0.01 | 0.94 | 1.60 |
| DXF time (s) | 0.20 | 0.20 | 0.37 | 0.36 | 0.23 |
| Peak memory (MB) | 1536 | 1538 | 2097 | 2158 | 1882 |

## scan drawings, calibrated

| Metric | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|
| runs | 2 | 2 | 2 | 2 | 2 |
| Wall F1 | 0.970 | 0.970 | 0.970 | 0.970 | 0.970 |
| Precision | 0.948 | 0.947 | 0.948 | 0.948 | 0.948 |
| Recall | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |
| IoU | 0.924 | 0.914 | 0.924 | 0.924 | 0.924 |
| Endpoint err (m) | 0.035 | 0.035 | 0.035 | 0.035 | 0.035 |
| Thickness err (m) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Junctions connected | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |
| Missing walls | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| False walls | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| Door F1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Window F1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Pipeline time (s) | 8.1 | 8.4 | 8.3 | 8.0 | 7.9 |
| NN time (s) | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 |
| Import time (s) | 0.22 | 0.44 | 0.01 | 0.16 | 0.19 |
| DXF time (s) | 0.20 | 0.19 | 0.19 | 0.19 | 0.18 |
| Peak memory (MB) | 1431 | 1434 | 1426 | 1427 | 1431 |

## Per case (calibrated, wall F1 / IoU)

| Case | seed | pp_default | pp_denoise_on | pp_minimal | pp_no_light | pp_no_persp |
|---|---|---|---|---|---|---|
| photo_persp | 1 | 0.960 / 0.824 | 0.960 / 0.824 | 0.832 / 0.533 | 0.955 / 0.846 | 0.801 / 0.535 |
| photo_persp | 2 | 0.984 / 0.926 | 0.984 / 0.926 | 0.882 / 0.596 | 0.984 / 0.926 | 0.794 / 0.629 |
| photo_rot90 | 1 | 0.960 / 0.891 | 0.960 / 0.872 | 0.960 / 0.871 | 0.960 / 0.871 | 0.960 / 0.891 |
| photo_rot90 | 2 | 0.984 / 0.949 | 0.984 / 0.945 | 0.983 / 0.950 | 0.983 / 0.950 | 0.984 / 0.949 |
| photo_shadow | 1 | 0.965 / 0.911 | 0.965 / 0.911 | 0.936 / 0.588 | 0.968 / 0.857 | 0.965 / 0.911 |
| photo_shadow | 2 | 0.984 / 0.935 | 0.984 / 0.935 | 0.975 / 0.632 | 0.966 / 0.902 | 0.984 / 0.935 |
| scan_noisy | 1 | 0.957 / 0.886 | 0.956 / 0.877 | 0.957 / 0.886 | 0.957 / 0.886 | 0.957 / 0.886 |
| scan_noisy | 2 | 0.984 / 0.962 | 0.983 / 0.951 | 0.984 / 0.962 | 0.984 / 0.962 | 0.984 / 0.962 |

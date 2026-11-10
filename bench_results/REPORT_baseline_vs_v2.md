# Benchmark comparison

Files: `bench_results/baseline.json`, `bench_results/v2.json`

Runs with errors: baseline: 0, v2: 0

## photo drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 6 | 6 |
| Wall F1 | 0.831 | 0.973 |
| Precision | 0.788 | 0.953 |
| Recall | 0.937 | 0.994 |
| IoU | 0.721 | 0.906 |
| Endpoint err (m) | 0.119 | 0.042 |
| Thickness err (m) | 0.039 | 0.000 |
| Junctions connected | 0.89 | 0.97 |
| Missing walls | 0.50 | 0.00 |
| False walls | 3.67 | 1.00 |
| Door F1 | 0.819 | 0.987 |
| Window F1 | 0.955 | 1.000 |
| Pipeline time (s) | 30.0 | 8.4 |
| NN time (s) | 0.4 | 0.3 |
| Import time (s) | 1.41 | 1.08 |
| DXF time (s) | 0.43 | 0.20 |
| Peak memory (MB) | 2332 | 1533 |

## photo drawings, uncalibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 6 | 6 |
| Wall F1 | 0.810 | 0.976 |
| Precision | 0.749 | 0.959 |
| Recall | 0.929 | 0.994 |
| IoU | 0.742 | 0.887 |
| Endpoint err (m) | 0.120 | 0.040 |
| Thickness err (m) | 0.037 | 0.021 |
| Junctions connected | 0.90 | 0.97 |
| Missing walls | 0.50 | 0.00 |
| False walls | 4.67 | 0.50 |
| Door F1 | 0.901 | 0.957 |
| Window F1 | 0.921 | 1.000 |
| Pipeline time (s) | 32.2 | 11.6 |
| NN time (s) | 3.4 | 3.2 |
| Import time (s) | 1.40 | 1.06 |
| DXF time (s) | 0.41 | 0.19 |
| Peak memory (MB) | 2329 | 1533 |
| Scale error (rel.) | 0.112 | 0.096 |

## real drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 1 | 1 |
| Wall F1 | 0.666 | 0.688 |
| Precision | 0.589 | 0.620 |
| Recall | 0.765 | 0.772 |
| IoU | 0.408 | 0.399 |
| Endpoint err (m) | 0.296 | 0.302 |
| Thickness err (m) | 0.076 | 0.107 |
| Junctions connected | 0.65 | 0.65 |
| Missing walls | 38.00 | 34.00 |
| False walls | 111.00 | 103.00 |
| Pipeline time (s) | 61.3 | 60.7 |
| NN time (s) | 1.4 | 1.5 |
| Import time (s) | 0.34 | 0.33 |
| DXF time (s) | 0.76 | 0.67 |
| Peak memory (MB) | 6022 | 6047 |

## scan drawings, calibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 18 | 18 |
| Wall F1 | 0.951 | 0.965 |
| Precision | 0.915 | 0.946 |
| Recall | 0.992 | 0.985 |
| IoU | 0.783 | 0.869 |
| Endpoint err (m) | 0.037 | 0.043 |
| Thickness err (m) | 0.052 | 0.016 |
| Junctions connected | 0.98 | 0.95 |
| Missing walls | 0.00 | 0.06 |
| False walls | 1.83 | 1.61 |
| Door F1 | 0.995 | 0.969 |
| Window F1 | 0.962 | 0.978 |
| Pipeline time (s) | 11.0 | 9.6 |
| NN time (s) | 0.3 | 0.3 |
| Import time (s) | 0.11 | 0.14 |
| DXF time (s) | 0.22 | 0.20 |
| Peak memory (MB) | 1487 | 1490 |

## scan drawings, uncalibrated

| Metric | baseline | v2 |
|---|---|---|
| runs | 18 | 18 |
| Wall F1 | 0.954 | 0.973 |
| Precision | 0.914 | 0.955 |
| Recall | 0.998 | 0.992 |
| IoU | 0.782 | 0.880 |
| Endpoint err (m) | 0.040 | 0.039 |
| Thickness err (m) | 0.059 | 0.022 |
| Junctions connected | 0.99 | 0.97 |
| Missing walls | 0.00 | 0.00 |
| False walls | 1.06 | 0.72 |
| Door F1 | 0.991 | 0.971 |
| Window F1 | 0.938 | 0.970 |
| Pipeline time (s) | 12.9 | 12.7 |
| NN time (s) | 2.6 | 2.8 |
| Import time (s) | 0.12 | 0.15 |
| DXF time (s) | 0.21 | 0.21 |
| Peak memory (MB) | 1486 | 1490 |
| Scale error (rel.) | 0.086 | 0.085 |

## Per case (calibrated, wall F1 / IoU)

| Case | seed | baseline | v2 |
|---|---|---|---|
| angled | 1 | 0.944 / 0.796 | 0.962 / 0.886 |
| angled | 2 | 0.968 / 0.834 | 0.973 / 0.923 |
| clean_filled | 1 | 0.943 / 0.780 | 0.957 / 0.872 |
| clean_filled | 2 | 0.977 / 0.848 | 0.983 / 0.947 |
| clean_hatched | 1 | 0.926 / 0.748 | 0.952 / 0.776 |
| clean_hatched | 2 | 0.972 / 0.797 | 0.983 / 0.851 |
| clean_hollow | 1 | 0.900 / 0.681 | 0.933 / 0.774 |
| clean_hollow | 2 | 0.961 / 0.806 | 0.983 / 0.867 |
| colour_cad | 1 | 0.914 / 0.773 | 0.954 / 0.817 |
| colour_cad | 2 | 0.972 / 0.796 | 0.983 / 0.864 |
| photo_persp | 1 | 0.801 / 0.490 | 0.960 / 0.824 |
| photo_persp | 2 | 0.492 / 0.538 | 0.984 / 0.926 |
| photo_rot90 | 1 | 0.947 / 0.802 | 0.960 / 0.891 |
| photo_rot90 | 2 | 0.983 / 0.818 | 0.984 / 0.949 |
| photo_shadow | 1 | 0.953 / 0.820 | 0.965 / 0.911 |
| photo_shadow | 2 | 0.812 / 0.857 | 0.984 / 0.935 |
| real:plan1 | 0 | 0.666 / 0.408 | 0.688 / 0.399 |
| scan_lowres | 1 | 0.945 / 0.822 | 0.959 / 0.893 |
| scan_lowres | 2 | 0.978 / 0.827 | 0.984 / 0.884 |
| scan_noisy | 1 | 0.934 / 0.757 | 0.957 / 0.886 |
| scan_noisy | 2 | 0.978 / 0.794 | 0.984 / 0.962 |
| scan_rot3 | 1 | 0.950 / 0.827 | 0.964 / 0.887 |
| scan_rot3 | 2 | 0.978 / 0.859 | 0.965 / 0.910 |
| thin_walls | 1 | 0.952 / 0.676 | 0.961 / 0.826 |
| thin_walls | 2 | 0.928 / 0.666 | 0.928 / 0.826 |

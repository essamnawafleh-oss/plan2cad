#!/bin/sh
# Full benchmark: baseline checkout vs this checkout, then preprocessing configurations.
#   BASELINE=/path/to/old/checkout P2C_REAL_DIR=/path/with/work.png+wall_data.py sh bench/run_all.sh
set -e
OUT=${OUT:-bench_results}
mkdir -p "$OUT"
REAL=""
[ -n "$P2C_REAL_DIR" ] && REAL="--real plan1"
python3 bench/run_bench.py --code "$BASELINE" --out "$OUT/baseline.json" --uncalibrated $REAL
python3 bench/run_bench.py --code . --out "$OUT/v2.json" --uncalibrated $REAL
PH=photo_persp,photo_shadow,photo_rot90,scan_noisy
python3 bench/run_bench.py --code . --cases $PH --out "$OUT/pp_default.json"
python3 bench/run_bench.py --code . --cases $PH --out "$OUT/pp_no_persp.json" --pp '{"perspective": "off"}'
python3 bench/run_bench.py --code . --cases $PH --out "$OUT/pp_no_light.json" --pp '{"lighting": "off"}'
python3 bench/run_bench.py --code . --cases $PH --out "$OUT/pp_denoise_on.json" --pp '{"denoise": "on"}'
python3 bench/run_bench.py --code . --cases $PH --out "$OUT/pp_minimal.json" --pp '{"perspective": "off", "lighting": "off", "denoise": "off", "deskew": false}'
python3 bench/report.py "$OUT"/baseline.json "$OUT"/v2.json > "$OUT/REPORT_baseline_vs_v2.md"
python3 bench/report.py "$OUT"/pp_*.json > "$OUT/REPORT_preprocessing.md"

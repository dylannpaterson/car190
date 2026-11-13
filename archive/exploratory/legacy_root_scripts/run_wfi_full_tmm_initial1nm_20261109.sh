#!/usr/bin/env bash

set -u

statistics_dir="20261109/corrected_statistics_4h_thinned"
third_decon_day="38.79784510416666"
detectors=(
    WFI01 WFI02 WFI03 WFI04 WFI05 WFI06 WFI07 WFI08 WFI09
    WFI10 WFI11 WFI12 WFI13 WFI14 WFI15 WFI16 WFI17 WFI18
)

for detector in "${detectors[@]}"; do
    detector_lower="${detector,,}"
    printf '\n===== Starting %s =====\n' "$detector"
    .venv/bin/python run_wfi_production.py \
        --statistics-dir "$statistics_dir" \
        --detector "$detector" \
        --tile-size 256 \
        --third-decon-day "$third_decon_day" \
        --warmup-prior-root \
        "$statistics_dir/production/$detector_lower/relative_response_constant_rate_warmup_start40" \
        --thickness-prior-log-sigma 0.1 \
        --thickness-prior-stages third-decon \
        --fit-led-log-offsets \
        --led-log-offset-prior-sigma-log 0.01 \
        --workers 2 \
        --batch-size 1 \
        --tile-timeout-seconds 3600
    printf '===== Finished %s (status %s) =====\n' "$detector" "$?"
done

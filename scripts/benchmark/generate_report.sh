#!/usr/bin/env bash
# ==============================================================================
# generate_report.sh - Merge and display benchmark reports
#
# Scans per-model results in run_*/ directories, merges them into unified
# CSV + Markdown reports, and prints a formatted table to the terminal.
#
# Usage:
#   scripts/benchmark/generate_report.sh [options] [dataset ...]
#
# Options:
#   --output-dir <dir>   Where to write merged files (default: results base)
#   --results-base <dir> Base directory with run_*/ dirs (default: from config)
#   -h, --help           Show help
#
# Examples:
#   scripts/benchmark/generate_report.sh                    # all datasets
#   scripts/benchmark/generate_report.sh arxivqa            # one dataset
#   scripts/benchmark/generate_report.sh arxivqa slidevqa   # explicit list
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_env.sh"

# ── Defaults ─────────────────────────────────────────────────────────────────
results_base="${RESULTS_BASE}"
output_dir=""
positional_datasets=()

# ── Usage ────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: scripts/benchmark/generate_report.sh [options] [dataset ...]

Merge per-model benchmark results and display a formatted report.

Positional:
  dataset ...            Dataset names to report (default: all from config)

Options:
  --results-base <dir>   Directory containing run_*/ subdirs
                         (default: $RESULTS_BASE)
  --output-dir <dir>     Where to write merged CSV/Markdown
                         (default: same as results-base)
  -h, --help             Show this help message

Examples:
  scripts/benchmark/generate_report.sh                    # all configured datasets
  scripts/benchmark/generate_report.sh arxivqa            # just arxivqa
  scripts/benchmark/generate_report.sh --results-base /tmp/results arxivqa
EOF
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-base)  results_base="$2"; shift 2 ;;
        --output-dir)    output_dir="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        -*)              err "Unknown option: $1"; usage; exit 1 ;;
        *)               positional_datasets+=("$1"); shift ;;
    esac
done

output_dir="${output_dir:-$results_base}"

# Use positional datasets, or fall back to configured DATASETS
if [[ ${#positional_datasets[@]} -gt 0 ]]; then
    datasets="${positional_datasets[*]}"
else
    datasets="${DATASETS}"
fi

# ── Merge script path ───────────────────────────────────────────────────────
MERGE_SCRIPT="$PROJECT_ROOT/scripts/merge_benchmark_reports.py"
if [[ ! -f "$MERGE_SCRIPT" ]]; then
    err "Merge script not found: $MERGE_SCRIPT"
    exit 1
fi

# ── Generate reports ─────────────────────────────────────────────────────────
step "Generating benchmark reports..."
echo -e "  ${DIM}Results base :${NC} $results_base"
echo -e "  ${DIM}Output dir   :${NC} $output_dir"
echo -e "  ${DIM}Datasets     :${NC} $datasets"
echo ""

has_results=false

for dataset in $datasets; do
    info "Merging: $dataset"

    if python "$MERGE_SCRIPT" \
        --results-base "$results_base" \
        --dataset "$dataset" \
        --output-dir "$output_dir" 2>/dev/null; then
        has_results=true
    else
        warn "No results found for $dataset (skipped)"
    fi

    echo ""
done

if [[ "$has_results" == "false" ]]; then
    err "No benchmark results found in $results_base"
    echo ""
    echo "  Run benchmarks first:"
    echo "    scripts/benchmark/start_services.sh"
    echo "    scripts/benchmark/run_benchmark.sh"
    echo "    scripts/benchmark/generate_report.sh"
    echo ""
    exit 1
fi

ok "Reports saved to: $output_dir/"

# List generated files
echo ""
for dataset in $datasets; do
    csv_file="$output_dir/${dataset}_results.csv"
    md_file="$output_dir/${dataset}_results.md"
    if [[ -f "$csv_file" ]]; then
        dim "  $csv_file"
    fi
    if [[ -f "$md_file" ]]; then
        dim "  $md_file"
    fi
done
echo ""

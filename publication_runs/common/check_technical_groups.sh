#!/usr/bin/env bash
# Check whether per-gene cis_only meta.csv files have the same set of
# technical-group covariate combinations as the ntc_shared meta they were
# supposed to inherit codes from. If a gene's subset is missing any group
# the ntc_shared fit had, re-running set_technical_groups() on the subset
# (as run_cis_deferred.py unconditionally does) renumbers technical_group_code
# and desyncs it from the alpha_x_prefit/alpha_y_prefit tensors loaded from
# the ntc_shared fit -- i.e. the bug is live for that gene.
#
# Usage:
#   ./check_technical_groups.sh <ntc_shared_meta.csv> <output_base_dir> [covariate_cols]
#
# Example:
#   ./check_technical_groups.sh \
#     /path/to/morris_20260806_ntc_shared/meta.csv \
#     /path/to/output \
#     lane
#
# covariate_cols defaults to "lane" (comma-separated for multiple, e.g. "experiment,lane").
# It scans <output_base_dir>/*/cis_only/meta.csv for per-gene subsets.

set -euo pipefail

NTC_META="${1:?usage: $0 <ntc_shared_meta.csv> <output_base_dir> [covariate_cols]}"
OUTBASE="${2:?usage: $0 <ntc_shared_meta.csv> <output_base_dir> [covariate_cols]}"
COVCOLS="${3:-lane}"

get_col_indices() {
    local header="$1" cols="$2"
    local IFS=','
    read -ra hdr_arr <<< "$header"
    local IFS=','
    read -ra want_arr <<< "$cols"
    local idxs=()
    for w in "${want_arr[@]}"; do
        for i in "${!hdr_arr[@]}"; do
            if [[ "${hdr_arr[$i]}" == "$w" ]]; then
                idxs+=($((i+1)))
            fi
        done
    done
    echo "${idxs[@]}"
}

NTC_HEADER=$(head -n1 "$NTC_META")
IDXS=$(get_col_indices "$NTC_HEADER" "$COVCOLS")
if [[ -z "$IDXS" ]]; then
    echo "ERROR: could not find covariate columns [$COVCOLS] in $NTC_META header: $NTC_HEADER" >&2
    exit 1
fi
FIELDLIST=$(echo "$IDXS" | tr ' ' ',')

echo "Using covariate columns: $COVCOLS (fields $FIELDLIST)"
NTC_GROUPS=$(tail -n +2 "$NTC_META" | cut -d',' -f"$FIELDLIST" | sort -u)
NTC_COUNT=$(echo "$NTC_GROUPS" | wc -l)
echo "=== ntc_shared: $NTC_COUNT distinct group(s) ==="
echo "$NTC_GROUPS"
echo

echo "=== Checking each gene's cis_only/meta.csv under $OUTBASE ==="
MISMATCHES=0
CHECKED=0
for meta in "$OUTBASE"/*/cis_only/meta.csv; do
    [[ -f "$meta" ]] || continue
    CHECKED=$((CHECKED+1))
    gene_dir=$(basename "$(dirname "$(dirname "$meta")")")
    GENE_HEADER=$(head -n1 "$meta")
    GIDXS=$(get_col_indices "$GENE_HEADER" "$COVCOLS")
    if [[ -z "$GIDXS" ]]; then
        echo "[$gene_dir] SKIP - covariate columns not found in header"
        continue
    fi
    GFIELDLIST=$(echo "$GIDXS" | tr ' ' ',')
    GENE_GROUPS=$(tail -n +2 "$meta" | cut -d',' -f"$GFIELDLIST" | sort -u)
    GENE_COUNT=$(echo "$GENE_GROUPS" | wc -l)

    if [[ "$GENE_GROUPS" == "$NTC_GROUPS" ]]; then
        echo "[$gene_dir] MATCH   - $GENE_COUNT groups, identical to ntc_shared -> bug INERT for this gene"
    else
        MISMATCHES=$((MISMATCHES+1))
        MISSING=$(comm -23 <(echo "$NTC_GROUPS") <(echo "$GENE_GROUPS") || true)
        EXTRA=$(comm -13 <(echo "$NTC_GROUPS") <(echo "$GENE_GROUPS") || true)
        echo "[$gene_dir] MISMATCH - $GENE_COUNT groups (ntc_shared has $NTC_COUNT)"
        if [[ -n "$MISSING" ]]; then
            echo "    groups in ntc_shared but ABSENT from this gene's subset (triggers renumbering -> BUG LIKELY LIVE):"
            echo "$MISSING" | sed 's/^/      /'
        fi
        if [[ -n "$EXTRA" ]]; then
            echo "    groups in this subset but NOT in ntc_shared (no alpha estimate exists for these -- separate problem):"
            echo "$EXTRA" | sed 's/^/      /'
        fi
    fi
done

echo
echo "=== Summary ==="
echo "Gene subsets checked: $CHECKED"
echo "Mismatched: $MISMATCHES"
if [[ "$MISMATCHES" -gt 0 ]]; then
    echo "-> These genes' fit_cis runs most likely used MISALIGNED alpha_x_prefit/alpha_y_prefit corrections."
else
    echo "-> No mismatches found across checked genes: bug appears inert here."
fi

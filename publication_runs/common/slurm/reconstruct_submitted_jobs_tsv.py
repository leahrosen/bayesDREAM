"""
Rebuild a submitted_jobs.tsv-compatible file from `sacct`, for datasets whose
jobs were submitted by hand (not via a dataset's submit_all.sh, which writes
this file itself as it submits). list_job_status.py needs exactly this format
(columns: stage, label, jobid, script) -- this script recovers it after the
fact by matching each job's SLURM job name back to the same
stage/label/script naming convention generate_slurm.py used to create it, so
you get list_job_status.py's grouped "NEEDS ATTENTION" view without having
used submit_all.sh.

Job names are matched literally against the patterns generate_slurm.py's
`job_name=` arguments actually produce (see domingo/generate_slurm.py and
morris/generate_slurm.py) -- if a naming convention changes there, update the
patterns here too. Array jobs (subset_sweep, cis_sweep) keep sacct's own
per-task jobid (e.g. '23009372_5'), one tsv row per task, since that's what
sacct actually reports states for -- there's no per-gene name to recover for
an individual sweep-array task, so its label is just its task index.

Usage
-----
    python reconstruct_submitted_jobs_tsv.py --dataset domingo --since today \\
        --outfile domingo_submitted_jobs.tsv
    python reconstruct_submitted_jobs_tsv.py --dataset morris --since today \\
        --outfile morris_submitted_jobs.tsv
    python list_job_status.py domingo_submitted_jobs.tsv morris_submitted_jobs.tsv

--since is passed straight through to `sacct --starttime=<since>` (sacct's
own syntax, e.g. 'today', '2026-08-11', 'now-2days'). Jobs still PENDING on
an unresolved --dependency (no eligible start time assigned yet) often don't
show up in a --starttime-filtered sacct query at all -- that's fine here,
since list_job_status.py already treats PENDING as not needing attention;
they'll appear once they actually start or resolve.
"""

import argparse
import os
import re
import subprocess
import sys

import pandas as pd

# (compiled regex, stage template, label template, script template) --
# checked in order, first match wins. \g<gene>/\g<mod>/\g<idx> are regex
# group backreferences, substituted via re.Match.expand()-style '\g<name>'.
_DOMINGO_PATTERNS = [
    (r"^domingo_ntc_shared$", "ntc_shared", "ntc_shared", "01_ntc_shared.sh"),
    (r"^domingo_subset_(?P<gene>[^_]+)$", "subset", r"\g<gene>", r"01b_subset_\g<gene>.sh"),
    (r"^domingo_cis_(?P<gene>[^_]+)$", "cis", r"\g<gene>", r"02_cis_\g<gene>.sh"),
    (r"^domingo_comp_(?P<gene>[^_]+)$", "compensation", r"\g<gene>", r"03_compensation_\g<gene>.sh"),
    (r"^domingo_trans_(?P<gene>[^_]+)$", "trans", r"\g<gene>", r"04_trans_\g<gene>.sh"),
    (r"^domingo_perm_(?P<gene>[^_]+)$", "permutation", r"\g<gene>", r"05_permutation_\g<gene>.sh"),
    (r"^domingo_sim_(?P<gene>[^_]+)$", "recapitulation", r"\g<gene>", r"06_recapitulation_\g<gene>.sh"),
    (r"^domingo_modsubset_(?P<gene>[^_]+)_(?P<mod>.+)$", r"modality_subset_\g<mod>", r"\g<gene>",
     r"07a_modality_subset_\g<gene>_\g<mod>.sh"),
    (r"^domingo_modperm_(?P<gene>[^_]+)_(?P<mod>.+)$", r"modality_permutation_\g<mod>", r"\g<gene>",
     r"08_modality_permutation_\g<gene>_\g<mod>.sh"),
    (r"^domingo_modsim_(?P<gene>[^_]+)_(?P<mod>.+)$", r"modality_recapitulation_\g<mod>", r"\g<gene>",
     r"09_modality_recapitulation_\g<gene>_\g<mod>.sh"),
    (r"^domingo_mod_multinomial_packed$", "modality_multinomial_packed", "all_genes",
     "07_modality_multinomial_packed.sh"),
    (r"^domingo_mod_(?P<gene>[^_]+)_(?P<mod>.+)$", r"modality_\g<mod>", r"\g<gene>",
     r"07_modality_\g<gene>_\g<mod>.sh"),
]

_MORRIS_PATTERNS = [
    (r"^morris_subset_sweep$", "subset_sweep", "all_sweep", "01c_subset_sweep.sh"),
    (r"^morris_subset_(?P<gene>[^_]+)$", "subset", r"\g<gene>", r"01b_subset_\g<gene>.sh"),
    (r"^morris_ntc_shared$", "ntc_shared", "sweep_shared", "01_ntc_shared.sh"),
    (r"^morris_ntc_packed$", "ntc", "all_primary", "01d_ntc_packed.sh"),
    (r"^morris_cis_sweep$", "cis_sweep", "all_sweep", "07_cis_sweep.sh"),
    (r"^morris_cis_(?P<gene>[^_]+)$", "cis", r"\g<gene>", r"02_cis_\g<gene>.sh"),
    (r"^morris_comp_(?P<gene>[^_]+)$", "compensation", r"\g<gene>", r"03_compensation_\g<gene>.sh"),
    (r"^morris_trans_packed$", "trans", "all_primary", "04_trans_packed.sh"),
    (r"^morris_perm_packed$", "permutation", "all_primary", "05_permutation_packed.sh"),
    (r"^morris_sim_packed$", "recapitulation", "all_primary", "06_recapitulation_packed.sh"),
]

_PATTERNS = {"domingo": _DOMINGO_PATTERNS, "morris": _MORRIS_PATTERNS}


def _classify(job_name: str, patterns: list) -> tuple:
    """Returns (stage, label, script) for the first matching pattern, or
    None if job_name doesn't match any of them (e.g. '<dataset>_preprocess',
    which isn't a generate_slurm.py-managed stage at all)."""
    for regex, stage_tmpl, label_tmpl, script_tmpl in patterns:
        m = re.match(regex, job_name)
        if m:
            return m.expand(stage_tmpl), m.expand(label_tmpl), m.expand(script_tmpl)
    return None


def _sacct_rows(since: str) -> list:
    """[(jobid, job_name), ...] via one sacct call, -X so array jobs report
    one row per TASK (not per job step) -- e.g. '23009372_5', not a step
    id like '23009372_5.batch'."""
    user = os.environ.get("USER") or subprocess.check_output(["whoami"]).decode().strip()
    out = subprocess.run(
        ["sacct", "-u", user, "--starttime", since, "--format=JobID,JobName%60", "-X", "-n", "-P"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jobid, job_name = parts[0].strip(), parts[1].strip()
        rows.append((jobid, job_name))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(_PATTERNS))
    parser.add_argument("--since", default="today", help="sacct --starttime value (default: 'today').")
    parser.add_argument("--outfile", required=True)
    args = parser.parse_args()

    patterns = _PATTERNS[args.dataset]
    try:
        sacct_rows = _sacct_rows(args.since)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"[ERROR] sacct query failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Only real, individually-queryable job IDs -- excludes pending arrays'
    # placeholder summary rows (jobid like '23009372_[1+', not a real task),
    # which would break list_job_status.py's single batched `sacct -j ...`
    # call if included (one malformed ID in that call degrades EVERY job in
    # the tsv to 'unknown', not just the bad one).
    _real_jobid = re.compile(r"^\d+(_\d+)?$")

    tsv_rows = []
    unmatched = set()
    skipped_placeholders = 0
    for jobid, job_name in sacct_rows:
        # job_name has no per-task suffix -- sacct repeats the same name for
        # every array task, so this only trips for genuinely unrecognized names.
        classified = _classify(job_name, patterns)
        if classified is None:
            if job_name.startswith(f"{args.dataset}_") and job_name != f"{args.dataset}_preprocess":
                unmatched.add(job_name)
            continue
        if not _real_jobid.match(jobid):
            skipped_placeholders += 1
            continue
        stage, label, script = classified
        # Array tasks: fold the task index into the label so
        # list_job_status.py's printed output stays readable per-task
        # (jobid already carries it, e.g. '23009372_5', but label is shown
        # separately in that tool's output).
        if "_" in jobid:
            label = f"{label}_task{jobid.split('_', 1)[1]}"
        tsv_rows.append({"stage": stage, "label": label, "jobid": jobid, "script": script})

    if unmatched:
        print(f"[WARN] {len(unmatched)} unrecognized {args.dataset}_* job name(s) -- not written to the tsv "
              f"(add a pattern in reconstruct_submitted_jobs_tsv.py if these are real pipeline stages):",
              file=sys.stderr)
        for name in sorted(unmatched):
            print(f"  {name}", file=sys.stderr)

    if skipped_placeholders:
        print(f"[INFO] skipped {skipped_placeholders} pending-array placeholder row(s) (e.g. '<id>_[1+', not a "
              f"real per-task job ID yet) -- they'll show up here once those tasks actually start.", file=sys.stderr)

    pd.DataFrame(tsv_rows, columns=["stage", "label", "jobid", "script"]).to_csv(args.outfile, sep="\t", index=False)
    print(f"[reconstruct_submitted_jobs_tsv] wrote {len(tsv_rows)} row(s) -> {args.outfile}")


if __name__ == "__main__":
    main()

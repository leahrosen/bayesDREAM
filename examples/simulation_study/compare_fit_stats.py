"""
Compare fit_stats.json across scenarios -- e.g. baseline vs. packed GPU-sharing test.

Usage:
    python compare_fit_stats.py <baseline_dir> <packed_dir_1> [<packed_dir_2> ...]

Each <dir> is a scenario directory (the one containing fit/recovery/fit_stats.json),
e.g. $DATA/scenario_0/rep_0. The first argument is treated as the baseline; all others
are compared against it (ratio > 1 means slower than baseline).
"""

import json
import os
import sys


def load(scenario_dir):
    path = os.path.join(scenario_dir, 'fit', 'recovery', 'fit_stats.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(dirs):
    labels = [os.path.basename(os.path.dirname(d.rstrip('/'))) + '/' + os.path.basename(d.rstrip('/'))
              for d in dirs]
    stats = [load(d) for d in dirs]

    print(f"{'':20s}" + "".join(f"{lbl:>18s}" for lbl in labels))
    for s, lbl, d in zip(stats, labels, dirs):
        if s is None:
            print(f"  {lbl}: fit_stats.json not found at {d} -- job hasn't reached that step yet")
            continue
        print(f"  {lbl}: device={s.get('device_resolved')}  host={s.get('hostname')}  "
              f"job={s.get('slurm_job_id')}_{s.get('slurm_array_task_id')}")

    all_step_names = ['fit_ntc', 'fit_cis', 'fit_trans']
    print()
    for step in all_step_names:
        row_elapsed = []
        for s in stats:
            entry = (s or {}).get('steps', {}).get(step)
            row_elapsed.append(entry['elapsed_sec'] if entry else None)

        baseline = row_elapsed[0]
        cells = []
        for v in row_elapsed:
            if v is None:
                cells.append(f"{'--':>18s}")
            elif v == baseline or baseline is None:
                cells.append(f"{v:>14.1f}s   ")
            else:
                ratio = v / baseline
                cells.append(f"{v:>10.1f}s x{ratio:4.2f}")
        print(f"{step:20s}" + "".join(cells))

    print()
    for step in ['peak_rss_mb', 'peak_gpu_mb']:
        print(f"{step:20s}", end="")
        for s in stats:
            steps = (s or {}).get('steps', {})
            # report from fit_trans if present, else fit_cis, else fit_ntc (whichever furthest along)
            entry = None
            for name in ['fit_trans', 'fit_cis', 'fit_ntc']:
                if name in steps:
                    entry = steps[name]
                    break
            val = entry.get(step) if entry else None
            print(f"{(f'{val:.0f}MB' if val is not None else '--'):>18s}", end="")
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])

"""
Git provenance helpers shared by build_design_matrix.py, simulate_scenario.py, and
(eventually) run_recovery_fit.py.

Two distinct guarantees, not to be confused:
- git_provenance() answers "what commit/branch was checked out, and was it clean"
  at the moment it's called. Cheap, read-only, safe to call from every array task.
- create_stable_snapshot_tag() answers "will this commit still be reachable if the
  source branch is later deleted or force-pushed past this point." A commit hash
  alone is sufficient to reproduce code *content* (git is content-addressed), but
  it's only guaranteed to still *exist* in the repo if something keeps it reachable.
  Dangling commits with no reachable ref are eventually garbage-collected by
  `git gc`. A tag is a ref, so it pins the commit independent of any branch's fate.
"""

import os
import subprocess
from datetime import datetime, timezone


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _git(*args) -> str:
    return subprocess.check_output(
        ['git', *args], cwd=_repo_dir(), stderr=subprocess.DEVNULL,
    ).decode().strip()


def git_provenance() -> dict:
    """Commit, branch, and working-tree cleanliness at the moment of the call.

    bayesdream_git_dirty covers TRACKED files only (modified/staged/deleted) -- the
    thing that actually threatens "the commit hash doesn't fully capture what ran".
    bayesdream_untracked_count is reported separately and does NOT count as dirty:
    an untracked file only matters for reproducibility if it's actually imported/used
    by the code that runs, which git can't determine from its mere presence. Treating
    any stray untracked file (docs, leftover pre-refactor modules, logo assets, a
    personal setup script, ...) as equivalent to an uncommitted code change would make
    this check too strict to be usable on a real, lived-in clone -- confirmed in
    practice on the Berzelius clone used for this study, which accumulated several
    untracked files unrelated to anything that actually executes.
    """
    try:
        lines = [l for l in _git('status', '--porcelain').splitlines() if l]
        untracked = [l for l in lines if l.startswith('??')]
        return {
            'bayesdream_commit': _git('rev-parse', 'HEAD'),
            'bayesdream_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'bayesdream_git_dirty': len(lines) > len(untracked),
            'bayesdream_untracked_count': len(untracked),
        }
    except Exception:
        return {'bayesdream_commit': 'unknown', 'bayesdream_branch': 'unknown',
                'bayesdream_git_dirty': None, 'bayesdream_untracked_count': None}


def create_stable_snapshot_tag(prefix: str = 'sim-study', push: bool = True,
                                timestamp: str = None) -> dict:
    """Create (and by default push) an annotated tag at the current HEAD.

    Call this ONCE per design-matrix build, not per-scenario -- many concurrent
    SLURM array tasks all tagging the same commit would be redundant and racy.

    Refuses to tag a dirty working tree (the tag would only pin the last commit,
    not what's actually about to run -- misleading rather than helpful). Never
    raises: tagging is best-effort provenance, not a hard requirement for the
    design matrix to be written.

    timestamp : str, optional
        Pre-generated '%Y%m%dT%H%M%SZ' string to reuse (e.g. so the tag name and
        the run's dated output directory name share the exact same timestamp,
        not two independently-generated ones a few milliseconds apart). Generates
        its own if not given.

    Returns
    -------
    dict with 'bayesdream_tag' (str or None if tagging was skipped/failed) and
    'bayesdream_tag_pushed' (bool).
    """
    prov = git_provenance()
    if prov['bayesdream_git_dirty']:
        print("[WARNING] Tracked files have uncommitted changes -- skipping stable "
              "snapshot tag (it would only pin the last commit, not what's about to "
              "run). Commit your changes first for a durable snapshot.")
        return {'bayesdream_tag': None, 'bayesdream_tag_pushed': False}
    if prov['bayesdream_commit'] == 'unknown':
        print("[WARNING] Could not determine current commit -- skipping stable snapshot tag.")
        return {'bayesdream_tag': None, 'bayesdream_tag_pushed': False}
    if prov.get('bayesdream_untracked_count'):
        print(f"[INFO] {prov['bayesdream_untracked_count']} untracked file(s) present "
              f"(not counted as dirty -- see git_provenance() docstring). Tagging anyway.")

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag_name = f"{prefix}-{timestamp}"
    try:
        _git('tag', '-a', tag_name, '-m', f"Stable snapshot for simulation study run ({tag_name})")
        print(f"[INFO] Created git tag '{tag_name}' at {prov['bayesdream_commit'][:12]} "
              f"(branch {prov['bayesdream_branch']})")
    except Exception as e:
        print(f"[WARNING] Could not create git tag: {e}")
        return {'bayesdream_tag': None, 'bayesdream_tag_pushed': False}

    pushed = False
    if push:
        try:
            _git('push', 'origin', tag_name)
            pushed = True
            print(f"[INFO] Pushed tag '{tag_name}' to origin")
        except Exception as e:
            print(f"[WARNING] Created tag '{tag_name}' locally but could not push it "
                  f"to origin ({e}) -- it will not survive this local clone being "
                  f"deleted. Push it manually later: git push origin {tag_name}")

    return {'bayesdream_tag': tag_name, 'bayesdream_tag_pushed': pushed}

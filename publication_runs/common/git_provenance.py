"""
Git provenance helpers shared by every publication_runs/ dataset.

Adapted from examples/simulation_study/_git_provenance.py (kept in that
directory too, scoped to the simulation study specifically) -- this copy is
dataset-agnostic. Two distinct guarantees, not to be confused:

- git_provenance() answers "what commit/branch was checked out, and was it
  clean" at the moment it's called. Cheap, read-only, safe to call from every
  job.
- create_stable_snapshot_tag() answers "will this commit still be reachable
  if the source branch is later deleted or force-pushed past this point." A
  commit hash alone reproduces code *content* (git is content-addressed), but
  it's only guaranteed to still *exist* if something keeps it reachable --
  dangling commits with no reachable ref are eventually `git gc`'d. A tag is
  a ref, so it pins the commit independent of any branch's fate.

Call create_stable_snapshot_tag() ONCE per dataset per generate_slurm.py
invocation, not once per job -- many jobs tagging the same commit is
redundant and racy. git_provenance() is fine to call from every job.
"""

import json
import os
import subprocess
from datetime import datetime, timezone


def _repo_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _git(*args) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_repo_dir(), stderr=subprocess.DEVNULL,
    ).decode().strip()


def git_provenance() -> dict:
    """Commit, branch, and working-tree cleanliness at the moment of the call.

    bayesdream_git_dirty covers TRACKED files only (modified/staged/deleted).
    bayesdream_untracked_count is reported separately and does NOT count as
    dirty -- an untracked file only matters for reproducibility if it's
    actually imported/used by the code that runs, which git can't determine
    from its mere presence.
    """
    try:
        lines = [l for l in _git("status", "--porcelain").splitlines() if l]
        untracked = [l for l in lines if l.startswith("??")]
        return {
            "bayesdream_commit": _git("rev-parse", "HEAD"),
            "bayesdream_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "bayesdream_git_dirty": len(lines) > len(untracked),
            "bayesdream_untracked_count": len(untracked),
        }
    except Exception:
        return {
            "bayesdream_commit": "unknown",
            "bayesdream_branch": "unknown",
            "bayesdream_git_dirty": None,
            "bayesdream_untracked_count": None,
        }


def create_stable_snapshot_tag(prefix: str, push: bool = True) -> dict:
    """Create (and by default push) an annotated tag at the current HEAD.

    Refuses to tag a dirty working tree (the tag would only pin the last
    commit, not what's about to run). Never raises -- tagging is best-effort
    provenance, not a hard requirement for job generation to proceed.

    Parameters
    ----------
    prefix : str
        Tag name prefix, e.g. ``'domingo-run'`` or ``'morris-run'``. The full
        tag is ``'{prefix}-{UTC timestamp}'``.

    Returns
    -------
    dict with 'bayesdream_tag' (str or None) and 'bayesdream_tag_pushed' (bool).
    """
    prov = git_provenance()
    if prov["bayesdream_git_dirty"]:
        print(
            "[WARNING] Tracked files have uncommitted changes -- skipping "
            "stable snapshot tag (it would only pin the last commit, not "
            "what's about to run). Commit your changes first."
        )
        return {"bayesdream_tag": None, "bayesdream_tag_pushed": False}
    if prov["bayesdream_commit"] == "unknown":
        print("[WARNING] Could not determine current commit -- skipping tag.")
        return {"bayesdream_tag": None, "bayesdream_tag_pushed": False}
    if prov.get("bayesdream_untracked_count"):
        print(
            f"[INFO] {prov['bayesdream_untracked_count']} untracked file(s) "
            f"present (not counted as dirty). Tagging anyway."
        )

    tag_name = f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    try:
        _git("tag", "-a", tag_name, "-m", f"Stable snapshot for {prefix} ({tag_name})")
        print(f"[INFO] Created git tag '{tag_name}' at {prov['bayesdream_commit'][:12]} "
              f"(branch {prov['bayesdream_branch']})")
    except Exception as e:
        print(f"[WARNING] Could not create git tag: {e}")
        return {"bayesdream_tag": None, "bayesdream_tag_pushed": False}

    pushed = False
    if push:
        try:
            _git("push", "origin", tag_name)
            pushed = True
            print(f"[INFO] Pushed tag '{tag_name}' to origin")
        except Exception as e:
            print(f"[WARNING] Created tag '{tag_name}' locally but could not "
                  f"push it ({e}) -- push manually: git push origin {tag_name}")

    return {"bayesdream_tag": tag_name, "bayesdream_tag_pushed": pushed}


def save_provenance_json(path: str, extra: dict = None) -> dict:
    """Write git_provenance() (plus any extra fields) to a JSON file.

    Called by each common/run_*.py script next to whatever output it
    produces, so every CSV/pt file on disk has a matching provenance.json
    recording the commit that generated it.
    """
    prov = git_provenance()
    prov["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if extra:
        prov.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)
    return prov


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print or tag current git provenance.")
    parser.add_argument("--tag-prefix", default=None, help="If set, create a stable snapshot tag with this prefix.")
    parser.add_argument("--no-push", action="store_true", help="Don't push the created tag.")
    args = parser.parse_args()

    if args.tag_prefix:
        print(json.dumps(create_stable_snapshot_tag(args.tag_prefix, push=not args.no_push), indent=2))
    else:
        print(json.dumps(git_provenance(), indent=2))

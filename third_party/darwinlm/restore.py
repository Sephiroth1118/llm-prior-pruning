"""Restore the level-database builders, which are modified versions of files of the DarwinLM repository.

DarwinLM (https://github.com/IST-DASLab/DarwinLM) declares no license, so this repository does not redistribute its files.
The level databases of the paper were built with modified versions of its pruner (structured second-order pruning) and of
its build script. patches.json records, for each modified file, the original DarwinLM file, the line ranges we replaced and
the lines we wrote. Lines that we took over from DarwinLM, also from other files of it or with other indentation, are
stored as references to the DarwinLM files, not as text. This script reads the originals from a DarwinLM clone, checks
their SHA-256, applies the changes and checks the SHA-256 of the result, so the restored files are exactly the ones we
used.

Usage:
  git clone https://github.com/IST-DASLab/DarwinLM.git && git -C DarwinLM checkout 3596da3973224da83255363a4fbe6757d569dc8c
  python third_party/darwinlm/restore.py DarwinLM

Only the level-database builders need the restored files (experiments/struct_prune.py, experiments/struct_prune_legacy.py,
evoprune/MoE/cli_build_db.py and the modules they import). The searches, the evaluation and the analysis scripts do not.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                        # repository root


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def apply(origin_lines, ops, sources):
    """ops: [i1, i2, new] in increasing order; origin_lines[i1:i2] is replaced by new. An item of new is one of our lines
    (a string), the index of a line of the original file (an integer), a line of a DarwinLM file ([source, index]) or such
    a line with its indentation set to n spaces ([source, index, n])."""
    def line(x):
        if isinstance(x, str):
            return x
        if isinstance(x, int):
            return origin_lines[x]
        text = sources[x[0]][x[1]]
        return text if len(x) == 2 else " " * x[2] + text.lstrip(" ")

    out, pos = [], 0
    for i1, i2, new in ops:
        out.extend(origin_lines[pos:i1])
        out.extend(line(x) for x in new)
        pos = i2
    out.extend(origin_lines[pos:])
    return out


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    darwinlm = Path(argv[0])
    spec = json.loads((HERE / "patches.json").read_text())
    sources, lines = [], {}
    for src in spec["sources"]:
        data = (darwinlm / src["path"]).read_bytes()
        if sha256(data) != src["sha256"]:
            sys.exit(f"{src['path']}: unexpected content; check out DarwinLM commit {spec['darwinlm_commit']}")
        lines[src["path"]] = data.decode("utf-8").splitlines(keepends=True)
        sources.append(lines[src["path"]])
    for f in spec["files"]:
        result = "".join(apply(lines[f["origin"]], f["ops"], sources)).encode("utf-8")
        if sha256(result) != f["target_sha256"]:
            sys.exit(f"{f['target']}: restored file does not match the recorded hash")
        target = ROOT / f["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result)
        print("restored", f["target"])


if __name__ == "__main__":
    main(sys.argv[1:])

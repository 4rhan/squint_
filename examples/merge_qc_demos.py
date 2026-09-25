"""Merge QC demo files (e.g. from parallel collect_qc_demos runs with different --start-seed) into one file
that train_squint_qc.py can load, renumbering the episodes traj_0, traj_1, ...

    python -m examples.merge_qc_demos demos/qc/place3.h5 /tmp/parts/part_*.h5
"""
import json
import sys

import h5py


def main(out, parts):
    n = 0
    with h5py.File(out, "w") as fo:
        for i, p in enumerate(parts):
            with h5py.File(p, "r") as fi:
                if i == 0:
                    meta = json.loads(fi.attrs["meta"])
                    meta["merged_from"] = len(parts)
                    fo.attrs["meta"] = json.dumps(meta)
                for k in sorted((k for k in fi if k.startswith("traj_")), key=lambda k: int(k.split("_")[1])):
                    fi.copy(k, fo, name=f"traj_{n}")
                    n += 1
    print(f"merged {n} demos from {len(parts)} files into {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

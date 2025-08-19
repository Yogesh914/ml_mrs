from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_top10_ranks(file_path: Path) -> Dict[str, int]:
    """
    Parse a single run output file and return a mapping of feature -> rank (1..10)
    based on the order under the "Top 10 selected features:" section.
    """
    text = file_path.read_text(errors="ignore")
    lines = text.splitlines()

    header_index = None
    for idx, line in enumerate(lines):
        if line.strip().lower().startswith("top 10 selected features"):
            header_index = idx
            break

    if header_index is None:
        return {}

    # Collect the next 10 feature lines following the header. Be tolerant of extra lines.
    feature_to_rank: Dict[str, int] = {}
    rank_counter = 0
    for line in lines[header_index + 1 : header_index + 1 + 30]:  # scan up to 30 lines to be safe
        if ":" not in line:
            continue
        left, _right = line.split(":", 1)
        feature_name = left.strip()
        if not feature_name:
            continue
        # Only count distinct features once per file
        if feature_name in feature_to_rank:
            continue
        rank_counter += 1
        feature_to_rank[feature_name] = rank_counter
        if rank_counter == 10:
            break

    return feature_to_rank


def compute_mean_ranks(base_dir: Path, output_path: Path) -> None:
    run_files: List[Path] = sorted(base_dir.glob("run_*_output.txt"))
    if not run_files:
        raise FileNotFoundError(f"No run_*_output.txt files found in {base_dir}")

    ranks_by_feature: Dict[str, List[int]] = defaultdict(list)

    for fp in run_files:
        feature_ranks = parse_top10_ranks(fp)
        for feature_name, rank in feature_ranks.items():
            ranks_by_feature[feature_name].append(rank)

    # Prepare rows: feature, mean_rank, appearances
    rows = []
    for feature_name, ranks in ranks_by_feature.items():
        if not ranks:
            continue
        mean_rank = sum(ranks) / float(len(ranks))
        appearances = len(ranks)
        rows.append((feature_name, mean_rank, appearances))

    # Sort by mean_rank ascending, then by appearances desc, then by feature name asc
    rows.sort(key=lambda x: (x[1], -x[2], x[0]))

    output_lines = ["feature\tmean_rank\tappearances"]
    for feature_name, mean_rank, appearances in rows:
        output_lines.append(f"{feature_name}\t{mean_rank:.4f}\t{appearances}")

    output_path.write_text("\n".join(output_lines))


def main() -> None:
    if len(sys.argv) >= 2:
        base_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        base_dir = (
            Path(__file__).parent / "experiment_results"
        ).resolve()

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2]).expanduser().resolve()
    else:
        output_path = (base_dir / "feature_mean_rank.txt").resolve()

    compute_mean_ranks(base_dir, output_path)
    print(f"Wrote mean ranks for features to: {output_path}")


if __name__ == "__main__":
    main()



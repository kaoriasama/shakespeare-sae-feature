#!/usr/bin/env python3
"""Recount SAE feature firing across the 38 Shakespeare works."""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_CSV = ROOT / "chunks" / "shakespeare_chunks_v2.csv"
REPORTS_DIR = ROOT / "reports"

LAYERS = (
    ("g2", 4, 12143, 4021, 220),
    ("g2", 11, 10671, 5138, 575),
    ("g2", 22, 9688, 6462, 234),
    ("g3", 9, 9103, 5922, 1359),
    ("g3", 17, 7217, 5604, 3563),
    ("g3", 29, 8351, 7284, 749),
)


def load_chunk_metadata():
    with CHUNKS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "folger_abbr", "genre"}
        assert reader.fieldnames is not None
        assert required.issubset(reader.fieldnames), (
            f"missing CSV columns: {sorted(required - set(reader.fieldnames))}"
        )
        rows = list(reader)

    chunk_to_work = {row["chunk_id"]: row["folger_abbr"] for row in rows}
    assert len(chunk_to_work) == len(rows), "duplicate chunk_id in chunks CSV"

    works = sorted({row["folger_abbr"] for row in rows})
    assert len(works) == 38, f"expected 38 works, found {len(works)}"

    genre_works = {
        genre: {
            row["folger_abbr"]
            for row in rows
            if row["genre"] == genre and row["folger_abbr"] not in {"Per", "TNK"}
        }
        for genre in ("comedy", "history", "tragedy")
    }
    expected_sizes = {"comedy": 14, "history": 10, "tragedy": 12}
    actual_sizes = {genre: len(items) for genre, items in genre_works.items()}
    assert actual_sizes == expected_sizes, (
        f"genre work counts differ: expected {expected_sizes}, found {actual_sizes}"
    )
    return chunk_to_work, works, genre_works


def recount_layer(model, layer, chunk_to_work, works, retain_patterns=False):
    npz_path = ROOT / "activations_v2" / f"{model}_L{layer}_firing.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        chunk_ids = data["chunk_ids"]
        firing_rate = data["firing_rate"]

        assert chunk_ids.shape == (4600,), (
            f"{model} L{layer}: chunk_ids shape is {chunk_ids.shape}, expected (4600,)"
        )
        assert firing_rate.shape == (4600, 16384), (
            f"{model} L{layer}: firing_rate shape is {firing_rate.shape}, "
            "expected (4600, 16384)"
        )

        npz_ids = [str(item) for item in chunk_ids.tolist()]
        csv_id_set = set(chunk_to_work)
        npz_id_set = set(npz_ids)
        if csv_id_set != npz_id_set:
            print(f"{model} L{layer}: chunk_id set mismatch", file=sys.stderr)
            print(
                f"  only in CSV ({len(csv_id_set - npz_id_set)}): "
                f"{sorted(csv_id_set - npz_id_set)[:20]}",
                file=sys.stderr,
            )
            print(
                f"  only in NPZ ({len(npz_id_set - csv_id_set)}): "
                f"{sorted(npz_id_set - csv_id_set)[:20]}",
                file=sys.stderr,
            )
        assert csv_id_set == npz_id_set, f"{model} L{layer}: chunk_id sets differ"
        assert len(npz_id_set) == len(npz_ids), (
            f"{model} L{layer}: duplicate chunk_id in NPZ"
        )

        work_rows = {work: [] for work in works}
        for row_index, chunk_id in enumerate(npz_ids):
            work_rows[chunk_to_work[chunk_id]].append(row_index)

        n_works_fired = np.zeros(firing_rate.shape[1], dtype=np.uint8)
        pattern_masks = (
            np.zeros(firing_rate.shape[1], dtype=np.uint64)
            if retain_patterns
            else None
        )
        for work_index, work in enumerate(works):
            fired = np.any(firing_rate[work_rows[work]] > 0, axis=0)
            n_works_fired += fired
            if pattern_masks is not None:
                pattern_masks[fired] |= np.uint64(1) << np.uint64(work_index)

    return n_works_fired, pattern_masks


def category_counts(n_works_fired):
    return (
        int(np.count_nonzero(n_works_fired == 38)),
        int(np.count_nonzero((n_works_fired >= 1) & (n_works_fired <= 37))),
        int(np.count_nonzero(n_works_fired == 0)),
    )


def mask_to_works(mask, works):
    value = int(mask)
    return tuple(work for index, work in enumerate(works) if value & (1 << index))


def build_partial_patterns(n_works_fired, masks, works):
    partial_masks = masks[(n_works_fired >= 1) & (n_works_fired <= 37)]
    counts = Counter(int(mask) for mask in partial_masks.tolist())
    patterns = [
        (n_features, mask_to_works(mask, works))
        for mask, n_features in counts.items()
    ]
    patterns.sort(key=lambda item: (-item[0], len(item[1]), " ".join(item[1])))
    return patterns


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    chunk_to_work, works, genre_works = load_chunk_metadata()
    layer_counts = {}
    layer_firing_counts = {}
    g3_l17_masks = None

    for model, layer, expected_all, expected_partial, expected_absent in LAYERS:
        retain_patterns = model == "g3" and layer == 17
        n_works_fired, masks = recount_layer(
            model, layer, chunk_to_work, works, retain_patterns=retain_patterns
        )
        actual = category_counts(n_works_fired)
        expected = (expected_all, expected_partial, expected_absent)
        if actual != expected:
            print(
                f"{model} L{layer}: classification mismatch: "
                f"expected all={expected_all}, partial={expected_partial}, "
                f"absent={expected_absent}; found all={actual[0]}, "
                f"partial={actual[1]}, absent={actual[2]}",
                file=sys.stderr,
            )
        assert actual == expected, f"{model} L{layer}: acceptance counts differ"
        layer_counts[(model, layer)] = actual
        layer_firing_counts[(model, layer)] = n_works_fired
        if retain_patterns:
            g3_l17_masks = masks
        print(
            f"{model} L{layer}: all={actual[0]}, partial={actual[1]}, "
            f"absent={actual[2]}"
        )

    assert g3_l17_masks is not None
    g3_l17_counts = layer_firing_counts[("g3", 17)]
    patterns = build_partial_patterns(g3_l17_counts, g3_l17_masks, works)
    partial_total = layer_counts[("g3", 17)][1]
    assert sum(n_features for n_features, _ in patterns) == partial_total

    feature_rows = []
    for model, layer, _, _, _ in LAYERS:
        for feature, n_works in enumerate(layer_firing_counts[(model, layer)].tolist()):
            category = "absent" if n_works == 0 else "all" if n_works == 38 else "partial"
            feature_rows.append((model, layer, feature, n_works, category))
    assert len(feature_rows) == 98304

    pattern_rows = [
        (rank, n_features, len(pattern_works), " ".join(pattern_works))
        for rank, (n_features, pattern_works) in enumerate(patterns, start=1)
    ]
    summary_rows = [
        (model, layer, *layer_counts[(model, layer)])
        for model, layer, _, _, _ in LAYERS
    ]

    write_csv(
        REPORTS_DIR / "feature_work_classification.csv",
        ("model", "layer", "feature", "n_works_fired", "category"),
        feature_rows,
    )
    write_csv(
        REPORTS_DIR / "g3_L17_partial_patterns.csv",
        ("rank", "n_features", "n_works", "works"),
        pattern_rows,
    )
    write_csv(
        REPORTS_DIR / "summary_table1_check.csv",
        ("model", "layer", "n_all", "n_partial", "n_absent"),
        summary_rows,
    )

    print(f"g3 L17 partial features: {partial_total}")
    print(f"g3 L17 distinct partial patterns: {len(patterns)}")
    for threshold in (2, 5, 10):
        shared = sum(n_features >= threshold for n_features, _ in patterns)
        print(f"g3 L17 patterns shared by >= {threshold} features: {shared}")

    print("g3 L17 top 20 partial patterns:")
    for rank, (n_features, pattern_works) in enumerate(patterns[:20], start=1):
        print(f"  {rank}: n_features={n_features}, works={' '.join(pattern_works)}")

    pattern_ranks = {pattern_works: rank for rank, (_, pattern_works) in enumerate(patterns, 1)}
    print("genre set pattern check:")
    for genre in ("comedy", "history", "tragedy"):
        genre_tuple = tuple(sorted(genre_works[genre]))
        rank = pattern_ranks.get(genre_tuple)
        status = f"present at rank {rank}" if rank is not None else "not present"
        print(
            f"  {genre} ({len(genre_tuple)} works): {status}; "
            f"works={' '.join(genre_tuple)}"
        )


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

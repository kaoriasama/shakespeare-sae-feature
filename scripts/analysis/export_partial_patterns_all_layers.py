#!/usr/bin/env python3
"""Export partial work-firing patterns for all six SAE layers."""

import csv
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_CSV = ROOT / "chunks" / "shakespeare_chunks_v2.csv"
REPORTS_DIR = ROOT / "reports"
EXISTING_L17_CSV = REPORTS_DIR / "g3_L17_partial_patterns.csv"

LAYERS = (
    ("g2", 4, 4021),
    ("g2", 11, 5138),
    ("g2", 22, 6462),
    ("g3", 9, 5922),
    ("g3", 17, 5604),
    ("g3", 29, 7284),
)
PATTERN_HEADER = ("rank", "n_features", "n_works", "works")
SUMMARY_HEADER = (
    "model",
    "layer",
    "n_partial",
    "n_patterns",
    "n_shared_ge2",
    "n_shared_ge5",
    "n_shared_ge10",
    "n_singlework_patterns",
    "top_single_work",
    "top_single_n",
    "n_complement37_patterns",
    "top_complement_missing",
    "top_complement_n",
    "n_shared_mid_4_31",
    "genre_set_found",
)


def load_metadata():
    with CHUNKS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "folger_abbr", "genre"}
        assert reader.fieldnames is not None, "chunks CSV has no header"
        assert required.issubset(reader.fieldnames), (
            f"missing chunks CSV columns: {sorted(required - set(reader.fieldnames))}"
        )
        rows = list(reader)

    assert len(rows) == 4600, f"expected 4600 chunk rows, found {len(rows)}"
    chunk_to_work = {row["chunk_id"]: row["folger_abbr"] for row in rows}
    assert len(chunk_to_work) == len(rows), "duplicate chunk_id in chunks CSV"
    works = tuple(sorted({row["folger_abbr"] for row in rows}))
    assert len(works) == 38, f"expected 38 works, found {len(works)}"

    genre_sets = {}
    expected_sizes = {"comedy": 14, "history": 10, "tragedy": 12}
    for genre, expected_size in expected_sizes.items():
        genre_works = tuple(
            sorted(
                {
                    row["folger_abbr"]
                    for row in rows
                    if row["genre"] == genre
                    and row["folger_abbr"] not in {"Per", "TNK"}
                }
            )
        )
        assert len(genre_works) == expected_size, (
            f"{genre}: expected {expected_size} works, found {len(genre_works)}"
        )
        genre_sets[genre] = genre_works
    return chunk_to_work, works, genre_sets


def mask_to_works(mask, works):
    value = int(mask)
    return tuple(work for index, work in enumerate(works) if value & (1 << index))


def calculate_patterns(model, layer, chunk_to_work, works):
    npz_path = ROOT / "activations_v2" / f"{model}_L{layer}_firing.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        assert {"chunk_ids", "firing_rate"}.issubset(data.files), (
            f"{model} L{layer}: required arrays are missing"
        )
        chunk_ids = data["chunk_ids"]
        firing_rate = data["firing_rate"]
        assert chunk_ids.shape == (4600,), (
            f"{model} L{layer}: chunk_ids shape {chunk_ids.shape}"
        )
        assert firing_rate.shape == (4600, 16384), (
            f"{model} L{layer}: firing_rate shape {firing_rate.shape}"
        )
        assert firing_rate.dtype == np.float32, (
            f"{model} L{layer}: firing_rate dtype {firing_rate.dtype}"
        )

        npz_ids = tuple(str(value) for value in chunk_ids.tolist())
        assert len(set(npz_ids)) == len(npz_ids), (
            f"{model} L{layer}: duplicate chunk_id in NPZ"
        )
        assert set(npz_ids) == set(chunk_to_work), (
            f"{model} L{layer}: NPZ/CSV chunk_id sets differ"
        )

        work_rows = {work: [] for work in works}
        for row_index, chunk_id in enumerate(npz_ids):
            work_rows[chunk_to_work[chunk_id]].append(row_index)

        n_works_fired = np.zeros(16384, dtype=np.uint8)
        masks = np.zeros(16384, dtype=np.uint64)
        for work_index, work in enumerate(works):
            fired = np.any(firing_rate[work_rows[work]] > 0, axis=0)
            n_works_fired += fired
            masks[fired] |= np.uint64(1) << np.uint64(work_index)

    partial = (n_works_fired >= 1) & (n_works_fired <= 37)
    counts = Counter(int(mask) for mask in masks[partial].tolist())
    patterns = [
        (n_features, mask_to_works(mask, works))
        for mask, n_features in counts.items()
    ]
    patterns.sort(key=lambda item: (-item[0], len(item[1]), " ".join(item[1])))
    n_partial = int(np.count_nonzero(partial))
    assert sum(n_features for n_features, _ in patterns) == n_partial
    return n_partial, patterns


def pattern_rows(patterns):
    return [
        (rank, n_features, len(pattern_works), " ".join(pattern_works))
        for rank, (n_features, pattern_works) in enumerate(patterns, start=1)
    ]


def csv_bytes(header, rows):
    text_buffer = io.StringIO(newline="")
    writer = csv.writer(text_buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + text_buffer.getvalue().encode("utf-8")


def build_summary(model, layer, n_partial, patterns, works, genre_sets):
    singlework = [item for item in patterns if len(item[1]) == 1]
    complement37 = [item for item in patterns if len(item[1]) == 37]
    assert singlework, f"{model} L{layer}: no single-work pattern"
    assert complement37, f"{model} L{layer}: no 37-work pattern"

    top_single_n, top_single_works = singlework[0]
    top_complement_n, top_complement_works = complement37[0]
    missing = sorted(set(works) - set(top_complement_works))
    assert len(missing) == 1

    pattern_work_sets = {pattern_works for _, pattern_works in patterns}
    genres_found = [
        genre
        for genre in ("comedy", "history", "tragedy")
        if genre_sets[genre] in pattern_work_sets
    ]

    return (
        model,
        layer,
        n_partial,
        len(patterns),
        sum(n_features >= 2 for n_features, _ in patterns),
        sum(n_features >= 5 for n_features, _ in patterns),
        sum(n_features >= 10 for n_features, _ in patterns),
        len(singlework),
        top_single_works[0],
        top_single_n,
        len(complement37),
        missing[0],
        top_complement_n,
        sum(
            n_features >= 2 and 4 <= len(pattern_works) <= 31
            for n_features, pattern_works in patterns
        ),
        ";".join(genres_found) if genres_found else "none",
    )


def print_table(header, rows):
    widths = [len(str(value)) for value in header]
    for row in rows:
        widths = [
            max(width, len(str(value)))
            for width, value in zip(widths, row)
        ]
    print("  ".join(str(value).ljust(width) for value, width in zip(header, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)))


def main():
    chunk_to_work, works, genre_sets = load_metadata()
    all_patterns = {}
    summary_rows = []

    for model, layer, expected_partial in LAYERS:
        n_partial, patterns = calculate_patterns(
            model, layer, chunk_to_work, works
        )
        assert n_partial == expected_partial, (
            f"{model} L{layer}: expected {expected_partial} partial features, "
            f"found {n_partial}"
        )
        all_patterns[(model, layer)] = patterns
        summary_rows.append(
            build_summary(
                model, layer, n_partial, patterns, works, genre_sets
            )
        )

    l17_patterns = all_patterns[("g3", 17)]
    l17_rows = pattern_rows(l17_patterns)
    assert len(l17_patterns) == 4353, (
        f"g3 L17: expected 4353 patterns, found {len(l17_patterns)}"
    )
    assert sum(n_features >= 2 for n_features, _ in l17_patterns) == 224, (
        "g3 L17: expected 224 patterns shared by at least two features"
    )
    assert l17_rows[0] == (1, 50, 1, "Ham"), (
        f"g3 L17: unexpected first data row {l17_rows[0]}"
    )
    expected_l17_bytes = csv_bytes(PATTERN_HEADER, l17_rows)
    assert expected_l17_bytes == EXISTING_L17_CSV.read_bytes(), (
        "g3 L17: in-memory result differs from existing CSV"
    )

    for model, layer, _ in LAYERS:
        if (model, layer) == ("g3", 17):
            continue
        output_path = REPORTS_DIR / f"{model}_L{layer}_partial_patterns.csv"
        output_path.write_bytes(
            csv_bytes(PATTERN_HEADER, pattern_rows(all_patterns[(model, layer)]))
        )
    (REPORTS_DIR / "partial_patterns_summary.csv").write_bytes(
        csv_bytes(SUMMARY_HEADER, summary_rows)
    )

    print("SUMMARY")
    print_table(SUMMARY_HEADER, summary_rows)
    for model, layer, _ in LAYERS:
        print(f"\n{model} L{layer} TOP 10")
        print_table(PATTERN_HEADER, pattern_rows(all_patterns[(model, layer)][:10]))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

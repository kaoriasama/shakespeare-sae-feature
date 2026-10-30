"""Verse index per work (B): how far each of the works leans to verse or prose.

Per chunk, chunks/shakespeare_chunks_v2.csv carries
    verse_ratio = n_verse_lines / (n_verse_lines + n_prose_paragraphs)
(tei_chunk_parser_v2.py).  Verse is counted in lines and prose in paragraphs,
so the ratio mixes two units: it is a verse/prose leaning, not a share of the
text.  The ratio is blank when n_verse_lines + n_prose_paragraphs == 0
(e.g. stage-direction-only chunks).

Per work this script reports
  n_chunks_total            all chunks of the work
  n_chunks_with_ratio       chunks whose verse_ratio is not blank
  n_chars_with_ratio        sum of n_chars over those chunks
  verse_index_weighted      sum(verse_ratio * n_chars) / sum(n_chars) over chunks with a ratio
  prose_only_share          chunks with verse_ratio == 0.0 / n_chunks_with_ratio
  verse_only_share          chunks with verse_ratio == 1.0 / n_chunks_with_ratio
  n_verse_lines_total       sum of n_verse_lines over all chunks (blank chunks add 0)
  n_prose_paragraphs_total  sum of n_prose_paragraphs over all chunks (blank chunks add 0)
  verse_index_pooled        n_verse_lines_total / (n_verse_lines_total + n_prose_paragraphs_total)
                            (shown to contrast with the chunk-weighted mean)
Rows are sorted by verse_index_weighted ascending (prose-leaning works first;
ties broken by the work abbreviation).

Read-only input
  chunks/shakespeare_chunks_v2.csv   folger_abbr, n_chars, n_verse_lines, n_prose_paragraphs, verse_ratio

Output
  reports/verse_index_by_work.csv    two leading '#' comment lines, then header and one row per work

Only the standard library is used.  No random numbers.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

CSV_HEADER = (
    "work", "n_chunks_total", "n_chunks_with_ratio", "n_chars_with_ratio",
    "verse_index_weighted", "prose_only_share", "verse_only_share",
    "n_verse_lines_total", "n_prose_paragraphs_total", "verse_index_pooled",
)

COMMENT_LINES = (
    "# verse_ratio (per chunk, from tei_chunk_parser_v2.py) = n_verse_lines / (n_verse_lines + n_prose_paragraphs);"
    " verse is counted in lines and prose in paragraphs, so this is a mixed-unit verse/prose leaning, not a share of the text."
    " verse_index_weighted = sum(verse_ratio * n_chars) / sum(n_chars); verse_index_pooled = n_verse_lines_total / (n_verse_lines_total + n_prose_paragraphs_total);"
    " prose_only_share / verse_only_share = fraction of chunks with verse_ratio == 0.0 / == 1.0 among chunks with a ratio.",
    "# Chunks with a blank verse_ratio (n_verse_lines + n_prose_paragraphs == 0) are excluded from n_chunks_with_ratio, n_chars_with_ratio,"
    " verse_index_weighted, prose_only_share and verse_only_share; they are counted in n_chunks_total and contribute 0 to n_verse_lines_total and n_prose_paragraphs_total.",
)

RECALC_TOLERANCE = 1e-9


def load_chunks(csv_path):
    """Rows (file order) with the columns needed, types converted; verse_ratio
    stays None when blank.  The stored verse_ratio is checked against its
    definition for every row."""
    csv.field_size_limit(10**9)
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        required = {"chunk_id", "folger_abbr", "n_chars", "n_verse_lines", "n_prose_paragraphs", "verse_ratio"}
        missing = required - set(reader.fieldnames)
        assert not missing, f"chunks CSV lacks columns: {sorted(missing)}"
        for row in reader:
            n_verse = int(row["n_verse_lines"])
            n_prose = int(row["n_prose_paragraphs"])
            text = row["verse_ratio"].strip()
            denominator = n_verse + n_prose
            if text == "":
                assert denominator == 0, f"{row['chunk_id']}: blank verse_ratio but n_verse_lines + n_prose_paragraphs = {denominator}"
                ratio = None
            else:
                assert denominator > 0, f"{row['chunk_id']}: verse_ratio given but n_verse_lines + n_prose_paragraphs = 0"
                ratio = float(text)
                assert abs(ratio - n_verse / denominator) <= RECALC_TOLERANCE, (
                    f"{row['chunk_id']}: verse_ratio {ratio} != n_verse_lines/(n_verse_lines+n_prose_paragraphs) = {n_verse / denominator}"
                )
            rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "work": row["folger_abbr"],
                    "n_chars": int(row["n_chars"]),
                    "n_verse_lines": n_verse,
                    "n_prose_paragraphs": n_prose,
                    "verse_ratio": ratio,
                }
            )
    ids = [row["chunk_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk_id in chunks CSV"
    return rows


def share(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def fmt(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"


def summarize(rows):
    """Per-work statistics, as a dict keyed by work (accumulated in file order)."""
    stats = {}
    for row in rows:
        s = stats.setdefault(
            row["work"],
            {
                "n_chunks_total": 0, "n_chunks_with_ratio": 0, "n_chars_with_ratio": 0,
                "weighted_sum": 0.0, "n_prose_only": 0, "n_verse_only": 0,
                "n_verse_lines_total": 0, "n_prose_paragraphs_total": 0,
            },
        )
        s["n_chunks_total"] += 1
        s["n_verse_lines_total"] += row["n_verse_lines"]
        s["n_prose_paragraphs_total"] += row["n_prose_paragraphs"]
        ratio = row["verse_ratio"]
        if ratio is None:
            continue
        s["n_chunks_with_ratio"] += 1
        s["n_chars_with_ratio"] += row["n_chars"]
        s["weighted_sum"] += ratio * row["n_chars"]
        if ratio == 0.0:
            s["n_prose_only"] += 1
        elif ratio == 1.0:
            s["n_verse_only"] += 1
    result = []
    for work, s in stats.items():
        pooled_denominator = s["n_verse_lines_total"] + s["n_prose_paragraphs_total"]
        result.append(
            {
                "work": work,
                "n_chunks_total": s["n_chunks_total"],
                "n_chunks_with_ratio": s["n_chunks_with_ratio"],
                "n_chars_with_ratio": s["n_chars_with_ratio"],
                "verse_index_weighted": share(s["weighted_sum"], s["n_chars_with_ratio"]),
                "prose_only_share": share(s["n_prose_only"], s["n_chunks_with_ratio"]),
                "verse_only_share": share(s["n_verse_only"], s["n_chunks_with_ratio"]),
                "n_verse_lines_total": s["n_verse_lines_total"],
                "n_prose_paragraphs_total": s["n_prose_paragraphs_total"],
                "verse_index_pooled": share(s["n_verse_lines_total"], pooled_denominator),
            }
        )
    # ascending verse index (prose-leaning first); undefined values sort last; ties by work
    result.sort(key=lambda r: (r["verse_index_weighted"] is None, r["verse_index_weighted"] or 0.0, r["work"]))
    return result


def to_csv_row(r):
    return (
        r["work"], r["n_chunks_total"], r["n_chunks_with_ratio"], r["n_chars_with_ratio"],
        fmt(r["verse_index_weighted"]), fmt(r["prose_only_share"]), fmt(r["verse_only_share"]),
        r["n_verse_lines_total"], r["n_prose_paragraphs_total"], fmt(r["verse_index_pooled"]),
    )


def write_csv(path, comment_lines, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        for line in comment_lines:
            assert line.startswith("#") and "\n" not in line
            handle.write(line + "\n")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    reports = root / "reports"
    print(f"root = {root}")

    rows = load_chunks(root / "chunks" / "shakespeare_chunks_v2.csv")
    n_blank = sum(1 for row in rows if row["verse_ratio"] is None)
    n_works = len(set(row["work"] for row in rows))
    print(f"chunks = {len(rows)}, works = {n_works}, chunks with blank verse_ratio = {n_blank}")

    result = summarize(rows)
    csv_rows = [to_csv_row(r) for r in result]
    write_csv(reports / "verse_index_by_work.csv", COMMENT_LINES, CSV_HEADER, csv_rows)

    print()
    print(f"{'work':<5} {'chunks':>6} {'w/ratio':>7} {'chars':>8} {'vi_wtd':>7} {'prose0':>7} {'verse1':>7} {'v_lines':>8} {'p_paras':>8} {'vi_pool':>7}")
    for row in csv_rows:
        print(f"{row[0]:<5} {row[1]:>6} {row[2]:>7} {row[3]:>8} {row[4]:>7} {row[5]:>7} {row[6]:>7} {row[7]:>8} {row[8]:>8} {row[9]:>7}")
    print(f"\nwritten: {reports / 'verse_index_by_work.csv'}")
    print(f"total elapsed {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

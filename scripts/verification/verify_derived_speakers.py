#!/usr/bin/env python3
"""空speakerを持つFolger TEI発話の派生話者名を独立検証する。

用途:
    全38作から子 ``speaker`` が不存在または空の ``sp`` を抽出し、
    設計メモ§5の5規則を判定順どおり独立実装する。作品・話者ラベル・
    発話本文を用いて既存chunks CSV内の対応 ``chunk_id`` も探索する。
引数:
    ``--corpus-root PATH``（既定: ``../../../じんもんこん/corpus_raw``）、
    ``--chunks-csv PATH``（既定: ``../../chunks/shakespeare_chunks_v2.csv``）。
期待値:
    sp単位で EPILOGUE.=3, CHORUS.=2, ALL.=2, MUSICIANS.=3,
    BOY.=2, WITHIN.=4（計16件）。対応チャンクの発見率100%。
実行例:
    ``python3 verify_derived_speakers.py``
    ``python3 verify_derived_speakers.py --corpus-root /abs/corpus_raw
    --chunks-csv /abs/shakespeare_chunks_v2.csv``
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET


EXPECTED = {
    "EPILOGUE.": 3,
    "CHORUS.": 2,
    "ALL.": 2,
    "MUSICIANS.": 3,
    "BOY.": 2,
    "WITHIN.": 4,
}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def normalize_space(text):
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def comparison_key(text):
    """Remove TEI/rendering punctuation differences for source-to-CSV matching."""
    return "".join(
        char.casefold() for char in unicodedata.normalize("NFC", text or "")
        if char.isalnum()
    )


def defaults():
    here = Path(__file__).resolve().parent
    return (
        here.joinpath("../../../じんもんこん/corpus_raw").resolve(),
        here.joinpath("../../chunks/shakespeare_chunks_v2.csv").resolve(),
    )


def find_play_xmls(root):
    files = []
    if not root.is_dir():
        raise FileNotFoundError("corpus root not found: {}".format(root))
    for play_dir in sorted((p for p in root.iterdir() if p.is_dir()),
                           key=lambda p: p.name):
        candidates = sorted(
            p for p in play_dir.rglob("*.xml")
            if "__MACOSX" not in p.parts
        )
        exact = [p for p in candidates if p.stem == play_dir.name]
        selected = exact or candidates
        if selected:
            files.append((play_dir.name, selected[0]))
    return files


def nearest_div_type(speech, parent):
    ancestor = parent.get(speech)
    while ancestor is not None:
        if local_name(ancestor.tag) in {"div1", "div2"}:
            return (ancestor.get("type") or "").strip().lower()
        ancestor = parent.get(ancestor)
    return ""


def derive_speaker(speech, div_type):
    """Apply the five rules in their mandated precedence."""
    if div_type == "epilogue":
        return "EPILOGUE."
    if div_type in {"prologue", "chorus"}:
        return "CHORUS."
    who = speech.get("who") or ""
    if " " in who:
        return "ALL."
    if who:
        return who.lstrip("#").rsplit("_", 1)[0].upper() + "."
    return "WITHIN."


def empty_direct_speakers(speech):
    speakers = [
        child for child in speech if local_name(child.tag) == "speaker"
    ]
    return not speakers or all(
        not normalize_space("".join(speaker.itertext()))
        for speaker in speakers
    )


def speech_body(speech):
    """Extract normalized speech text excluding direct speaker children."""
    return normalize_space("".join(
        text
        for child in speech
        if local_name(child.tag) != "speaker"
        for text in child.itertext()
    ))


def extract_cases(play, path):
    root = ET.parse(str(path)).getroot()
    parent = {child: node for node in root.iter() for child in node}
    cases = []
    for speech in (node for node in root.iter()
                   if local_name(node.tag) == "sp"):
        if not empty_direct_speakers(speech):
            continue
        div_type = nearest_div_type(speech, parent)
        cases.append({
            "play": play,
            "who": speech.get("who") or "",
            "div_type": div_type,
            "derived": derive_speaker(speech, div_type),
            "body": speech_body(speech),
        })
    return cases


def load_chunks(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "folger_abbr", "text_content"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV lacks required columns: {}".format(
                ", ".join(sorted(required))
            ))
        return [
            {
                "chunk_id": row["chunk_id"],
                "play": row["folger_abbr"],
                "text": normalize_space(row["text_content"]),
            }
            for row in reader
        ]


def matching_chunk_ids(case, chunks):
    """Find label-bearing chunks and disambiguate them by source speech text."""
    same_play = [
        row for row in chunks
        if row["play"] == case["play"] and case["derived"] in row["text"]
    ]
    if not case["body"]:
        return [row["chunk_id"] for row in same_play]
    # TEI token spacing and rendered punctuation differ. Alphanumeric prefixes
    # remain stable and survive the first fragment of a split speech.
    probe = comparison_key(case["body"])[:80]
    exact = [
        row["chunk_id"] for row in same_play
        if probe in comparison_key(row["text"])
    ]
    if exact:
        return exact
    # Fall back for short stage-direction speeches and boundary adjacency.
    probe = comparison_key(case["body"])[:30]
    return [
        row["chunk_id"] for row in same_play
        if probe in comparison_key(row["text"])
    ]


def main():
    corpus_default, csv_default = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=corpus_default)
    parser.add_argument("--chunks-csv", type=Path, default=csv_default)
    args = parser.parse_args()

    try:
        play_xmls = find_play_xmls(args.corpus_root.resolve())
        if len(play_xmls) != 38:
            raise ValueError("expected 38 play XMLs, found {}".format(
                len(play_xmls)
            ))
        cases = []
        for play, path in play_xmls:
            cases.extend(extract_cases(play, path))
        chunks = load_chunks(args.chunks_csv.resolve())
    except (OSError, ValueError, ET.ParseError, csv.Error) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    print("=== 1. 空 <speaker> の <sp> 全件 ===")
    found = 0
    for number, case in enumerate(cases, 1):
        chunk_ids = matching_chunk_ids(case, chunks)
        case["chunk_ids"] = chunk_ids
        if chunk_ids:
            found += 1
        print("{:02d}. {} | who={!r} | div_type={!r} | derived={}".format(
            number, case["play"], case["who"], case["div_type"],
            case["derived"]
        ))
        print("    chunk_id: {}".format(
            ", ".join(chunk_ids) if chunk_ids else "[NOT FOUND]"
        ))

    distribution = Counter(case["derived"] for case in cases)
    print("\n=== 2. 派生話者名分布（sp単位） ===")
    for label in EXPECTED:
        print("{}: {}".format(label, distribution[label]))
    unexpected = sorted(set(distribution) - set(EXPECTED))
    for label in unexpected:
        print("{}: {} [unexpected]".format(label, distribution[label]))

    print("\n=== 3. 期待値との照合 ===")
    all_match = True
    for label, expected in EXPECTED.items():
        actual = distribution[label]
        match = actual == expected
        all_match = all_match and match
        print("{} actual={} expected={} {}".format(
            label, actual, expected, "OK" if match else "MISMATCH"
        ))

    print("\n=== 4. 対応チャンク発見率 ===")
    print("{}/{} sp ({:.2f}%)".format(
        found, len(cases), 100.0 * found / len(cases) if cases else 0.0
    ))
    print("分布照合: {}".format("OK" if all_match else "MISMATCH"))
    return 0 if all_match and found == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())

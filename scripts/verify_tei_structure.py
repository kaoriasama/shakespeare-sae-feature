#!/usr/bin/env python3
"""Folger TEI 38作品の構造・発話長・stage type を検証する。

用途:
    XMLごとの構造シグネチャ、全発話の未正規化 ``itertext()`` 文字数、
    stage type を独立走査して標準出力へ集計する。
引数:
    ``--corpus-root PATH``。既定値はスクリプト位置基準の
    ``../../じんもんこん/corpus_raw``。
期待値:
    38作品すべてが ``(div1, div2, milestone, p, sp, speaker, stage)``、
    逸脱0。総発話31,906、p50=111、p95=731、1400字超=412 (1.29%)。
    stage総数10,233（entrance=2420, delivery=2294, exit=2269,
    business=2268, modifier=708, mixed=188, location=83, (none)=2,
    dumbshow=1）。
実行例:
    ``python3 verify_tei_structure.py``
    ``python3 verify_tei_structure.py --corpus-root /abs/corpus_raw``
"""

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


SIGNATURE_ELEMENTS = (
    "div1", "div2", "milestone", "p", "sp", "speaker", "stage"
)
EXPECTED_SIGNATURE = SIGNATURE_ELEMENTS
STAGE_ORDER = (
    "entrance", "exit", "delivery", "business", "modifier",
    "location", "mixed", "dumbshow", "(none)"
)


def local_name(tag):
    """Return the local component of a possibly namespaced XML tag."""
    return tag.rsplit("}", 1)[-1]


def default_corpus_root():
    """Resolve the documented default relative to this script."""
    return Path(__file__).resolve().parent.joinpath(
        "../../じんもんこん/corpus_raw"
    ).resolve()


def find_play_xmls(root):
    """Select one primary XML per immediate play directory."""
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


def nearest_rank(values, percentile):
    """Return the nearest-rank percentile from a sorted nonempty sequence."""
    index = max(0, math.ceil(percentile / 100.0 * len(values)) - 1)
    return values[index]


def scan_xml(play, path):
    """Return one play's signature, speech lengths, and stage counts."""
    root = ET.parse(str(path)).getroot()
    tags = {local_name(element.tag) for element in root.iter()}
    signature = tuple(name for name in SIGNATURE_ELEMENTS if name in tags)
    lengths = []
    stages = Counter()
    for element in root.iter():
        tag = local_name(element.tag)
        if tag == "sp":
            # 過去の検証と同じく、speaker とTEI整形改行を含む未正規化長。
            lengths.append(len("".join(element.itertext())))
        elif tag == "stage":
            stage_type = (element.get("type") or "").strip().lower()
            stages[stage_type or "(none)"] += 1
    return signature, lengths, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path,
                        default=default_corpus_root())
    args = parser.parse_args()

    try:
        play_xmls = find_play_xmls(args.corpus_root.resolve())
    except (OSError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    if len(play_xmls) != 38:
        print("ERROR: expected 38 play XMLs, found {}".format(len(play_xmls)),
              file=sys.stderr)
        return 2

    signatures = defaultdict(list)
    all_lengths = []
    all_stages = Counter()
    per_play = []
    for play, path in play_xmls:
        try:
            signature, lengths, stages = scan_xml(play, path)
        except (OSError, ET.ParseError) as exc:
            print("ERROR: {}: {}".format(path, exc), file=sys.stderr)
            return 2
        signatures[signature].append(play)
        all_lengths.extend(lengths)
        all_stages.update(stages)
        per_play.append((play, signature))

    print("=== 1. 構造シグネチャの作品別分布 ===")
    for signature in sorted(signatures, key=lambda item: (len(item), item)):
        plays = signatures[signature]
        print("{}作品 {}: {}".format(len(plays), signature, " ".join(plays)))
    print("作品別:")
    for play, signature in per_play:
        print("  {}: {}".format(play, signature))

    deviations = [(play, sig) for play, sig in per_play
                  if sig != EXPECTED_SIGNATURE]
    print("\n=== 2. 共通シグネチャからの逸脱 ===")
    print("期待シグネチャ: {}".format(EXPECTED_SIGNATURE))
    print("逸脱: {}作品".format(len(deviations)))
    for play, signature in deviations:
        print("  {}: {}".format(play, signature))

    all_lengths.sort()
    print("\n=== 3. 発話長分布（未正規化 itertext 文字数） ===")
    print("総発話数: {:,}".format(len(all_lengths)))
    for percentile in (50, 75, 90, 95, 99, 100):
        print("p{}: {:,}字".format(
            percentile, nearest_rank(all_lengths, percentile)
        ))
    over_limit = sum(length > 1400 for length in all_lengths)
    ratio = 100.0 * over_limit / len(all_lengths)
    print("1400字超: {:,}件 ({:.2f}%)".format(over_limit, ratio))

    print("\n=== 4. stage type 分布 ===")
    print("stage総数: {:,}".format(sum(all_stages.values())))
    for stage_type in STAGE_ORDER:
        print("{}: {:,}".format(stage_type, all_stages[stage_type]))
    unknown = sorted(set(all_stages) - set(STAGE_ORDER))
    for stage_type in unknown:
        print("{}: {:,} [unexpected]".format(
            stage_type, all_stages[stage_type]
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

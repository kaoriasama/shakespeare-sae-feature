#!/usr/bin/env python3
"""Gemma-2-2B Neuronpedia 一括JSONの演劇・古語ラベルを網羅検索する。

用途:
    全26層の res-16k export の ``description`` を小文字化し、先頭定数
    ``SEARCH_TERMS`` の部分一致で検索する。層別件数、保存済み残差層
    L5/L12/L20、採用候補層 L4/L11/L22 の全ヒットを表示する。
引数:
    ``--export-root PATH``。既定値はスクリプト位置基準の
    ``../../../研究計画/1＿Neuronpedia詩人名ラベル調査``。
期待値:
    26層合計100件超。L4 idx 7718 ``references to Shakespeare and
    theatrical works``、L11 idx 6671、L22 idx 9813, 15948 等を含む。
実行例:
    ``python3 search_neuronpedia_local.py``
    ``python3 search_neuronpedia_local.py --export-root /abs/exports``
"""

import argparse
import json
import re
import sys
from pathlib import Path


SEARCH_TERMS = (
    "shakespear", "theatre", "theater", "theatrical", "playwright",
    "drama", "tragedy", "tragic", "archaic", "old english",
    "early modern", "elizabethan", "soliloquy", "iambic", "verse",
    "thou", "thee", "thy", "doth", "hath", "stage direction",
    "play script", "dialogue",
)
SAVED_RESIDUAL_LAYERS = (5, 12, 20)
CANDIDATE_LAYERS = (4, 11, 22)
FILE_RE = re.compile(r"gemma-2-2b_(\d+)-gemmascope-res-16k\.json$")


def documented_default():
    return Path(__file__).resolve().parent.joinpath(
        "../../../研究計画/1＿Neuronpedia詩人名ラベル調査"
    ).resolve()


def resolve_export_root(requested):
    """Resolve the documented path, tolerating its known spacing variant."""
    requested = requested.expanduser().resolve()
    if requested.is_dir():
        return requested
    parent = requested.parent
    matches = sorted(
        p for p in parent.glob("1＿*Neuronpedia*")
        if p.is_dir()
    ) if parent.is_dir() else []
    if len(matches) == 1:
        print("NOTE: export directory resolved to {}".format(matches[0]),
              file=sys.stderr)
        return matches[0]
    raise FileNotFoundError("export root not found: {}".format(requested))


def load_layer(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("{}: top-level JSON is not a list".format(path))
    return data


def matching_records(records):
    hits = []
    for record in records:
        if not isinstance(record, dict):
            continue
        description = str(record.get("description") or "")
        lowered = description.lower()
        matched = next((term for term in SEARCH_TERMS if term in lowered),
                       None)
        if matched is not None:
            index = record.get("index", record.get("idx", ""))
            hits.append((index, description, matched))
    return hits


def index_sort_key(hit):
    try:
        return (0, int(hit[0]))
    except (TypeError, ValueError):
        return (1, str(hit[0]))


def print_layer_hits(title, layers, hits_by_layer):
    print("\n=== {} ===".format(title))
    for layer in layers:
        print("L{}: {}件".format(layer, len(hits_by_layer[layer])))
        for index, description, term in sorted(
                hits_by_layer[layer], key=index_sort_key):
            print("  idx={} | matched_term={!r} | {}".format(
                index, term, description
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path,
                        default=documented_default())
    args = parser.parse_args()

    try:
        root = resolve_export_root(args.export_root)
        files = {}
        for path in root.glob("gemma-2-2b_*-gemmascope-res-16k.json"):
            match = FILE_RE.search(path.name)
            if match:
                files[int(match.group(1))] = path
        missing = sorted(set(range(26)) - set(files))
        if missing:
            raise FileNotFoundError("missing layer exports: {}".format(
                ", ".join("L{}".format(n) for n in missing)
            ))
        hits_by_layer = {
            layer: matching_records(load_layer(files[layer]))
            for layer in range(26)
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    print("=== 1. 層別ヒット数 ===")
    for layer in range(26):
        print("L{}: {}件".format(layer, len(hits_by_layer[layer])))
    print("全26層合計: {}件".format(
        sum(len(hits) for hits in hits_by_layer.values())
    ))
    print_layer_hits("2. 保存済み残差層の全ヒット",
                     SAVED_RESIDUAL_LAYERS, hits_by_layer)
    print_layer_hits("3. 採用候補層の全ヒット",
                     CANDIDATE_LAYERS, hits_by_layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())

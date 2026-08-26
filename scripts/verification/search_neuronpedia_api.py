#!/usr/bin/env python3
"""Neuronpedia APIで Gemma-3-4B の演劇・古語候補特徴を検索する。

用途:
    Explanation Search APIへ13検索語をPOSTし、応答とヒットをJSON保存する。
    IncompleteRead 時は ``e.partial`` を救出し、完全に読めるJSONオブジェクト
    だけを ``JSONDecoder.raw_decode`` で回収する。
引数:
    ``--out PATH``（既定値 ``./np_g3_hits.json``）。
期待値:
    L9/L17/L22/L29 の16kに数件以上、262kに多数。特に L9 16k
    idx 7715 ``theater and plays``、L17 16k idx 9814
    ``shakespearean language``、L29 16k idx 11332
    ``archaic english forms`` を含む。
実行例:
    ``python3 search_neuronpedia_api.py --out ./np_g3_hits.json``

外部通信を行うため、オフライン検証では実行せず構文と設計だけを確認する。
"""

import argparse
from collections import Counter
import http.client
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request


API_URL = "https://www.neuronpedia.org/api/explanation/search-model"
MODEL_ID = "gemma-3-4b"
SEARCH_TERMS = (
    "shakespear", "theatre", "theater", "theatrical", "playwright",
    "drama", "tragedy", "tragic", "archaic", "old english",
    "early modern", "elizabethan", "soliloquy",
)
LAYERS = (9, 17, 22, 29)
WIDTHS = ("16k", "65k", "262k")


def raw_decode_objects(text):
    """Recover independently complete JSON objects from possibly partial text."""
    decoder = json.JSONDecoder()
    objects = []
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        position = max(end, start + 1)
    return objects


def decode_response(payload, partial):
    text = payload.decode("utf-8", errors="replace")
    if not partial:
        try:
            return json.loads(text), False
        except json.JSONDecodeError:
            pass
    return raw_decode_objects(text), True


def request_query(query):
    body = json.dumps(
        {"modelId": MODEL_ID, "query": query}
    ).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "jinmonkon2026-verification/1.0",
        },
    )
    partial = False
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            try:
                payload = response.read()
            except http.client.IncompleteRead as exc:
                payload = exc.partial
                partial = True
    except http.client.IncompleteRead as exc:
        payload = exc.partial
        partial = True
    value, recovered = decode_response(payload, partial)
    return value, partial or recovered


def iter_dicts(value):
    """Yield nested dict records without assuming a fixed API envelope."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def layer_width(record):
    """Infer a result's layer and SAE width from common response fields."""
    searchable = " ".join(
        str(record.get(key, ""))
        for key in (
            "layer", "layerId", "saeId", "source", "sourceId",
            "setName", "url"
        )
    ).lower()
    layer_match = re.search(r"(?:^|[^0-9])(?:l|layer[-_ ]?)(0?9|17|22|29)"
                            r"(?:[^0-9]|$)", searchable)
    if not layer_match:
        layer_match = re.search(r"(?:^|[-_/])(9|17|22|29)(?:[-_/]|$)",
                                searchable)
    width_match = re.search(r"(262k|65k|16k)", searchable)
    if not layer_match or not width_match:
        return None
    return int(layer_match.group(1)), width_match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("./np_g3_hits.json"))
    args = parser.parse_args()

    responses = []
    try:
        for number, query in enumerate(SEARCH_TERMS):
            if number:
                time.sleep(1)
            print("POST query={!r}".format(query), file=sys.stderr)
            value, partial = request_query(query)
            responses.append({
                "query": query,
                "partial_response_recovered": partial,
                "response": value,
            })
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("ERROR: API request failed: {}".format(exc), file=sys.stderr)
        return 2

    output = {
        "modelId": MODEL_ID,
        "endpoint": API_URL,
        "queries": list(SEARCH_TERMS),
        "results": responses,
    }
    try:
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        print("ERROR: cannot write {}: {}".format(args.out, exc),
              file=sys.stderr)
        return 2

    distribution = Counter()
    for response in responses:
        for record in iter_dicts(response["response"]):
            key = layer_width(record)
            if key is not None:
                distribution[key] += 1

    print("=== 層別 × 幅別ヒット分布 ===")
    print("layer\t16k\t65k\t262k")
    for layer in LAYERS:
        print("L{}\t{}\t{}\t{}".format(
            layer,
            distribution[(layer, "16k")],
            distribution[(layer, "65k")],
            distribution[(layer, "262k")],
        ))
    print("保存先: {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

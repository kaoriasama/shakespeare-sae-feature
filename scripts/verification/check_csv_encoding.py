#!/usr/bin/env python3
"""チャンクCSVのBOM・文字コード・非ASCII文字と文字化けを検証する。

用途:
    BOMとUTF系エンコーディングを判定し、全 ``text_content`` の非ASCII
    文字、U+FFFD、制御文字を集計する。代表UTF-8列をCP932で誤読した
    結果と、実データから抽出した連続バイト列の誤読結果を表示する。
引数:
    ``--csv PATH``。既定値はスクリプト位置基準の
    ``../../chunks/shakespeare_chunks_v2.csv``。
期待値:
    現行CSVは先頭 ``EF BB BF``、utf-8、U+FFFD=0、制御文字=0、
    非ASCII 19種類。CP932誤解釈は3バイト単独では実挙動を再現できない。
    実Excelでの化けは連続バイト列をCP932が2バイトずつ解釈して発生する
    （例: ``mak’st`` を含むUTF-8列全体では ``mak窶冱t`` 型になる）。
    ``è`` は ``ﾃｨ`` になる。単独列と実データ連続列をともに実測表示する。
実行例:
    ``python3 check_csv_encoding.py``
    ``python3 check_csv_encoding.py --csv /abs/shakespeare_chunks_v2.csv``
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
import unicodedata


def default_csv():
    return Path(__file__).resolve().parent.joinpath(
        "../../chunks/shakespeare_chunks_v2.csv"
    ).resolve()


def detect_encoding(raw):
    """Return a file-I-like conservative encoding label."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw.decode("utf-8-sig")
        return "utf-8"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raw.decode("utf-16")
        return "utf-16"
    try:
        raw.decode("ascii")
        return "us-ascii"
    except UnicodeDecodeError:
        pass
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "unknown-8bit"


def is_disallowed_control(character):
    return character not in "\n\t" and unicodedata.category(character) == "Cc"


def extract_sample(texts, target, preferred):
    """Extract a real roughly nine-character window around one target."""
    for needle in (preferred, target):
        for text in texts:
            needle_position = text.find(needle)
            if needle_position < 0:
                continue
            target_position = text.find(target, needle_position)
            if target_position < 0:
                continue
            return text[
                max(0, target_position - 4):
                min(len(text), target_position + 5)
            ]
    raise ValueError("no text_content sample contains {!r}".format(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=default_csv())
    args = parser.parse_args()
    path = args.csv.expanduser().resolve()

    try:
        raw = path.read_bytes()
        encoding = detect_encoding(raw)
        if encoding not in {"utf-8", "us-ascii"}:
            raise UnicodeError("CSV is not UTF-8: {}".format(encoding))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "text_content" not in (reader.fieldnames or []):
                raise ValueError("CSV lacks text_content column")
            inventory = Counter()
            replacement_count = 0
            control_count = 0
            row_count = 0
            texts = []
            for row in reader:
                row_count += 1
                text = row["text_content"]
                texts.append(text)
                inventory.update(char for char in text if not char.isascii())
                replacement_count += text.count("\ufffd")
                control_count += sum(
                    is_disallowed_control(char) for char in text
                )
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    first_three = raw[:3]
    print("=== 1. 先頭3バイト / BOM ===")
    print("先頭3バイト: {}".format(first_three.hex(" ").upper()))
    print("UTF-8 BOM: {}".format(
        "あり" if first_three == b"\xef\xbb\xbf" else "なし"
    ))

    print("\n=== 2. file -I 相当の判定 ===")
    print("encoding: {}".format(encoding))
    print("MIME相当: text/csv; charset={}".format(encoding))

    print("\n=== 3. text_content 非ASCII文字インベントリ ===")
    print("CSVデータ行: {:,}".format(row_count))
    print("非ASCII種類数: {}".format(len(inventory)))
    print("非ASCII総数: {:,}".format(sum(inventory.values())))
    print("頻度上位20:")
    for character, count in inventory.most_common(20):
        print("  U+{:04X} {!r}: {:,} | category={} | {}".format(
            ord(character), character, count,
            unicodedata.category(character),
            unicodedata.name(character, "<unnamed>"),
        ))

    print("\n=== 4. U+FFFD 置換文字 ===")
    print("U+FFFD: {}".format(replacement_count))
    print("\n=== 5. 制御文字（改行・タブを除く） ===")
    print("制御文字: {}".format(control_count))

    print("\n=== 6. UTF-8をCP932として誤解釈 ===")
    examples = (("’", "E2 80 99"), ("—", "E2 80 94"), ("è", "C3 A8"))
    for character, expected_hex in examples:
        encoded = character.encode("utf-8")
        mojibake = encoded.decode("cp932", errors="replace")
        print("{!r} | UTF-8 {} | CP932誤読 {!r}".format(
            character, expected_hex, mojibake
        ))

    # 実Excelの化け方を厳密に再現するには、対象文字の数バイトだけで
    # なく隣接文字を含む連続バイト列をCP932として読む必要がある。
    print("\n=== 7. 連続バイト列での CP932 誤解釈シミュレーション ===")
    sample_specs = (
        ("’", "mak’st"), ("—", "—Majesty"), ("è", "utterèd")
    )
    for character, preferred in sample_specs:
        sample = extract_sample(texts, character, preferred)
        encoded = sample.encode("utf-8")
        mojibake = encoded.decode("cp932", errors="replace")
        isolated = character.encode("utf-8")
        isolated_mojibake = isolated.decode("cp932", errors="replace")
        print("サンプル: {!r} (real text_content)".format(sample))
        print("  UTF-8: {!r}".format(encoded))
        print("  CP932 全体デコード: {!r}".format(mojibake))
        print("  参考: 単独 {!r}.decode('cp932', errors='replace') = {!r}".format(
            isolated, isolated_mojibake
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

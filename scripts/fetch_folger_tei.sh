#!/bin/bash
# Usage: ./fetch_folger_tei.sh <output_dir>
#
# 用途: Folger Shakespeare の全38戯曲の TEI XML zip を再取得する。
# 引数: <output_dir> は実行前に絶対パスで指定すること。
# 期待値: 38作品が作品略号別ディレクトリへ展開され、各zipが unzip -tq
#         に合格する。既存の作品ディレクトリは変更せずスキップする。
# 実行例: ./fetch_folger_tei.sh /absolute/path/to/corpus_raw
#
# 必要コマンド: bash, curl, unzip（実行前に存在を確認すること）。
# 取得履歴: 既存15作は 2026-07-19、残り23作は 2026-07-28 に取得済み。
# 権利表示: Folger Digital Texts, CC BY-NC 3.0 Unported。
# 非商用条件、帰属表示、ライセンス原文を確認してから利用すること。

set -u

USER_AGENT='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36'

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <output_dir>" >&2
    exit 2
fi

output_dir=$1
case "$output_dir" in
    /*) ;;
    *)
        echo "Error: <output_dir> must be an absolute path." >&2
        exit 2
        ;;
esac

for command_name in bash curl unzip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Error: required command not found: $command_name" >&2
        exit 1
    fi
done

echo "License warning: Folger Digital Texts are licensed CC BY-NC 3.0 Unported."
echo "Confirm attribution and noncommercial-use requirements before redistribution."

mkdir -p "$output_dir" || exit 1

# Folger略号とURLコード。2文字略号は末尾を「_」で3文字に揃える。
plays='
Ant Ant
Cor Cor
JC JC_
Rom Rom
Tim Tim
Tit Tit
Err Err
LLL LLL
MV MV_
Wiv Wiv
Shr Shr
TGV TGV
1H4 1H4
2H4 2H4
H5 H5_
1H6 1H6
2H6 2H6
3H6 3H6
H8 H8_
Jn Jn_
R2 R2_
R3 R3_
TNK TNK
AWW AWW
AYL AYL
Ado Ado
Cym Cym
Ham Ham
MM MM_
MND MND
Mac Mac
Oth Oth
Per Per
TN TN_
Tmp Tmp
Tro Tro
WT WT_
Lr Lr_
'

had_failure=0
while read -r abbr code; do
    [ -n "$abbr" ] || continue
    destination="$output_dir/$abbr"
    if [ -d "$destination" ]; then
        echo "SKIP $abbr: existing directory $destination"
        continue
    fi

    zip_path=$(mktemp "${TMPDIR:-/tmp}/folger_${abbr}.XXXXXX.zip") || exit 1
    url="https://flgr.sh/txtfss${code}xml"
    echo "GET  $abbr: $url"

    if ! curl --fail --location --silent --show-error \
        --user-agent "$USER_AGENT" --output "$zip_path" "$url"; then
        echo "FAIL $abbr: download failed" >&2
        rm -f "$zip_path"
        had_failure=1
        sleep 1
        continue
    fi
    if ! unzip -tq "$zip_path" >/dev/null; then
        echo "FAIL $abbr: zip integrity test failed" >&2
        rm -f "$zip_path"
        had_failure=1
        sleep 1
        continue
    fi

    if ! mkdir "$destination"; then
        echo "FAIL $abbr: cannot create $destination" >&2
        rm -f "$zip_path"
        had_failure=1
        sleep 1
        continue
    fi
    if unzip -q "$zip_path" -d "$destination"; then
        echo "OK   $abbr: extracted to $destination"
    else
        echo "FAIL $abbr: extraction failed; partial directory retained" >&2
        had_failure=1
    fi
    rm -f "$zip_path"
    # curl完了から次のリクエスト開始まで最低1秒を空ける。
    sleep 1
done <<EOF
$plays
EOF

exit "$had_failure"

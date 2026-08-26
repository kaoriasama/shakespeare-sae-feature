import csv
from pathlib import Path
import numpy as np

# リポジトリのルート
CG = Path(__file__).resolve().parents[2]

# チャンクID → 作品名 の対応表を読む
work_of = {}
with open(CG / "chunks/shakespeare_chunks_v2.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        work_of[row["chunk_id"]] = row["folger_abbr"]

for model, layer in [("g2", 4), ("g2", 11), ("g2", 22), ("g3", 9), ("g3", 17), ("g3", 29)]:
    z = np.load(CG / f"activations_v2/{model}_L{layer}_firing.npz")
    chunk_ids = list(z["chunk_ids"])
    rate = z["firing_rate"]                    # 4600チャンク × 16384特徴量 の発火率
    works = sorted({work_of[c] for c in chunk_ids})
    per_work = []
    for w in works:                            # 作品ごとに…
        rows = [i for i, c in enumerate(chunk_ids) if work_of[c] == w]
        per_work.append(rate[rows].mean(axis=0))   # …チャンク発火率を平均する
    per_work = np.array(per_work)              # 38作品 × 16384特徴量
    n_works_fired = (per_work > 0).sum(axis=0) # 発火した作品数（0〜38）
    n_all = int((n_works_fired == 38).sum())
    n_absent = int((n_works_fired == 0).sum())
    n_partial = 16384 - n_all - n_absent
    weakest = per_work.min(axis=0)             # 38作品での最小の作品内平均
    col5 = np.where(weakest > 0.5)[0]          # 最弱作品ですら0.5超の特徴量
    print(f"{model} L{layer}: 全38作 {n_all} / 一部 {n_partial} / 不在 {n_absent} / 5列目 {len(col5)}")
    for f_ in col5:
        w = works[int(per_work[:, f_].argmin())]
        print(f"    f{f_}  最弱作品 {w} = {weakest[f_]:.3f}")

"""Length-equalized recount of SAE feature firing across the 38 Shakespeare works (A).

A feature is "present" in a work when it fires (firing_rate > 0) in at least one
chunk of that work.

* Observed presence (W_obs) uses every chunk of every work.
* Length-equalized presence (W_r) uses, for each work, a random subset of its
  chunks whose total n_chars does not exceed a budget B.  B defaults to the
  smallest work's total n_chars (derived from the data).  The subsets are drawn
  `reps` times with a single RNG; the drawing does not depend on the layer, so
  the very same subsets are applied to every layer.

Subset rule (prefix rule): for work w, perm = rng.permutation(n_chunks_w); walk
perm from the start and adopt chunks while the cumulative n_chars stays <= B;
stop at the first chunk that would exceed B (no skipping).  Generation order is
fixed: for rep in range(reps): for work in sorted(works).

Read-only inputs
  activations_v2/{model}_L{layer}_firing.npz   chunk_ids (4600,), firing_rate (4600,16384)
  chunks/shakespeare_chunks_v2.csv            chunk_id, folger_abbr, n_chars
  reports/{model}_L{layer}_partial_patterns.csv  used to cross-check the observed counts

Outputs (all under reports/)
  equalized_presence_{model}_L{layer}.npz
  length_equalized_reps_{model}_L{layer}.npz
  length_equalized_samples_first3.json
  length_equalized_by_work.csv
  length_equalized_classification.csv

Only the standard library and numpy are used.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_LAYERS = "g2_L4,g2_L11,g2_L22,g3_L9,g3_L17,g3_L29"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_layers(text):
    layers = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        model, layer = token.split("_L")
        layers.append((model, int(layer)))
    if not layers:
        raise ValueError("no layers given")
    return layers


def load_chunks(csv_path):
    """Return chunk rows (file order) with only the columns needed."""
    csv.field_size_limit(10**9)
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "folger_abbr", "n_chars"}
        assert reader.fieldnames is not None
        missing = required - set(reader.fieldnames)
        assert not missing, f"chunks CSV lacks columns: {sorted(missing)}"
        for row in reader:
            rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "folger_abbr": row["folger_abbr"],
                    "n_chars": int(row["n_chars"]),
                }
            )
    ids = [row["chunk_id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk_id in chunks CSV"
    return rows


def load_presence(npz_path, csv_chunk_ids):
    """Return P = firing_rate > 0 (bool, chunks x features) after checking that
    the npz chunk order is identical to the CSV order."""
    with np.load(npz_path, allow_pickle=False) as data:
        npz_ids = [str(item) for item in data["chunk_ids"].tolist()]
        assert npz_ids == csv_chunk_ids, (
            f"{npz_path.name}: chunk_ids differ from CSV chunk_id (order included)"
        )
        firing_rate = data["firing_rate"]
        assert firing_rate.shape[0] == len(csv_chunk_ids)
        assert np.isfinite(firing_rate).all(), f"{npz_path.name}: non-finite values"
        presence = firing_rate > 0
    return presence


def presence_stats(W):
    """W: (n_works, n_features) bool.  Returns n_fired, (n_all, n_partial, n_none),
    single (per work), complement (per work)."""
    n_works = W.shape[0]
    n_fired = W.sum(axis=0, dtype=np.int32)
    n_all = int(np.count_nonzero(n_fired == n_works))
    n_partial = int(np.count_nonzero((n_fired >= 1) & (n_fired <= n_works - 1)))
    n_none = int(np.count_nonzero(n_fired == 0))
    # exactly-one-work features, attributed to the work in which they fire
    single = (W & (n_fired == 1)[None, :]).sum(axis=1, dtype=np.int32)
    # exactly-(n_works-1) features, attributed to the one work in which they do NOT fire
    complement = (~W & (n_fired == n_works - 1)[None, :]).sum(axis=1, dtype=np.int32)
    return n_fired, (n_all, n_partial, n_none), single, complement


def rank_average(values):
    """1-based ranks with ties replaced by the average rank."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = rank_average(x)
    ry = rank_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def read_partial_patterns(csv_path):
    """Rows of reports/{model}_L{layer}_partial_patterns.csv as
    (n_features, n_works, tuple(works))."""
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            works = tuple(row["works"].split())
            n_works = int(row["n_works"])
            assert n_works == len(works), f"{csv_path.name}: n_works != len(works) at rank {row['rank']}"
            rows.append((int(row["n_features"]), n_works, works))
    return rows


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# subset generation (layer independent)
# --------------------------------------------------------------------------- #
def generate_selections(rng, works, n_chars_rows, budget, reps):
    """selections[rep][work_index] = local chunk indices (into the work's row
    list) in adoption order.  Prefix rule: adopt along the permutation while the
    cumulative n_chars stays <= budget; stop at the first excess."""
    selections = []
    for _rep in range(reps):
        per_work = []
        for wi, _work in enumerate(works):
            n_c = n_chars_rows[wi]
            perm = rng.permutation(len(n_c))
            cum = np.cumsum(n_c[perm])
            # cum is non-decreasing, so {cum <= budget} is a prefix of perm;
            # its length is the position of the first excess.
            n_take = int(np.count_nonzero(cum <= budget))
            per_work.append(perm[:n_take].astype(np.int64))
        selections.append(per_work)
    return selections


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=None, help="character budget per work (default: smallest work's total n_chars)")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    reports = root / "reports"
    layers = parse_layers(args.layers)
    print(f"root     = {root}")
    print(f"seed     = {args.seed}, reps = {args.reps}")
    print(f"layers   = {', '.join(f'{m}_L{l}' for m, l in layers)}")

    # ---- chunks ---------------------------------------------------------- #
    chunk_rows = load_chunks(root / "chunks" / "shakespeare_chunks_v2.csv")
    csv_chunk_ids = [row["chunk_id"] for row in chunk_rows]
    works = sorted({row["folger_abbr"] for row in chunk_rows})
    n_works = len(works)
    work_index = {w: i for i, w in enumerate(works)}
    rows_of_work = [[] for _ in works]
    for row_idx, row in enumerate(chunk_rows):
        rows_of_work[work_index[row["folger_abbr"]]].append(row_idx)
    rows_of_work = [np.array(r, dtype=np.int64) for r in rows_of_work]
    n_chars_rows = [np.array([chunk_rows[i]["n_chars"] for i in r], dtype=np.int64) for r in rows_of_work]
    n_chunks_total = np.array([len(r) for r in rows_of_work], dtype=np.int64)
    n_chars_total = np.array([int(c.sum()) for c in n_chars_rows], dtype=np.int64)
    print(f"chunks   = {len(chunk_rows)}, works = {n_works}")
    print(f"works    = {' '.join(works)}")

    budget = int(n_chars_total.min()) if args.budget is None else int(args.budget)
    min_work = works[int(np.argmin(n_chars_total))]
    print(f"budget B = {budget} chars (smallest work: {min_work}, {int(n_chars_total.min())} chars, {int(n_chunks_total[work_index[min_work]])} chunks)")

    # ---- subsets (one RNG, layer independent) ---------------------------- #
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    selections = generate_selections(rng, works, n_chars_rows, budget, args.reps)
    chunks_used = np.array([[len(sel) for sel in per_work] for per_work in selections], dtype=np.int32)
    print(f"subsets generated in {time.time() - t0:.1f}s")
    mean_used = chunks_used.mean(axis=0)
    all_used_min = bool(np.all(chunks_used[:, work_index[min_work]] == n_chunks_total[work_index[min_work]]))
    print(f"check: smallest work {min_work} uses all {int(n_chunks_total[work_index[min_work]])} chunks in every rep: {all_used_min}")

    samples = {
        "seed": args.seed,
        "budget_chars": budget,
        "reps_total": args.reps,
        "rule": "for rep: for work in sorted(works): perm = rng.permutation(n_chunks); adopt along perm while cumulative n_chars <= budget; stop at first excess",
        "reps": {},
    }
    for rep in range(min(3, args.reps)):
        samples["reps"][str(rep)] = {
            w: [chunk_rows[int(rows_of_work[wi][j])]["chunk_id"] for j in selections[rep][wi]]
            for wi, w in enumerate(works)
        }
    with (reports / "length_equalized_samples_first3.json").open("w", encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=1)

    # ---- per layer ------------------------------------------------------- #
    by_work_rows = []
    classification_rows = []
    order_by_chars = np.argsort(n_chars_total, kind="mergesort")

    for model, layer in layers:
        tag = f"{model}_L{layer}"
        t_layer = time.time()
        print(f"\n===== {tag} =====")
        P = load_presence(root / "activations_v2" / f"{tag}_firing.npz", csv_chunk_ids)
        n_features = P.shape[1]
        P_work = [P[rows] for rows in rows_of_work]
        del P

        # observed (all chunks)
        W_obs = np.stack([pw.any(axis=0) for pw in P_work])
        n_fired_obs, cls_obs, single_obs, complement_obs = presence_stats(W_obs)
        print(f"observed: n_all={cls_obs[0]} n_partial={cls_obs[1]} n_none={cls_obs[2]}")

        # cross-check against reports/{tag}_partial_patterns.csv (acceptance 1)
        pattern_rows = read_partial_patterns(reports / f"{tag}_partial_patterns.csv")
        csv_total = sum(n for n, _, _ in pattern_rows)
        csv_single = np.zeros(n_works, dtype=np.int64)
        csv_complement = np.zeros(n_works, dtype=np.int64)
        all_works_set = set(works)
        for n_feat, n_w, pattern in pattern_rows:
            if n_w == 1:
                csv_single[work_index[pattern[0]]] += n_feat
            elif n_w == n_works - 1:
                missing = all_works_set - set(pattern)
                assert len(missing) == 1, f"{tag}: 37-work pattern with {len(missing)} missing works"
                csv_complement[work_index[missing.pop()]] += n_feat
        ok_total = csv_total == cls_obs[1]
        ok_single = bool(np.array_equal(csv_single, single_obs))
        ok_complement = bool(np.array_equal(csv_complement, complement_obs))
        print(f"cross-check vs {tag}_partial_patterns.csv: total n_features {csv_total} == n_partial {cls_obs[1]}: {ok_total}; single per work match: {ok_single}; complement per work match: {ok_complement}")
        assert ok_total, f"{tag}: CSV total n_features {csv_total} != n_partial {cls_obs[1]}"
        assert ok_single, f"{tag}: single-work counts differ from CSV: csv={csv_single.tolist()} obs={single_obs.tolist()}"
        assert ok_complement, f"{tag}: complement counts differ from CSV: csv={csv_complement.tolist()} obs={complement_obs.tolist()}"
        # informational: full multiset of partial patterns
        partial_mask = (n_fired_obs >= 1) & (n_fired_obs <= n_works - 1)
        obs_patterns = Counter(
            tuple(w for w, flag in zip(works, col) if flag) for col in W_obs[:, partial_mask].T.tolist()
        )
        csv_patterns = Counter()
        for n_feat, _n_w, pattern in pattern_rows:
            csv_patterns[tuple(sorted(pattern))] += n_feat
        print(f"informational: full partial-pattern multiset identical to CSV: {obs_patterns == csv_patterns} ({len(obs_patterns)} patterns)")

        # length-equalized replicates
        presence_sum = np.zeros((n_works, n_features), dtype=np.int32)
        single_r = np.zeros((args.reps, n_works), dtype=np.int32)
        complement_r = np.zeros((args.reps, n_works), dtype=np.int32)
        cls_r = np.zeros((args.reps, 3), dtype=np.int32)
        W = np.empty((n_works, n_features), dtype=bool)
        t0 = time.time()
        for rep in range(args.reps):
            per_work = selections[rep]
            for wi in range(n_works):
                W[wi] = P_work[wi][per_work[wi]].any(axis=0)
            _n_fired, cls, single, complement = presence_stats(W)
            cls_r[rep] = cls
            single_r[rep] = single
            complement_r[rep] = complement
            presence_sum += W
        presence_prob = (presence_sum.astype(np.float64) / args.reps).astype(np.float32)
        print(f"{args.reps} replicates computed in {time.time() - t0:.1f}s")
        del P_work

        single_eq_mean = single_r.mean(axis=0)
        complement_eq_mean = complement_r.mean(axis=0)
        cls_eq_mean = cls_r.mean(axis=0)
        rho = {
            "single_obs": spearman(n_chars_total, single_obs),
            "complement_obs": spearman(n_chars_total, complement_obs),
            "single_eq": spearman(n_chars_total, single_eq_mean),
            "complement_eq": spearman(n_chars_total, complement_eq_mean),
        }

        # outputs
        np.savez_compressed(
            reports / f"equalized_presence_{tag}.npz",
            works=np.array(works),
            feature_ids=np.arange(n_features, dtype=np.int32),
            presence_prob=presence_prob,
            presence_obs=W_obs,
            budget_chars=np.int64(budget),
            reps=np.int64(args.reps),
            seed=np.int64(args.seed),
            n_chunks_total=n_chunks_total,
            n_chars_total=n_chars_total,
        )
        np.savez_compressed(
            reports / f"length_equalized_reps_{tag}.npz",
            single=single_r,
            complement=complement_r,
            classification=cls_r,
            chunks_used=chunks_used,
            works=np.array(works),
        )
        for wi, w in enumerate(works):
            s_lo, s_hi = np.percentile(single_r[:, wi], [2.5, 97.5])
            c_lo, c_hi = np.percentile(complement_r[:, wi], [2.5, 97.5])
            by_work_rows.append(
                (
                    model, layer, w, int(n_chunks_total[wi]), int(n_chars_total[wi]),
                    f"{mean_used[wi]:.4f}",
                    int(single_obs[wi]), f"{single_eq_mean[wi]:.4f}", f"{s_lo:.4f}", f"{s_hi:.4f}",
                    int(complement_obs[wi]), f"{complement_eq_mean[wi]:.4f}", f"{c_lo:.4f}", f"{c_hi:.4f}",
                )
            )
        classification_rows.append(
            (
                model, layer, budget, args.reps, args.seed,
                cls_obs[0], cls_obs[1], cls_obs[2],
                f"{cls_eq_mean[0]:.4f}", f"{cls_eq_mean[1]:.4f}", f"{cls_eq_mean[2]:.4f}",
                f"{rho['single_obs']:.6f}", f"{rho['complement_obs']:.6f}",
                f"{rho['single_eq']:.6f}", f"{rho['complement_eq']:.6f}",
            )
        )

        # stdout summary table (works sorted by total chars ascending)
        print(f"equalized mean: n_all={cls_eq_mean[0]:.2f} n_partial={cls_eq_mean[1]:.2f} n_none={cls_eq_mean[2]:.2f}")
        print(f"{'work':<5} {'n_chars':>8} {'chunks':>6} {'used':>7} {'single_obs':>10} {'single_eq':>10} {'compl_obs':>9} {'compl_eq':>9}")
        for wi in order_by_chars:
            print(
                f"{works[wi]:<5} {int(n_chars_total[wi]):>8} {int(n_chunks_total[wi]):>6} {mean_used[wi]:>7.2f} "
                f"{int(single_obs[wi]):>10} {single_eq_mean[wi]:>10.2f} {int(complement_obs[wi]):>9} {complement_eq_mean[wi]:>9.2f}"
            )
        print(
            f"Spearman rho(n_chars, single): obs={rho['single_obs']:+.4f} -> eq={rho['single_eq']:+.4f}; "
            f"rho(n_chars, complement): obs={rho['complement_obs']:+.4f} -> eq={rho['complement_eq']:+.4f}"
        )
        print(f"layer done in {time.time() - t_layer:.1f}s")

    write_csv(
        reports / "length_equalized_by_work.csv",
        (
            "model", "layer", "work", "n_chunks_total", "n_chars_total", "mean_chunks_used",
            "single_obs", "single_eq_mean", "single_eq_p2_5", "single_eq_p97_5",
            "complement_obs", "complement_eq_mean", "complement_eq_p2_5", "complement_eq_p97_5",
        ),
        by_work_rows,
    )
    write_csv(
        reports / "length_equalized_classification.csv",
        (
            "model", "layer", "budget_chars", "reps", "seed",
            "n_all_obs", "n_partial_obs", "n_none_obs",
            "n_all_eq_mean", "n_partial_eq_mean", "n_none_eq_mean",
            "rho_chars_single_obs", "rho_chars_complement_obs",
            "rho_chars_single_eq", "rho_chars_complement_eq",
        ),
        classification_rows,
    )
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

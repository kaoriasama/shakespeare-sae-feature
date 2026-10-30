"""Sparsity of single-work SAE features (A).

For each layer, a feature is "single-work" when it fires (firing_rate > 0) in
at least one chunk of exactly one of the works.  For those features this
script counts in how many chunks of the whole corpus they fire at all, to show
that most of them fire in one chunk only (a sporadic firing rather than a
property of the work).

Two definitions of "single-work" are tabulated:
  * observed (all chunks):  P = firing_rate > 0 (chunks x features);
    W_obs[w, f] = P[c, f] is True for some chunk c of work w;
    S_obs = {f : W_obs[:, f].sum() == 1}.
    W_obs is asserted to be identical to `presence_obs` stored in
    reports/equalized_presence_{model}_L{layer}.npz.
  * length-equalized:  S_eq = {f : (presence_prob[:, f] >= threshold).sum() == 1}
    with `presence_prob` from the same npz (threshold: --presence-threshold).

For each set, n_fired_chunks[f] = P[:, f].sum() is bucketed into
==1, ==2, 3..5, >=6.  For S_obs the per-feature mean of firing_rate over the
chunks in which the feature fires is also taken and its median is reported.

Read-only inputs
  activations_v2/{model}_L{layer}_firing.npz        chunk_ids (n_chunks,), firing_rate (n_chunks, n_features)
  reports/equalized_presence_{model}_L{layer}.npz   works, presence_prob, presence_obs
  chunks/shakespeare_chunks_v2.csv                  chunk_id, folger_abbr

Output
  reports/single_work_feature_sparsity.csv          one row per layer

Only the standard library and numpy are used.  No random numbers.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_LAYERS = "g2_L4,g2_L11,g2_L22,g3_L9,g3_L17,g3_L29"

CSV_HEADER = (
    "model", "layer",
    "n_single_obs", "n_single_obs_fired_1chunk", "n_single_obs_fired_2",
    "n_single_obs_fired_3_5", "n_single_obs_fired_6plus", "share_single_obs_1chunk",
    "n_single_eq", "n_single_eq_fired_1chunk", "share_single_eq_1chunk",
    "median_fr_in_fired_chunk_single_obs",
)


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
    """Return (chunk_ids in file order, folger_abbr per chunk in file order)."""
    csv.field_size_limit(10**9)
    chunk_ids = []
    works_of_chunk = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        missing = {"chunk_id", "folger_abbr"} - set(reader.fieldnames)
        assert not missing, f"chunks CSV lacks columns: {sorted(missing)}"
        for row in reader:
            chunk_ids.append(row["chunk_id"])
            works_of_chunk.append(row["folger_abbr"])
    assert len(chunk_ids) == len(set(chunk_ids)), "duplicate chunk_id in chunks CSV"
    return chunk_ids, works_of_chunk


def load_firing_rate(npz_path, csv_chunk_ids):
    """Return firing_rate (chunks x features, float32) after checking that the
    npz chunk order is identical to the CSV order."""
    with np.load(npz_path, allow_pickle=False) as data:
        npz_ids = [str(item) for item in data["chunk_ids"].tolist()]
        assert npz_ids == csv_chunk_ids, (
            f"{npz_path.name}: chunk_ids differ from CSV chunk_id (order included)"
        )
        firing_rate = data["firing_rate"]
    assert firing_rate.ndim == 2 and firing_rate.shape[0] == len(csv_chunk_ids)
    assert np.isfinite(firing_rate).all(), f"{npz_path.name}: non-finite values"
    assert float(firing_rate.min()) >= 0.0, f"{npz_path.name}: negative firing_rate"
    return firing_rate


def load_equalized_presence(npz_path, works, n_features):
    """Return (presence_prob, presence_obs) after checking works and shapes."""
    with np.load(npz_path, allow_pickle=False) as data:
        npz_works = [str(item) for item in data["works"].tolist()]
        assert npz_works == works, f"{npz_path.name}: works differ from sorted CSV folger_abbr"
        presence_prob = data["presence_prob"]
        presence_obs = data["presence_obs"]
    expected = (len(works), n_features)
    assert presence_prob.shape == expected, f"{npz_path.name}: presence_prob shape {presence_prob.shape} != {expected}"
    assert presence_obs.shape == expected, f"{npz_path.name}: presence_obs shape {presence_obs.shape} != {expected}"
    assert presence_obs.dtype == np.bool_, f"{npz_path.name}: presence_obs dtype {presence_obs.dtype}"
    return presence_prob, presence_obs


def bucket_counts(n_fired):
    """Counts of n_fired == 1, == 2, 3..5, >= 6 (n_fired must be >= 1)."""
    n_fired = np.asarray(n_fired)
    assert n_fired.size == 0 or int(n_fired.min()) >= 1, "a selected feature fires in no chunk"
    n1 = int(np.count_nonzero(n_fired == 1))
    n2 = int(np.count_nonzero(n_fired == 2))
    n3_5 = int(np.count_nonzero((n_fired >= 3) & (n_fired <= 5)))
    n6plus = int(np.count_nonzero(n_fired >= 6))
    assert n1 + n2 + n3_5 + n6plus == n_fired.size
    return n1, n2, n3_5, n6plus


def share(numerator, denominator, digits):
    return "" if denominator == 0 else f"{numerator / denominator:.{digits}f}"


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--presence-threshold", type=float, default=0.5,
                        help="a work 'has' a feature when presence_prob >= this value (length-equalized definition)")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    reports = root / "reports"
    layers = parse_layers(args.layers)
    threshold = float(args.presence_threshold)
    print(f"root      = {root}")
    print(f"threshold = {threshold} (presence_prob >= threshold)")
    print(f"layers    = {', '.join(f'{m}_L{l}' for m, l in layers)}")

    # ---- chunks ---------------------------------------------------------- #
    csv_chunk_ids, works_of_chunk = load_chunks(root / "chunks" / "shakespeare_chunks_v2.csv")
    works = sorted(set(works_of_chunk))
    n_works = len(works)
    work_index = {w: i for i, w in enumerate(works)}
    rows_of_work = [[] for _ in works]
    for row_idx, work in enumerate(works_of_chunk):
        rows_of_work[work_index[work]].append(row_idx)
    rows_of_work = [np.array(rows, dtype=np.int64) for rows in rows_of_work]
    print(f"chunks    = {len(csv_chunk_ids)}, works = {n_works}")
    print(f"works     = {' '.join(works)}")

    # ---- per layer ------------------------------------------------------- #
    out_rows = []
    table = []
    for model, layer in layers:
        tag = f"{model}_L{layer}"
        t_layer = time.time()
        print(f"\n===== {tag} =====")
        firing_rate = load_firing_rate(root / "activations_v2" / f"{tag}_firing.npz", csv_chunk_ids)
        n_features = firing_rate.shape[1]
        P = firing_rate > 0
        n_fired_chunks = P.sum(axis=0, dtype=np.int64)                      # (n_features,)
        W_obs = np.stack([P[rows].any(axis=0) for rows in rows_of_work])     # (n_works, n_features)

        presence_prob, presence_obs = load_equalized_presence(
            reports / f"equalized_presence_{tag}.npz", works, n_features
        )
        identical = bool(np.array_equal(W_obs, presence_obs))
        print(f"check: W_obs identical to presence_obs in equalized_presence_{tag}.npz: {identical}")
        assert identical, f"{tag}: W_obs differs from presence_obs"

        # observed single-work features
        n_works_obs = W_obs.sum(axis=0, dtype=np.int64)
        S_obs = np.flatnonzero(n_works_obs == 1)
        obs_n1, obs_n2, obs_n3_5, obs_n6 = bucket_counts(n_fired_chunks[S_obs])

        # length-equalized single-work features
        n_works_eq = (presence_prob >= threshold).sum(axis=0, dtype=np.int64)
        S_eq = np.flatnonzero(n_works_eq == 1)
        eq_n1, eq_n2, eq_n3_5, eq_n6 = bucket_counts(n_fired_chunks[S_eq])

        # per-feature mean firing_rate over the chunks in which it fires (S_obs)
        # firing_rate is 0 exactly where P is False (firing_rate >= 0 asserted),
        # so the mean over fired chunks is the column sum divided by n_fired_chunks.
        if S_obs.size:
            mean_fr_fired = firing_rate[:, S_obs].astype(np.float64).sum(axis=0) / n_fired_chunks[S_obs]
            median_fr = f"{float(np.median(mean_fr_fired)):.6f}"
        else:
            median_fr = ""
        del firing_rate, P

        n_single_obs = int(S_obs.size)
        n_single_eq = int(S_eq.size)
        out_rows.append(
            (
                model, layer,
                n_single_obs, obs_n1, obs_n2, obs_n3_5, obs_n6, share(obs_n1, n_single_obs, 4),
                n_single_eq, eq_n1, share(eq_n1, n_single_eq, 4),
                median_fr,
            )
        )
        table.append((tag, n_single_obs, obs_n1, obs_n2, obs_n3_5, obs_n6, share(obs_n1, n_single_obs, 4),
                      n_single_eq, eq_n1, eq_n2, eq_n3_5, eq_n6, share(eq_n1, n_single_eq, 4), median_fr))
        print(f"observed  single-work: n={n_single_obs} fired in 1 chunk={obs_n1} 2={obs_n2} 3-5={obs_n3_5} >=6={obs_n6} share_1chunk={share(obs_n1, n_single_obs, 4)}")
        print(f"equalized single-work: n={n_single_eq} fired in 1 chunk={eq_n1} 2={eq_n2} 3-5={eq_n3_5} >=6={eq_n6} share_1chunk={share(eq_n1, n_single_eq, 4)}")
        print(f"median over S_obs of mean firing_rate in fired chunks = {median_fr}")
        print(f"layer done in {time.time() - t_layer:.1f}s")

    write_csv(reports / "single_work_feature_sparsity.csv", CSV_HEADER, out_rows)

    # ---- stdout summary table ------------------------------------------- #
    print()
    print(f"{'layer':<7} {'n_obs':>6} {'=1':>6} {'=2':>5} {'3-5':>5} {'>=6':>5} {'share1':>7} "
          f"{'n_eq':>6} {'=1':>6} {'=2':>5} {'3-5':>5} {'>=6':>5} {'share1':>7} {'median_fr':>10}")
    for row in table:
        print(f"{row[0]:<7} {row[1]:>6} {row[2]:>6} {row[3]:>5} {row[4]:>5} {row[5]:>5} {row[6]:>7} "
              f"{row[7]:>6} {row[8]:>6} {row[9]:>5} {row[10]:>5} {row[11]:>5} {row[12]:>7} {row[13]:>10}")
    print(f"\nwritten: {reports / 'single_work_feature_sparsity.csv'}")
    print(f"total elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

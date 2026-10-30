"""Step 1: how close are length-equalized firing-work sets to the classical reference groups?

For every layer, S_f = {works w : presence_prob[w, f] >= threshold} is the
length-equalized firing-work set of feature f (presence_prob comes from
reports/equalized_presence_{model}_L{layer}.npz written by
length_equalized_patterns.py).  Features are kept when, counted over all 38
works, 1 <= |S_f| <= 37 and min_size <= |S_f| <= max_size.

For every reference group T with universe U (all_works minus the universe's
exclusions), the Jaccard index J_f = |S_U ∩ T| / |S_U ∪ T| with S_U = S_f ∩ U
is computed (J = 0 when the union is empty).  Observed statistics are
n_obs(theta) = #{f : J_f >= theta} for each theta and max_J_obs, plus the five
features with the highest J.

Null baseline: for each kept feature the size k_f = |S_U(f)| is retained and
S_U(f) is replaced by a uniformly random k_f-subset of U.  The number of works
x shared by a uniform random k_f-subset of U and the fixed group T follows
Hypergeometric(ngood=|T|, nbad=|U|-|T|, nsample=k_f), so x is drawn directly
with rng.hypergeometric and J_null = x / (k_f + |T| - x) (Jaccard of two sets
of sizes k_f and |T| sharing x elements; J_null = 0 when k_f = 0).  This is
done null_reps times, drawing independently for every kept feature in each
replicate, with a single RNG consumed in a fixed order (layers, then groups).
For each statistic: null_mean, null_p95 (95th percentile) and
p_value = (#{null >= observed} + 1) / (null_reps + 1).

Outputs (reports/)
  genre_proximity_step1.csv        one row per layer x group
  genre_proximity_step1_best.csv   top-5 features per layer x group

Only the standard library and numpy are used.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_LAYERS = "g2_L4,g2_L11,g2_L22,g3_L9,g3_L17,g3_L29"
NULL_BLOCK = 1000  # replicates drawn per numpy call (bounds memory; stream order unchanged)


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


def parse_thetas(text):
    thetas = []
    for token in text.split(","):
        token = token.strip()
        if token:
            thetas.append((token, float(token)))
    if not thetas:
        raise ValueError("no thetas given")
    return thetas


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def load_work_genres(csv_path):
    """work -> genre label from the chunks CSV (asserts one label per work)."""
    csv.field_size_limit(10**9)
    work_genres = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            work_genres.setdefault(row["folger_abbr"], set()).add(row["genre"])
    multi = {w: sorted(g) for w, g in work_genres.items() if len(g) != 1}
    assert not multi, f"works with more than one genre label in chunks CSV: {multi}"
    return {w: next(iter(g)) for w, g in work_genres.items()}


def validate_groups(spec, works, work_genres):
    """Structural checks of reference_groups.json (assert) and comparison of the
    Folio groups with the chunks CSV genre column (report only)."""
    all_works = list(spec["all_works"])
    print("--- reference groups validation ---")
    print(f"all_works in JSON: {len(all_works)}; equals the npz work list (same order): {all_works == works}")
    assert set(all_works) == set(works), "reference_groups.json all_works differs from the data works"
    assert len(set(all_works)) == len(all_works), "duplicate entries in all_works"
    work_set = set(all_works)
    universes = {}
    for name, uni in spec["universes"].items():
        exclude = list(uni["exclude"])
        assert set(exclude) <= work_set, f"universe {name}: exclusions not in all_works: {sorted(set(exclude) - work_set)}"
        members = [w for w in all_works if w not in set(exclude)]
        universes[name] = {"exclude": exclude, "members": members}
        print(f"universe {name}: size {len(members)} (excluded: {' '.join(exclude)})")
    by_universe = {name: [] for name in universes}
    for gname, grp in spec["groups"].items():
        members = list(grp["members"])
        uni = grp["universe"]
        assert uni in universes, f"group {gname}: unknown universe {uni}"
        assert len(set(members)) == len(members), f"group {gname}: duplicate members"
        assert set(members) <= work_set, f"group {gname}: members not in all_works: {sorted(set(members) - work_set)}"
        outside = set(members) - set(universes[uni]["members"])
        assert not outside, f"group {gname}: members outside universe {uni}: {sorted(outside)}"
        by_universe[uni].append(gname)
        print(f"group {gname:<14} universe={uni:<8} size={len(members):>2}  members: {' '.join(members)}")
    for uni, gnames in by_universe.items():
        sets = [set(spec["groups"][g]["members"]) for g in gnames]
        total = sum(len(s) for s in sets)
        union = set().union(*sets) if sets else set()
        disjoint = total == len(union)
        covers = union == set(universes[uni]["members"])
        print(f"universe {uni}: groups {gnames} sizes {[len(s) for s in sets]} sum={total}; pairwise disjoint={disjoint}; cover universe={covers}")

    # Folio groups vs chunks CSV genre column (report, do not fail)
    labels = sorted(set(work_genres.values()))
    print(f"chunks CSV genre labels: {labels}")
    for label in labels:
        ws = sorted(w for w, g in work_genres.items() if g == label)
        print(f"  CSV genre '{label}': {len(ws)} works: {' '.join(ws)}")
    for gname, grp in spec["groups"].items():
        if not gname.startswith("folio_"):
            continue
        label = gname[len("folio_"):]
        members = set(grp["members"])
        if label not in set(work_genres.values()):
            print(f"  {gname}: no CSV label '{label}' found; cannot compare automatically")
            continue
        csv_set = {w for w, g in work_genres.items() if g == label}
        same = csv_set == members
        print(f"  {gname} ({len(members)}) vs CSV genre '{label}' ({len(csv_set)}): identical={same}"
              + ("" if same else f"; only in JSON: {sorted(members - csv_set)}; only in CSV: {sorted(csv_set - members)}"))
    print("--- end validation ---")
    return universes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--null-reps", type=int, default=10000)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=30)
    parser.add_argument("--thetas", default="0.6,0.7,0.8")
    parser.add_argument("--groups-file", type=Path, default=None, help="default: <root>/scripts/analysis/reference_groups.json")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    reports = root / "reports"
    groups_file = args.groups_file if args.groups_file is not None else root / "scripts" / "analysis" / "reference_groups.json"
    layers = parse_layers(args.layers)
    thetas = parse_thetas(args.thetas)
    theta_show = "0.7" if "0.7" in [t for t, _ in thetas] else thetas[len(thetas) // 2][0]
    print(f"root = {root}")
    print(f"seed = {args.seed}, null_reps = {args.null_reps}, presence_threshold = {args.presence_threshold}, "
          f"size range = [{args.min_size}, {args.max_size}], thetas = {[t for t, _ in thetas]}")
    print(f"groups file = {groups_file}")

    with groups_file.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    # work list from the first layer's npz (all layers are checked against it)
    first_tag = f"{layers[0][0]}_L{layers[0][1]}"
    with np.load(reports / f"equalized_presence_{first_tag}.npz", allow_pickle=False) as data:
        works = [str(w) for w in data["works"].tolist()]
    n_works = len(works)
    work_index = {w: i for i, w in enumerate(works)}

    work_genres = load_work_genres(root / "chunks" / "shakespeare_chunks_v2.csv")
    universes = validate_groups(spec, works, work_genres)

    group_names = list(spec["groups"].keys())
    group_masks = {}
    for gname in group_names:
        grp = spec["groups"][gname]
        u_mask = np.ones(n_works, dtype=bool)
        for w in universes[grp["universe"]]["exclude"]:
            u_mask[work_index[w]] = False
        t_mask = np.zeros(n_works, dtype=bool)
        for w in grp["members"]:
            t_mask[work_index[w]] = True
        group_masks[gname] = (u_mask, t_mask, grp["universe"])

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    best_rows = []
    table = {}

    for model, layer in layers:
        tag = f"{model}_L{layer}"
        t_layer = time.time()
        print(f"\n===== {tag} =====")
        with np.load(reports / f"equalized_presence_{tag}.npz", allow_pickle=False) as data:
            layer_works = [str(w) for w in data["works"].tolist()]
            assert layer_works == works, f"{tag}: work order differs from {first_tag}"
            presence_prob = data["presence_prob"]
            feature_ids = data["feature_ids"].astype(np.int64)
            print(f"presence npz: budget_chars={int(data['budget_chars'])} reps={int(data['reps'])} seed={int(data['seed'])}")
        S = presence_prob >= args.presence_threshold  # (n_works, n_features)
        size38 = S.sum(axis=0)
        selected = (size38 >= 1) & (size38 <= n_works - 1) & (size38 >= args.min_size) & (size38 <= args.max_size)
        n_sel = int(selected.sum())
        sel_ids = feature_ids[selected]
        S_sel = S[:, selected]
        size38_sel = size38[selected]
        print(f"features with 1..{n_works - 1} works at threshold: {int(((size38 >= 1) & (size38 <= n_works - 1)).sum())}; "
              f"selected (size in [{args.min_size}, {args.max_size}]): {n_sel}")

        for gname in group_names:
            u_mask, t_mask, uni = group_masks[gname]
            U_size = int(u_mask.sum())
            T_size = int(t_mask.sum())
            S_U = S_sel & u_mask[:, None]
            k = S_U.sum(axis=0).astype(np.int64)                      # |S_U|
            common = (S_U & t_mask[:, None]).sum(axis=0).astype(np.int64)  # |S_U ∩ T|
            union = k + T_size - common                                # |S_U ∪ T|
            J = np.where(union > 0, common / np.maximum(union, 1), 0.0)
            n_obs = {tok: int((J >= val).sum()) for tok, val in thetas}
            maxJ_obs = float(J.max()) if n_sel > 0 else 0.0

            # top-5 features (J descending, feature_id ascending)
            order = sorted(range(n_sel), key=lambda i: (-J[i], int(sel_ids[i])))[:5]
            for rank, i in enumerate(order, start=1):
                col = S_sel[:, i]
                in_U = col & u_mask
                missing = [w for wi, w in enumerate(works) if t_mask[wi] and not in_U[wi]]
                extra = [w for wi, w in enumerate(works) if in_U[wi] and not t_mask[wi]]
                excluded_present = [w for wi, w in enumerate(works) if col[wi] and not u_mask[wi]]
                best_rows.append(
                    (
                        model, layer, gname, rank, int(sel_ids[i]), f"{J[i]:.6f}", int(k[i]), int(common[i]),
                        " ".join(missing), " ".join(extra), " ".join(excluded_present), int(size38_sel[i]),
                    )
                )

            # null baseline (see module docstring for the hypergeometric equivalence)
            n_null = {tok: np.zeros(args.null_reps, dtype=np.int64) for tok, _ in thetas}
            maxJ_null = np.zeros(args.null_reps, dtype=np.float64)
            pos = k > 0
            k_pos = k[pos]
            n_pos = int(pos.sum())
            for start in range(0, args.null_reps, NULL_BLOCK):
                b = min(NULL_BLOCK, args.null_reps - start)
                if n_pos == 0:
                    continue
                x = rng.hypergeometric(ngood=T_size, nbad=U_size - T_size, nsample=k_pos, size=(b, n_pos))
                J_null = x / (k_pos[None, :] + T_size - x)
                for tok, val in thetas:
                    n_null[tok][start:start + b] = (J_null >= val).sum(axis=1)
                maxJ_null[start:start + b] = J_null.max(axis=1)

            def null_summary(obs, null):
                mean = float(np.mean(null))
                p95 = float(np.percentile(null, 95))
                p = (int(np.count_nonzero(null >= obs)) + 1) / (args.null_reps + 1)
                return mean, p95, p

            row = [model, layer, gname, U_size, T_size, n_sel, f"{float(k.mean()) if n_sel else 0.0:.4f}"]
            cells = {}
            for tok, _ in thetas:
                mean, p95, p = null_summary(n_obs[tok], n_null[tok])
                row += [n_obs[tok], f"{mean:.4f}", f"{p95:.4f}", f"{p:.6f}"]
                cells[tok] = (n_obs[tok], p95, p)
            mean, p95, p = null_summary(maxJ_obs, maxJ_null)
            row += [f"{maxJ_obs:.6f}", f"{mean:.6f}", f"{p95:.6f}", f"{p:.6f}"]
            summary_rows.append(tuple(row))
            table[(tag, gname)] = cells[theta_show]
            print(f"  {gname:<14} |U|={U_size:>2} |T|={T_size:>2} mean|S_U|={float(k.mean()) if n_sel else 0.0:>6.2f}  "
                  + "  ".join(f"n>={tok}: {n_obs[tok]:>3} (null p95 {cells[tok][1]:>5.1f}, p={cells[tok][2]:.4f})" for tok, _ in thetas)
                  + f"  maxJ={maxJ_obs:.3f} (null mean {mean:.3f}, p95 {p95:.3f}, p={p:.4f})")
        print(f"layer done in {time.time() - t_layer:.1f}s")

    header = ["model", "layer", "group", "universe_size", "group_size", "n_features_selected", "mean_size_in_universe"]
    for tok, _ in thetas:
        header += [f"n_obs_{tok}", f"null_mean_{tok}", f"null_p95_{tok}", f"p_{tok}"]
    header += ["maxJ_obs", "null_maxJ_mean", "null_maxJ_p95", "p_maxJ"]
    write_csv(reports / "genre_proximity_step1.csv", header, summary_rows)
    write_csv(
        reports / "genre_proximity_step1_best.csv",
        ("model", "layer", "group", "rank", "feature_id", "jaccard", "size_in_universe", "n_common",
         "missing_works", "extra_works", "excluded_members_present", "size_all38"),
        best_rows,
    )

    # layer x group table: n_obs(theta_show) / null_p95(theta_show) / p
    print(f"\n=== layer x group: n_obs(J>={theta_show}) / null_p95 / p_value ===")
    print(f"{'layer':<8}" + "".join(f"{g:>22}" for g in group_names))
    for model, layer in layers:
        tag = f"{model}_L{layer}"
        line = f"{tag:<8}"
        for gname in group_names:
            n_obs, p95, p = table[(tag, gname)]
            line += f"{n_obs:>8}/{p95:>5.1f}/{p:.4f}"
        print(line)
    print(f"\ntotal elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

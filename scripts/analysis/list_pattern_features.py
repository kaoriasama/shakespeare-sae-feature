"""B: list noteworthy SAE features with their strongest text passages and auxiliary statistics.

Three categories are selected per layer (S_f = {works : presence_prob >= threshold}, counted over
all 38 works):

  genre_near  rows of reports/genre_proximity_step1_best.csv whose jaccard >= --genre-min-j
              (target = reference group name)
  silence     features whose S_f is exactly "all 38 works except w"; targets w are the three works
              with the largest complement_eq_mean in reports/length_equalized_by_work.csv for the
              layer, plus the constant additions {Mac, Wiv} required by the execution contract;
              ranked by the mean firing rate over all 4600 chunks (descending), top --cap
  unique      features whose S_f is exactly {w}; targets w are the three works with the largest
              single_eq_mean plus the contract constants {Mac, Ham}; ranked by the mean firing rate
              over the chunks of w (descending), top --cap

Ties in every ranking are broken by feature_id ascending.  For every selected row the script
writes the firing-work sets (equalized and observed), firing-rate statistics, the three chunks
with the highest firing rate (ties: earlier CSV row first), Spearman correlations with the verse
ratio / stage-direction counts / speaker changes, and the Neuronpedia auto-interpretation.

Outputs (reports/)
  pattern_feature_ids.csv            one row per feature x category x target
  pattern_feature_excerpts.txt       the text_content of the top-3 chunks, verbatim
  pattern_feature_neuronpedia.json   raw Neuronpedia responses keyed "{model}_L{layer}_f{feature_id}"
                                     (plus "_fetch_log": per-key fetch status, "_fetch_meta")

Neuronpedia: public API, no key.  --neuronpedia fetch (default) queries the API, cache reads the
saved JSON only, off leaves the NP columns blank.

Only the standard library and numpy are used.
"""

import argparse
import csv
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

LAYERS = [("g2", 4), ("g2", 11), ("g2", 22), ("g3", 9), ("g3", 17), ("g3", 29)]
CATEGORIES = ["genre_near", "silence", "unique"]
# Constant additions to the target works, fixed by the execution contract (plan v3-1, B):
# silence targets always include Mac and Wiv; unique targets always include Mac and Ham.
CONTRACT_SILENCE_EXTRA = ["Mac", "Wiv"]
CONTRACT_UNIQUE_EXTRA = ["Mac", "Ham"]
STAGE_COLUMNS = ["n_stage_entrance", "n_stage_exit", "n_stage_delivery",
                 "n_stage_business", "n_stage_modifier", "n_stage_other"]
NP_MODEL = {"g2": "gemma-2-2b", "g3": "gemma-3-4b"}
NP_SOURCE = {"g2": "{layer}-gemmascope-res-16k", "g3": "{layer}-gemmascope-2-res-16k"}
NP_URL = "https://www.neuronpedia.org/api/feature/{np_model}/{np_source}/{feature_id}"
NP_USER_AGENT = "jinmoncom2026-analysis/0.1 (python urllib)"
NP_TIMEOUT = 20.0
NP_SLEEP = 0.5
NP_ATTEMPTS = 2          # attempts per feature (safety net against transient network errors)
NP_ABORT_AFTER = 3       # abort fetching when the first N requests fail consecutively
NP_TOP_ACTS = 5
NP_TOP_LOGITS = 10

CSV_HEADER = [
    "model", "layer", "category", "target", "rank_in_category", "n_candidates_in_category",
    "feature_id", "jaccard", "n_works_eq", "works_eq", "n_works_obs", "works_obs",
    "mean_fr_all", "mean_fr_fired", "n_chunks_fired", "max_fr",
    "top1_chunk_id", "top1_work", "top1_act_scene", "top1_fr",
    "top2_chunk_id", "top2_work", "top2_act_scene", "top2_fr",
    "top3_chunk_id", "top3_work", "top3_act_scene", "top3_fr",
    "rho_verse", "n_verse_used", "mean_fr_prose_only", "mean_fr_verse_only", "verse_over_prose",
    "rho_stage", "rho_speaker_changes",
    "np_status", "np_explanation", "np_pos_tokens", "np_neg_tokens", "np_frac_nonzero", "np_top_act_tokens",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_chunks(csv_path):
    csv.field_size_limit(10**9)
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chunk_id", "folger_abbr", "act", "scene", "verse_ratio", "n_speaker_changes",
                    "text_content", *STAGE_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        assert not missing, f"chunks CSV lacks columns: {sorted(missing)}"
        for row in reader:
            rows.append(row)
    ids = [r["chunk_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk_id in chunks CSV"
    return rows


def rank_average(values):
    """1-based ranks, ties replaced by the average rank."""
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
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation (average ranks for ties); nan when either side is constant."""
    rx = rank_average(x)
    ry = rank_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def one_line(text):
    """Keep CSV/TXT headings on one line."""
    return str(text).replace("\r", "\\r").replace("\n", "\\n")


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_by_work(csv_path):
    """(model, layer) -> work -> (single_eq_mean, complement_eq_mean)."""
    table = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], int(row["layer"]))
            table.setdefault(key, {})[row["work"]] = (float(row["single_eq_mean"]), float(row["complement_eq_mean"]))
    return table


def read_best(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def top_works(values_by_work, key_index, n_top, extra):
    """Top-n works by the chosen statistic (descending; ties by work name), union with extras, sorted."""
    ranked = sorted(values_by_work.items(), key=lambda kv: (-kv[1][key_index], kv[0]))
    chosen = [w for w, _ in ranked[:n_top]]
    for w in extra:
        assert w in values_by_work, f"contract constant work {w} not found in length_equalized_by_work.csv"
    return sorted(set(chosen) | set(extra)), chosen


# --------------------------------------------------------------------------- #
# Neuronpedia
# --------------------------------------------------------------------------- #
def np_key(model, layer, feature_id):
    return f"{model}_L{layer}_f{feature_id}"


def np_url(model, layer, feature_id):
    return NP_URL.format(np_model=NP_MODEL[model], np_source=NP_SOURCE[model].format(layer=layer), feature_id=feature_id)


class _KeepAliveHTTPSConnection(http.client.HTTPSConnection):
    """urllib.request unconditionally adds 'Connection: close' to every request.  The server then
    closes the connection right after the response, and the HTTPS proxy of the sandboxed
    environment drops the not-yet-forwarded tail of chunked responses (bodies of 60-90 KB came
    back truncated at ~64 KiB in about half of the requests).  Removing that header keeps the
    connection open until this client has read the terminating chunk; the client closes it."""

    def request(self, method, url, body=None, headers=None, *, encode_chunked=False):
        headers = {k: v for k, v in (headers or {}).items() if k.lower() != "connection"}
        return super().request(method, url, body, headers, encode_chunked=encode_chunked)


class _KeepAliveHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_KeepAliveHTTPSConnection, req, context=self._context, check_hostname=self._check_hostname)


_NP_OPENER = urllib.request.build_opener(_KeepAliveHTTPSHandler())


def np_fetch_one(url):
    """GET the feature JSON with urllib.request (keep-alive opener above).  Returns (dict or None,
    status).  The body is read in 16 KiB pieces; a truncated or unparsable body is retried once."""
    status = "error_unknown"
    for attempt in range(NP_ATTEMPTS):
        if attempt > 0:
            time.sleep(NP_SLEEP)
        req = urllib.request.Request(url, headers={"User-Agent": NP_USER_AGENT, "Accept": "application/json"})
        try:
            with _NP_OPENER.open(req, timeout=NP_TIMEOUT) as resp:
                code = resp.status
                parts = []
                try:
                    while True:
                        piece = resp.read(16384)
                        if not piece:
                            break
                        parts.append(piece)
                except http.client.IncompleteRead as error:
                    parts.append(error.partial)
                body = b"".join(parts)
        except urllib.error.HTTPError as error:
            return None, f"http_{error.code}"
        except Exception as error:  # URLError, socket.timeout, ...
            status = f"error_{type(error).__name__}"
            continue
        if code != 200:
            return None, f"http_{code}"
        try:
            return json.loads(body.decode("utf-8")), "ok"
        except ValueError:
            status = "error_truncated_json"
            continue
    return None, status


def np_extract(data):
    """Pull the reported fields out of a raw feature response."""
    explanation = ""
    explanations = data.get("explanations") or []
    if explanations:
        explanation = explanations[0].get("description") or ""
    pos = data.get("pos_str") or []
    neg = data.get("neg_str") or []
    frac = data.get("frac_nonzero")
    acts = data.get("activations") or []
    # activations are returned sorted by maxValue descending; sort explicitly (stable) to be safe
    acts_sorted = sorted(acts, key=lambda a: -(a.get("maxValue") or 0.0))
    top_tokens = []
    for act in acts_sorted[:NP_TOP_ACTS]:
        idx = act.get("maxValueTokenIndex")
        tokens = act.get("tokens") or []
        if idx is not None and 0 <= idx < len(tokens):
            top_tokens.append(tokens[idx])
    return {
        "np_explanation": one_line(explanation),
        "np_pos_tokens": one_line("|".join(str(t) for t in pos[:NP_TOP_LOGITS])),
        "np_neg_tokens": one_line("|".join(str(t) for t in neg[:NP_TOP_LOGITS])),
        "np_frac_nonzero": "" if frac is None else str(frac),
        "np_top_act_tokens": one_line("|".join(str(t) for t in top_tokens)),
    }


NP_EMPTY = {"np_status": "", "np_explanation": "", "np_pos_tokens": "", "np_neg_tokens": "",
            "np_frac_nonzero": "", "np_top_act_tokens": ""}


def neuronpedia_lookup(keys_in_order, mode, cache_path):
    """keys_in_order: list of (model, layer, feature_id) without duplicates, in output order.
    Returns key -> NP column dict, plus a summary dict."""
    summary = {"mode": mode, "n_ok": 0, "n_fail": 0, "statuses": {}, "aborted": False}
    results = {}
    if mode == "off":
        for key in keys_in_order:
            results[np_key(*key)] = dict(NP_EMPTY)
        return results, summary

    if mode == "cache":
        assert cache_path.exists(), f"--neuronpedia cache but {cache_path} does not exist"
        with cache_path.open("r", encoding="utf-8") as handle:
            store = json.load(handle)
        fetch_log = store.get("_fetch_log", {})
        for key in keys_in_order:
            k = np_key(*key)
            if k in store:
                cols = {"np_status": "ok", **np_extract(store[k])}
            else:
                cols = dict(NP_EMPTY)
                cols["np_status"] = fetch_log.get(k, "cache_miss")
            results[k] = cols
            summary["statuses"][cols["np_status"]] = summary["statuses"].get(cols["np_status"], 0) + 1
        summary["n_ok"] = summary["statuses"].get("ok", 0)
        summary["n_fail"] = len(keys_in_order) - summary["n_ok"]
        return results, summary

    assert mode == "fetch"
    store = {}
    fetch_log = {}
    consecutive_fail_from_start = 0
    aborted = False
    for i, key in enumerate(keys_in_order):
        k = np_key(*key)
        if aborted:
            cols = dict(NP_EMPTY)
            cols["np_status"] = "not_fetched_aborted"
        else:
            if i > 0:
                time.sleep(NP_SLEEP)
            url = np_url(*key)
            data, status = np_fetch_one(url)
            if data is not None:
                store[k] = data
                cols = {"np_status": "ok", **np_extract(data)}
                consecutive_fail_from_start = -1  # a success happened; the abort rule no longer applies
            else:
                cols = dict(NP_EMPTY)
                cols["np_status"] = status
                if consecutive_fail_from_start >= 0:
                    consecutive_fail_from_start += 1
                    if consecutive_fail_from_start >= NP_ABORT_AFTER:
                        aborted = True
            print(f"  NP {k}: {cols['np_status']}" + (f" | {cols['np_explanation'][:70]}" if cols["np_status"] == "ok" else ""))
        fetch_log[k] = cols["np_status"]
        results[k] = cols
        summary["statuses"][cols["np_status"]] = summary["statuses"].get(cols["np_status"], 0) + 1
    summary["aborted"] = aborted
    summary["n_ok"] = summary["statuses"].get("ok", 0)
    summary["n_fail"] = len(keys_in_order) - summary["n_ok"]
    store["_fetch_log"] = fetch_log
    store["_fetch_meta"] = {
        "url_template": NP_URL, "np_model": NP_MODEL, "np_source": NP_SOURCE,
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aborted_after_consecutive_failures": aborted,
        "fields_used": {
            "np_explanation": "explanations[0].description",
            "np_pos_tokens": f"pos_str[:{NP_TOP_LOGITS}]",
            "np_neg_tokens": f"neg_str[:{NP_TOP_LOGITS}]",
            "np_frac_nonzero": "frac_nonzero",
            "np_top_act_tokens": f"activations (sorted by maxValue desc)[:{NP_TOP_ACTS}] -> tokens[maxValueTokenIndex]",
        },
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=1)
    return results, summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--cap", type=int, default=10)
    parser.add_argument("--genre-min-j", type=float, default=0.6)
    parser.add_argument("--neuronpedia", choices=["fetch", "cache", "off"], default="fetch")
    parser.add_argument("--neuronpedia-cache", type=Path, default=None,
                        help="default: <root>/reports/pattern_feature_neuronpedia.json")
    args = parser.parse_args()

    t_start = time.time()
    root = args.root.resolve()
    reports = root / "reports"
    cache_path = args.neuronpedia_cache if args.neuronpedia_cache is not None else reports / "pattern_feature_neuronpedia.json"
    print(f"root = {root}")
    print(f"presence_threshold = {args.presence_threshold}, cap = {args.cap}, genre_min_j = {args.genre_min_j}, "
          f"neuronpedia = {args.neuronpedia}, cache = {cache_path}")

    # ---- chunks ---------------------------------------------------------- #
    chunk_rows = load_chunks(root / "chunks" / "shakespeare_chunks_v2.csv")
    n_chunks = len(chunk_rows)
    csv_chunk_ids = [r["chunk_id"] for r in chunk_rows]
    chunk_work = [r["folger_abbr"] for r in chunk_rows]
    works = sorted(set(chunk_work))
    n_works = len(works)
    work_index = {w: i for i, w in enumerate(works)}
    rows_of_work = {w: np.array([i for i, cw in enumerate(chunk_work) if cw == w], dtype=np.int64) for w in works}
    verse_raw = [r["verse_ratio"].strip() for r in chunk_rows]
    verse_mask = np.array([v != "" for v in verse_raw], dtype=bool)
    verse_ratio = np.array([float(v) if v != "" else np.nan for v in verse_raw], dtype=np.float64)
    prose_mask = verse_mask & (verse_ratio == 0.0)
    verse_only_mask = verse_mask & (verse_ratio == 1.0)
    stage_total = np.array([sum(int(r[c]) for c in STAGE_COLUMNS) for r in chunk_rows], dtype=np.float64)
    speaker_changes = np.array([int(r["n_speaker_changes"]) for r in chunk_rows], dtype=np.float64)
    print(f"chunks = {n_chunks}, works = {n_works}; verse_ratio given for {int(verse_mask.sum())} chunks "
          f"(prose-only {int(prose_mask.sum())}, verse-only {int(verse_only_mask.sum())})")

    by_work = read_by_work(reports / "length_equalized_by_work.csv")
    best_rows = read_best(reports / "genre_proximity_step1_best.csv")

    out_rows = []      # dicts, final output order
    excerpt_rows = []  # (row dict, [(k, chunk_idx, fr)])
    counts = []        # (tag, category, target, n_candidates, n_selected)

    for model, layer in LAYERS:
        tag = f"{model}_L{layer}"
        t_layer = time.time()
        print(f"\n===== {tag} =====")
        with np.load(reports / f"equalized_presence_{tag}.npz", allow_pickle=False) as data:
            layer_works = [str(w) for w in data["works"].tolist()]
            assert layer_works == works, f"{tag}: works differ from the chunks CSV work list"
            presence_prob = data["presence_prob"]
            presence_obs = data["presence_obs"]
            feature_ids = data["feature_ids"].astype(np.int64)
        assert np.array_equal(feature_ids, np.arange(len(feature_ids))), f"{tag}: feature_ids are not 0..N-1"
        S = presence_prob >= args.presence_threshold
        n_eq = S.sum(axis=0)
        with np.load(root / "activations_v2" / f"{tag}_firing.npz", allow_pickle=False) as data:
            npz_ids = [str(x) for x in data["chunk_ids"].tolist()]
            assert npz_ids == csv_chunk_ids, f"{tag}: firing npz chunk_ids differ from the CSV (order included)"
            firing = data["firing_rate"]
        assert firing.shape == (n_chunks, len(feature_ids)), f"{tag}: firing_rate shape {firing.shape}"
        mean_fr_all = firing.mean(axis=0, dtype=np.float64)

        selected = []  # (category, target, rank, n_candidates, feature_id, jaccard_str)

        # (a) genre_near
        groups_in_layer = []
        for row in best_rows:
            if row["model"] == model and int(row["layer"]) == layer and row["group"] not in groups_in_layer:
                groups_in_layer.append(row["group"])
        for group in sorted(groups_in_layer):
            cands = [row for row in best_rows
                     if row["model"] == model and int(row["layer"]) == layer and row["group"] == group
                     and float(row["jaccard"]) >= args.genre_min_j]
            cands.sort(key=lambda r: (-float(r["jaccard"]), int(r["feature_id"])))
            for rank, row in enumerate(cands, start=1):
                selected.append(("genre_near", group, rank, len(cands), int(row["feature_id"]), row["jaccard"]))
            counts.append((tag, "genre_near", group, len(cands), len(cands)))

        # (b) silence: S == all works except w
        stats = by_work[(model, layer)]
        silence_targets, top3_c = top_works(stats, 1, 3, CONTRACT_SILENCE_EXTRA)
        print(f"silence targets: top3 complement_eq_mean = {top3_c} + contract {CONTRACT_SILENCE_EXTRA} -> {silence_targets}")
        for w in silence_targets:
            wi = work_index[w]
            mask = (n_eq == n_works - 1) & (~S[wi])
            cand_ids = np.flatnonzero(mask)
            order = sorted(cand_ids.tolist(), key=lambda f: (-mean_fr_all[f], f))
            chosen = order[:args.cap]
            for rank, f in enumerate(chosen, start=1):
                selected.append(("silence", w, rank, len(cand_ids), int(f), ""))
            counts.append((tag, "silence", w, len(cand_ids), len(chosen)))

        # (c) unique: S == {w}
        unique_targets, top3_s = top_works(stats, 0, 3, CONTRACT_UNIQUE_EXTRA)
        print(f"unique targets: top3 single_eq_mean = {top3_s} + contract {CONTRACT_UNIQUE_EXTRA} -> {unique_targets}")
        for w in unique_targets:
            wi = work_index[w]
            mask = (n_eq == 1) & S[wi]
            cand_ids = np.flatnonzero(mask)
            if len(cand_ids):
                mean_in_w = firing[rows_of_work[w]][:, cand_ids].mean(axis=0, dtype=np.float64)
                key_of = {int(f): float(m) for f, m in zip(cand_ids.tolist(), mean_in_w.tolist())}
            else:
                key_of = {}
            order = sorted(cand_ids.tolist(), key=lambda f: (-key_of[f], f))
            chosen = order[:args.cap]
            for rank, f in enumerate(chosen, start=1):
                selected.append(("unique", w, rank, len(cand_ids), int(f), ""))
            counts.append((tag, "unique", w, len(cand_ids), len(chosen)))

        # ---- per-row statistics ---------------------------------------- #
        for category, target, rank, n_cand, f, jaccard_str in selected:
            col = firing[:, f].astype(np.float64)
            fired = col > 0
            n_fired = int(fired.sum())
            order = np.argsort(-col, kind="stable")  # ties: earlier CSV row first
            top = [(k + 1, int(order[k]), float(col[order[k]])) for k in range(3)]
            rho_v = spearman(col[verse_mask], verse_ratio[verse_mask])
            m_prose = float(col[prose_mask].mean()) if prose_mask.any() else float("nan")
            m_verse = float(col[verse_only_mask].mean()) if verse_only_mask.any() else float("nan")
            ratio = m_verse / m_prose if (m_prose > 0) else None
            row = {
                "model": model, "layer": layer, "category": category, "target": target,
                "rank_in_category": rank, "n_candidates_in_category": n_cand,
                "feature_id": f, "jaccard": jaccard_str,
                "n_works_eq": int(n_eq[f]), "works_eq": " ".join(w for wi, w in enumerate(works) if S[wi, f]),
                "n_works_obs": int(presence_obs[:, f].sum()),
                "works_obs": " ".join(w for wi, w in enumerate(works) if presence_obs[wi, f]),
                "mean_fr_all": fmt(float(mean_fr_all[f])),
                "mean_fr_fired": fmt(float(col[fired].mean())) if n_fired else "",
                "n_chunks_fired": n_fired, "max_fr": fmt(float(col.max())),
                "rho_verse": fmt(rho_v, 4), "n_verse_used": int(verse_mask.sum()),
                "mean_fr_prose_only": fmt(m_prose), "mean_fr_verse_only": fmt(m_verse),
                "verse_over_prose": fmt(ratio, 4) if ratio is not None else "",
                "rho_stage": fmt(spearman(col, stage_total), 4),
                "rho_speaker_changes": fmt(spearman(col, speaker_changes), 4),
            }
            for k, ci, fr in top:
                cr = chunk_rows[ci]
                row[f"top{k}_chunk_id"] = cr["chunk_id"]
                row[f"top{k}_work"] = cr["folger_abbr"]
                row[f"top{k}_act_scene"] = f"{cr['act']}.{cr['scene']}"
                row[f"top{k}_fr"] = fmt(fr)
            out_rows.append(row)
            excerpt_rows.append((row, top))

        for c in CATEGORIES:
            line = "; ".join(f"{t}: {nc} cand / {ns} sel" for (tg, cat, t, nc, ns) in counts if tg == tag and cat == c)
            print(f"  {c:<10} {line}")
        del firing
        print(f"layer done in {time.time() - t_layer:.1f}s ({len(selected)} rows)")

    # ---- Neuronpedia ----------------------------------------------------- #
    keys = []
    seen = set()
    for row in out_rows:
        key = (row["model"], row["layer"], row["feature_id"])
        if key not in seen:
            seen.add(key)
            keys.append(key)
    print(f"\n===== Neuronpedia ({args.neuronpedia}): {len(keys)} unique features =====")
    np_cols, np_summary = neuronpedia_lookup(keys, args.neuronpedia, cache_path)
    for row in out_rows:
        row.update(np_cols[np_key(row["model"], row["layer"], row["feature_id"])])
    print(f"Neuronpedia summary: ok={np_summary['n_ok']} failed={np_summary['n_fail']} statuses={np_summary['statuses']} "
          f"aborted={np_summary['aborted']}")

    # ---- outputs --------------------------------------------------------- #
    write_csv(reports / "pattern_feature_ids.csv", CSV_HEADER, [[row[c] for c in CSV_HEADER] for row in out_rows])
    with (reports / "pattern_feature_excerpts.txt").open("w", encoding="utf-8", newline="") as handle:
        for row, top in excerpt_rows:
            handle.write(
                f"==== {row['model']} L{row['layer']} | {row['category']} | {row['target']} | f{row['feature_id']} | "
                f"works_eq={row['n_works_eq']} | mean_fr_all={float(row['mean_fr_all']):.4f} | "
                f"rho_verse={row['rho_verse']} | NP: {row['np_explanation']}\n"
            )
            for k, ci, fr in top:
                cr = chunk_rows[ci]
                handle.write(f"--- top{k} {cr['chunk_id']} ({cr['folger_abbr']} {cr['act']}.{cr['scene']}) firing_rate={fr:.4f} ---\n")
                handle.write(cr["text_content"])
                handle.write("\n\n")
    print(f"\nrows written: {len(out_rows)}; total elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

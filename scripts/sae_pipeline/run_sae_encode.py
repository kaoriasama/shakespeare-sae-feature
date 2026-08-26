#!/usr/bin/env python3
"""Stream Shakespeare chunks through Gemma models and Gemma Scope SAEs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import heapq
import json
import os
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = str(PROJECT_ROOT / "chunks" / "shakespeare_chunks_v2.csv")
CSV_SHA256 = "d6c48d0934a30e96706ec7aae9505b66cce10acd96e656593b3d1b58d322327d"
OUT_DIR = str(PROJECT_ROOT / "activations_v2")
SAE_ROOT = str(PROJECT_ROOT / "sae_weights")
FEATURES = 16_384
MAX_TOKENS = 1_024
DEFAULT_G2_VARIANTS = "4=average_l0_124,11=average_l0_80,22=average_l0_72"

MODELS = {
    "g2": {
        "hf_id": "google/gemma-2-2b",
        "n_layers": 26,
        "hidden": 2304,
        "layers": [4, 11, 22],
    },
    "g3": {
        "hf_id": "google/gemma-3-4b-pt",
        "n_layers": 34,
        "hidden": 2560,
        "layers": [9, 17, 29],
    },
}

MARKERS = {
    "g2": {4: 7718, 11: 6671, 22: 9813},
    "g3": {9: 7715, 17: 9814, 29: 11332},
}

# These must be set before torch or transformers is imported in main().
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Third-party modules are assigned after argparse has handled --help.
np = None
torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode Shakespeare residual-stream activations with Gemma Scope "
            "JumpReLU SAEs."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="process only the first N chunks and write under OUT_DIR/smoke/",
    )
    parser.add_argument(
        "--models",
        choices=("both", "g2", "g3"),
        default="both",
        help="models to process (default: both)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="execution device; auto prefers MPS when available (default: auto)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="number of chunks per output shard (default: 500)",
    )
    parser.add_argument(
        "--g2-variants",
        default=DEFAULT_G2_VARIANTS,
        metavar="LAYER=VARIANT,...",
        help="Gemma 2 SAE variants by layer (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.shard_size <= 0:
        parser.error("--shard-size must be a positive integer")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_and_verify_rows():
    actual_sha256 = sha256_file(CSV_PATH)
    if actual_sha256 != CSV_SHA256:
        raise RuntimeError(
            "CSV SHA-256 mismatch for {!r}: expected {}, got {}. "
            "Refusing to load any model.".format(
                CSV_PATH, CSV_SHA256, actual_sha256
            )
        )

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header: {!r}".format(CSV_PATH))
        missing = {"chunk_id", "text_content"}.difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                "CSV is missing required columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        rows = [
            {"chunk_id": row["chunk_id"], "text_content": row["text_content"]}
            for row in reader
        ]

    assert len(rows) == 4600, "expected 4600 CSV rows, found {}".format(len(rows))
    return rows, actual_sha256


def parse_g2_variants(spec):
    variants = {}
    for item in spec.split(","):
        item = item.strip()
        if not item or "=" not in item:
            raise ValueError(
                "invalid --g2-variants item {!r}; expected LAYER=VARIANT".format(
                    item
                )
            )
        layer_text, variant = item.split("=", 1)
        layer = int(layer_text.strip())
        variant = variant.strip()
        if not variant:
            raise ValueError("empty SAE variant for layer {}".format(layer))
        if layer in variants:
            raise ValueError("duplicate SAE variant for layer {}".format(layer))
        variants[layer] = variant
    required = set(MODELS["g2"]["layers"])
    if set(variants) != required:
        raise ValueError(
            "--g2-variants must specify exactly layers {}; got {}".format(
                sorted(required), sorted(variants)
            )
        )
    return variants


def choose_device(requested):
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is not available")
    return requested


def model_tags(selection):
    if selection == "both":
        return ["g2", "g3"]
    return [selection]


def pretokenize_all(rows, tags, AutoTokenizer):
    tokenized = {}
    for tag in tags:
        config = MODELS[tag]
        tokenizer = AutoTokenizer.from_pretrained(config["hf_id"])
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise RuntimeError("{} tokenizer has no BOS token id".format(tag))

        all_ids = []
        counts = []
        offenders = []
        for row in rows:
            encoded = tokenizer(
                row["text_content"],
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            input_ids = encoded["input_ids"]
            if not input_ids or input_ids[0] != bos_id:
                raise RuntimeError(
                    "{} chunk {!r} does not start with BOS id {}".format(
                        tag, row["chunk_id"], bos_id
                    )
                )
            count = len(input_ids)
            counts.append(count)
            all_ids.append(input_ids)
            if count > MAX_TOKENS:
                offenders.append((row["chunk_id"], count))

        if offenders:
            details = ", ".join(
                "{} ({})".format(chunk_id, count)
                for chunk_id, count in offenders
            )
            raise RuntimeError(
                "{} token sequences exceed {} tokens: {}".format(
                    tag, MAX_TOKENS, details
                )
            )

        tokenized[tag] = {
            "tokenizer": tokenizer,
            "input_ids": all_ids,
            "stats": {
                "n_chunks": len(counts),
                "max": max(counts),
                "mean": float(sum(counts)) / float(len(counts)),
            },
        }
        print(
            "{} token pre-scan: n={}, max={}, mean={:.2f}".format(
                tag,
                len(counts),
                tokenized[tag]["stats"]["max"],
                tokenized[tag]["stats"]["mean"],
            ),
            flush=True,
        )
    return tokenized


def normalized_key_map(keys, path):
    normalized = {}
    for key in keys:
        lower = key.lower()
        if lower in normalized:
            raise RuntimeError(
                "case-insensitive duplicate key {!r} in {!r}".format(lower, path)
            )
        normalized[lower] = key
    required = {"w_enc", "w_dec", "b_dec", "b_enc", "threshold"}
    missing = required.difference(normalized)
    if missing:
        raise RuntimeError(
            "SAE {!r} is missing keys: {}".format(path, ", ".join(sorted(missing)))
        )
    return normalized


def validate_sae_arrays(arrays, hidden, path):
    expected = {
        "w_enc": (hidden, FEATURES),
        "b_enc": (FEATURES,),
        "threshold": (FEATURES,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError(
                "SAE {!r} key {} has shape {}, expected {}".format(
                    path, key, arrays[key].shape, shape
                )
            )
        if arrays[key].dtype != np.float32:
            raise RuntimeError(
                "SAE {!r} key {} has dtype {}, expected float32".format(
                    path, key, arrays[key].dtype
                )
            )


def load_sae(tag, layer, g2_variants, safe_open):
    config = MODELS[tag]
    if tag == "g2":
        variant = g2_variants[layer]
        path = os.path.join(
            SAE_ROOT,
            "gemma-scope-2b-pt-res",
            "layer_{}".format(layer),
            "width_16k",
            variant,
            "params.npz",
        )
        with np.load(path, allow_pickle=False) as source:
            key_map = normalized_key_map(source.files, path)
            arrays = {
                name: np.ascontiguousarray(source[key_map[name]])
                for name in ("w_enc", "b_enc", "threshold")
            }
    else:
        path = os.path.join(
            SAE_ROOT,
            "gemma-scope-2-4b-pt",
            "resid_post",
            "layer_{}_width_16k_l0_medium".format(layer),
            "params.safetensors",
        )
        with safe_open(path, framework="numpy", device="cpu") as source:
            key_map = normalized_key_map(source.keys(), path)
            arrays = {
                name: np.ascontiguousarray(source.get_tensor(key_map[name]))
                for name in ("w_enc", "b_enc", "threshold")
            }

    validate_sae_arrays(arrays, config["hidden"], path)
    return arrays


def load_model(config, AutoModelForCausalLM, Gemma3ForConditionalGeneration):
    if config["hf_id"] != MODELS["g3"]["hf_id"]:
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], torch_dtype=torch.bfloat16
        )
        return model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config["hf_id"], torch_dtype=torch.bfloat16
        )
    except ValueError:
        # Gemma 3's multimodal config is not accepted by some AutoModel versions.
        if Gemma3ForConditionalGeneration is None:
            raise RuntimeError(
                "AutoModelForCausalLM rejected Gemma 3 and the installed "
                "transformers has no Gemma3ForConditionalGeneration fallback"
            )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            config["hf_id"], torch_dtype=torch.bfloat16
        )
    return model


def jump_relu_stats(x, sae):
    # Official Gemma Scope convention: do not subtract b_dec from x.
    pre = x @ sae["w_enc"] + sae["b_enc"]
    acts = np.maximum(pre, np.float32(0.0)) * (pre > sae["threshold"])
    fired = acts > 0
    counts = fired.sum(axis=0)
    firing_rate = counts.astype(np.float32) / np.float32(x.shape[0])
    activation_sum = acts.sum(axis=0, dtype=np.float32)
    mean_act = np.zeros(FEATURES, dtype=np.float32)
    np.divide(
        activation_sum,
        counts,
        out=mean_act,
        where=counts != 0,
        casting="unsafe",
    )
    return acts, firing_rate, mean_act


def ensure_final_absent(final_path):
    if os.path.exists(final_path):
        raise FileExistsError(
            "refusing to overwrite pre-existing output: {!r}".format(final_path)
        )


def atomic_write_npz(final_path, arrays):
    ensure_final_absent(final_path)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    ensure_final_absent(final_path)
    os.replace(tmp_path, final_path)


def atomic_write_json(final_path, value):
    ensure_final_absent(final_path)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    ensure_final_absent(final_path)
    os.replace(tmp_path, final_path)


def shard_paths(out_dir, tag, layers, start):
    layer_paths = {
        layer: os.path.join(
            out_dir, "{}_L{}_shard{:04d}.npz".format(tag, layer, start)
        )
        for layer in layers
    }
    marker_path = os.path.join(
        out_dir, "{}_markers_shard{:04d}.json".format(tag, start)
    )
    return layer_paths, marker_path


def shard_is_complete(out_dir, tag, layers, start):
    layer_paths, marker_path = shard_paths(out_dir, tag, layers, start)
    layer_exists = [os.path.exists(layer_paths[layer]) for layer in layers]
    marker_exists = os.path.exists(marker_path)
    if all(layer_exists):
        return True
    if any(layer_exists) or marker_exists:
        existing = [
            layer_paths[layer] for layer in layers if os.path.exists(layer_paths[layer])
        ]
        if marker_exists:
            existing.append(marker_path)
        raise RuntimeError(
            "partial shard outputs found; refusing to replace them: {}".format(
                ", ".join(repr(path) for path in existing)
            )
        )
    return False


def add_marker_candidates(heap, values, input_ids, chunk_id, tokenizer, serial):
    candidate_positions = np.arange(values.shape[0])
    if candidate_positions.size > 50:
        local = np.argpartition(values, -50)[-50:]
        candidate_positions = candidate_positions[local]
    for non_bos_position in candidate_positions:
        full_position = int(non_bos_position) + 1
        act = float(values[non_bos_position])
        record = {
            "chunk_id": chunk_id,
            "position": full_position,
            "token": tokenizer.decode(
                [int(input_ids[full_position])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "act": act,
        }
        item = (act, serial, record)
        serial += 1
        if len(heap) < 50:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    return serial


def heap_records(heap):
    return [
        item[2]
        for item in sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    ]


def format_eta(seconds):
    if seconds < 0 or not np.isfinite(seconds):
        return "unknown"
    return str(dt.timedelta(seconds=int(seconds)))


def process_shard(
    tag,
    config,
    model,
    tokenizer,
    all_input_ids,
    rows,
    saes,
    out_dir,
    start,
    stop,
    device,
    model_started,
    verification_state,
):
    layers = config["layers"]
    n_shard = stop - start
    chunk_ids = np.asarray(
        [rows[index]["chunk_id"] for index in range(start, stop)], dtype=str
    )
    firing = {
        layer: np.empty((n_shard, FEATURES), dtype=np.float32) for layer in layers
    }
    mean_act = {
        layer: np.empty((n_shard, FEATURES), dtype=np.float32) for layer in layers
    }
    marker_max = {
        layer: np.empty(n_shard, dtype=np.float32) for layer in layers
    }
    marker_heaps = {layer: [] for layer in layers}
    marker_serial = {layer: 0 for layer in layers}

    shard_started = time.monotonic()
    total = len(rows)
    for local_index, global_index in enumerate(range(start, stop)):
        ids = all_input_ids[global_index]
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_tensor, output_hidden_states=True)
        hs = out.hidden_states
        expected_hidden_states = config["n_layers"] + 1
        assert len(hs) == expected_hidden_states, (
            "{} returned {} hidden states, expected {}".format(
                tag, len(hs), expected_hidden_states
            )
        )

        for layer in layers:
            residual = hs[layer + 1][0].detach().to("cpu", dtype=torch.float32).numpy()
            expected_shape = (len(ids), config["hidden"])
            assert residual.shape == expected_shape, (
                "{} layer {} residual shape {} != {}".format(
                    tag, layer, residual.shape, expected_shape
                )
            )

            if verification_state is not None and verification_state["count"] < 10:
                chunk_id = rows[global_index]["chunk_id"]
                verification_state["arrays"][
                    "{}_L{}_{}".format(tag, layer, chunk_id)
                ] = residual.copy()

            non_bos = residual[1:]
            if non_bos.shape[0] == 0:
                raise RuntimeError(
                    "{} chunk {!r} contains BOS but no content tokens".format(
                        tag, rows[global_index]["chunk_id"]
                    )
                )
            acts, chunk_firing, chunk_mean = jump_relu_stats(non_bos, saes[layer])
            firing[layer][local_index] = chunk_firing
            mean_act[layer][local_index] = chunk_mean
            marker = MARKERS[tag][layer]
            marker_values = acts[:, marker]
            marker_max[layer][local_index] = np.max(marker_values)
            marker_serial[layer] = add_marker_candidates(
                marker_heaps[layer],
                marker_values,
                ids,
                rows[global_index]["chunk_id"],
                tokenizer,
                marker_serial[layer],
            )

        if verification_state is not None and verification_state["count"] < 10:
            chunk_id = rows[global_index]["chunk_id"]
            verification_state["arrays"][
                "{}_ids_{}".format(tag, chunk_id)
            ] = np.asarray(ids, dtype=np.int32)
            verification_state["count"] += 1
            if verification_state["count"] == verification_state["target"]:
                atomic_write_npz(
                    verification_state["path"], verification_state["arrays"]
                )
                verification_state = None

        completed = global_index + 1
        if completed % 25 == 0 or completed == total:
            elapsed = time.monotonic() - model_started
            eta = elapsed / float(completed) * float(total - completed)
            print(
                "{} shard {:04d} {}/{} elapsed={} ETA={}".format(
                    tag,
                    start,
                    completed,
                    total,
                    format_eta(elapsed),
                    format_eta(eta),
                ),
                flush=True,
            )

        del out, hs, input_tensor

    marker_payload = {
        "model": tag,
        "shard_start": start,
        "shard_stop": stop,
        "layers": {
            str(layer): {
                "marker": MARKERS[tag][layer],
                "top_tokens": heap_records(marker_heaps[layer]),
            }
            for layer in layers
        },
    }
    layer_paths, marker_path = shard_paths(out_dir, tag, layers, start)
    # Marker first makes the three layer files a reliable completion sentinel.
    atomic_write_json(marker_path, marker_payload)
    for layer in layers:
        atomic_write_npz(
            layer_paths[layer],
            {
                "chunk_ids": chunk_ids,
                "firing_rate": firing[layer],
                "mean_act": mean_act[layer],
                "marker_max": marker_max[layer],
            },
        )
    return time.monotonic() - shard_started, verification_state


def load_merged_field(out_dir, tag, layer, starts, total, field):
    merged = np.empty((total, FEATURES), dtype=np.float32)
    merged_ids = []
    offset = 0
    for start in starts:
        shard_path = os.path.join(
            out_dir, "{}_L{}_shard{:04d}.npz".format(tag, layer, start)
        )
        with np.load(shard_path, allow_pickle=False) as shard:
            values = shard[field]
            ids = shard["chunk_ids"]
            end = offset + values.shape[0]
            merged[offset:end] = values
            merged_ids.extend(ids.tolist())
            offset = end
    assert offset == total, "merged {} rows, expected {}".format(offset, total)
    return np.asarray(merged_ids, dtype=str), merged


def merge_layer_outputs(out_dir, tag, layers, starts, rows):
    expected_ids = [row["chunk_id"] for row in rows]
    for layer in layers:
        destinations = {
            "firing_rate": os.path.join(
                out_dir, "{}_L{}_firing.npz".format(tag, layer)
            ),
            "mean_act": os.path.join(
                out_dir, "{}_L{}_mean_act.npz".format(tag, layer)
            ),
        }
        for field, destination in destinations.items():
            if os.path.exists(destination):
                print("keeping completed output {}".format(destination), flush=True)
                continue
            chunk_ids, values = load_merged_field(
                out_dir, tag, layer, starts, len(rows), field
            )
            assert chunk_ids.tolist() == expected_ids, (
                "merged chunk order mismatch for {} layer {}".format(tag, layer)
            )
            atomic_write_npz(
                destination, {"chunk_ids": chunk_ids, field: values}
            )
            del values


def merge_markers(out_dir, tags, starts_by_tag):
    destination = os.path.join(out_dir, "markers.json")
    if os.path.exists(destination):
        print("keeping completed output {}".format(destination), flush=True)
        return

    merged = {}
    for tag in tags:
        config = MODELS[tag]
        merged[tag] = {}
        for layer in config["layers"]:
            heap = []
            serial = 0
            n_chunks_fired = 0
            firing_rate_sum = 0.0
            n_chunks = 0
            max_act = 0.0
            for start in starts_by_tag[tag]:
                marker_path = os.path.join(
                    out_dir, "{}_markers_shard{:04d}.json".format(tag, start)
                )
                with open(marker_path, "r", encoding="utf-8") as handle:
                    marker_shard = json.load(handle)
                for record in marker_shard["layers"][str(layer)]["top_tokens"]:
                    item = (float(record["act"]), serial, record)
                    serial += 1
                    if len(heap) < 50:
                        heapq.heappush(heap, item)
                    elif item[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, item)

                layer_path = os.path.join(
                    out_dir, "{}_L{}_shard{:04d}.npz".format(tag, layer, start)
                )
                marker = MARKERS[tag][layer]
                with np.load(layer_path, allow_pickle=False) as layer_shard:
                    marker_rates = layer_shard["firing_rate"][:, marker]
                    marker_maxima = layer_shard["marker_max"]
                n_chunks_fired += int(np.count_nonzero(marker_rates > 0))
                firing_rate_sum += float(marker_rates.sum(dtype=np.float64))
                n_chunks += int(marker_rates.shape[0])
                if marker_maxima.size:
                    max_act = max(max_act, float(marker_maxima.max()))

            merged[tag][str(layer)] = {
                "marker": MARKERS[tag][layer],
                "top_tokens": heap_records(heap),
                "summary": {
                    "n_chunks_fired": n_chunks_fired,
                    "mean_firing_rate_across_chunks": (
                        firing_rate_sum / float(n_chunks) if n_chunks else 0.0
                    ),
                    "max_act_overall": max_act,
                },
            }
    atomic_write_json(destination, merged)


def process_model(
    tag,
    rows,
    tokenized,
    out_dir,
    shard_size,
    device,
    g2_variants,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
    safe_open,
):
    config = MODELS[tag]
    layers = config["layers"]
    starts = list(range(0, len(rows), shard_size))
    completed = {
        start: shard_is_complete(out_dir, tag, layers, start) for start in starts
    }
    pending = [start for start in starts if not completed[start]]
    verification_path = os.path.join(
        out_dir, "{}_verification_samples.npz".format(tag)
    )
    if pending and not os.path.exists(verification_path) and pending[0] != 0:
        raise RuntimeError(
            "verification samples are missing but initial shards are already complete; "
            "refusing to re-forward completed shards"
        )
    if not pending and not os.path.exists(verification_path):
        raise RuntimeError(
            "all {} shards are complete but {!r} is missing".format(
                tag, verification_path
            )
        )

    model_started = time.monotonic()
    shard_timings = []
    model = None
    saes = None
    verification_state = None
    if pending:
        saes = {
            layer: load_sae(tag, layer, g2_variants, safe_open) for layer in layers
        }
        model = load_model(
            config, AutoModelForCausalLM, Gemma3ForConditionalGeneration
        )
        model.to(device).eval()
        if not os.path.exists(verification_path):
            verification_state = {
                "path": verification_path,
                "arrays": {},
                "count": 0,
                "target": min(10, len(rows)),
            }

    for start in starts:
        stop = min(start + shard_size, len(rows))
        if completed[start]:
            print("{} shard {:04d} complete; skipping".format(tag, start), flush=True)
            shard_timings.append(
                {
                    "start": start,
                    "stop": stop,
                    "seconds": 0.0,
                    "skipped": True,
                }
            )
            continue
        seconds, verification_state = process_shard(
            tag,
            config,
            model,
            tokenized["tokenizer"],
            tokenized["input_ids"],
            rows,
            saes,
            out_dir,
            start,
            stop,
            device,
            model_started,
            verification_state,
        )
        shard_timings.append(
            {
                "start": start,
                "stop": stop,
                "seconds": seconds,
                "skipped": False,
            }
        )

    if verification_state is not None:
        if verification_state["count"] != verification_state["target"]:
            raise RuntimeError(
                "collected {} verification chunks, expected {}".format(
                    verification_state["count"], verification_state["target"]
                )
            )
        atomic_write_npz(verification_path, verification_state["arrays"])

    if model is not None:
        del model
    if saes is not None:
        del saes
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    merge_layer_outputs(out_dir, tag, layers, starts, rows)
    return {
        "wall_seconds": time.monotonic() - model_started,
        "shards": shard_timings,
        "starts": starts,
    }


def main():
    args = parse_args()
    invocation_argv = list(sys.argv)
    started_at = dt.datetime.now().astimezone()
    started_stamp = started_at.strftime("%Y%m%d-%H%M%S")

    # Delayed imports keep --help independent of heavyweight libraries.
    global np, torch
    import numpy as numpy_module
    import torch as torch_module
    import transformers
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import Gemma3ForConditionalGeneration
    except ImportError:
        Gemma3ForConditionalGeneration = None

    np = numpy_module
    torch = torch_module

    rows, csv_sha256 = read_and_verify_rows()
    selected_rows = rows if args.limit is None else rows[: args.limit]
    out_dir = OUT_DIR if args.limit is None else os.path.join(OUT_DIR, "smoke")
    os.makedirs(out_dir, exist_ok=True)
    tags = model_tags(args.models)
    g2_variants = parse_g2_variants(args.g2_variants)
    device = choose_device(args.device)
    print("chosen device: {}".format(device), flush=True)

    tokenized = pretokenize_all(selected_rows, tags, AutoTokenizer)
    token_stats = {tag: tokenized[tag]["stats"] for tag in tags}
    model_results = {}
    starts_by_tag = {}
    for tag in tags:
        result = process_model(
            tag,
            selected_rows,
            tokenized[tag],
            out_dir,
            args.shard_size,
            device,
            g2_variants,
            AutoModelForCausalLM,
            Gemma3ForConditionalGeneration,
            safe_open,
        )
        starts_by_tag[tag] = result.pop("starts")
        model_results[tag] = result

    merge_markers(out_dir, tags, starts_by_tag)

    metadata = {
        "started_at": started_at.isoformat(),
        "argv": invocation_argv,
        "resolved_args": {
            "limit": args.limit,
            "models": args.models,
            "device_requested": args.device,
            "device": device,
            "shard_size": args.shard_size,
            "g2_variants": g2_variants,
            "out_dir": out_dir,
            "n_selected": len(selected_rows),
        },
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "csv_sha256": csv_sha256,
        "token_counts": token_stats,
        "models": model_results,
        "total_wall_seconds": (
            dt.datetime.now().astimezone() - started_at
        ).total_seconds(),
    }
    meta_path = os.path.join(out_dir, "run_meta_{}.json".format(started_stamp))
    atomic_write_json(meta_path, metadata)


if __name__ == "__main__":
    main()

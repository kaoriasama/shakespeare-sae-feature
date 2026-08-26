#!/usr/bin/env python3
"""Build a compact SQLite summary of the full SAE activation outputs."""

import argparse
import csv
import datetime as dt
import gc
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = str(PROJECT_ROOT / "chunks" / "shakespeare_chunks_v2.csv")
ACT_DIR = str(PROJECT_ROOT / "activations_v2")
DB_PATH = str(PROJECT_ROOT / "activations_v2" / "features.db")
MODELS = {"g2": [4, 11, 22], "g3": [9, 17, 29]}
MARKERS = {
    "g2": {4: 7718, 11: 6671, 22: 9813},
    "g3": {9: 7715, 17: 9814, 29: 11332},
}
FEATURES = 16384
GENRES = ["comedy", "history", "tragedy", "tragicomedy"]
SCRIPT_VERSION = "1.0"
NOTE = (
    "Summary views derived from activations_v2/*.npz. The npz files remain "
    "the authoritative full-resolution source (all 16,384 features per chunk). "
    "This DB intentionally stores per-feature summaries only — it is not a "
    "substitute for the npz."
)

INTEGER_COLUMNS = {
    "n_chars",
    "n_speeches",
    "n_speakers",
    "n_speaker_changes",
    "n_stage_entrance",
    "n_stage_exit",
    "n_stage_delivery",
    "n_stage_business",
    "n_stage_modifier",
    "n_stage_other",
    "n_verse_lines",
    "n_prose_paragraphs",
    "contains_song",
}
REAL_COLUMNS = {"verse_ratio"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the SAE per-feature SQLite summary database."
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        metavar="PATH",
        help="output SQLite path (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output database",
    )
    return parser.parse_args()


def quote_identifier(name):
    return '"{}"'.format(name.replace('"', '""'))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def convert_csv_value(name, value):
    if name in INTEGER_COLUMNS:
        return None if value == "" else int(value)
    if name in REAL_COLUMNS:
        return None if value == "" else nullable_float(value)
    return value


def nullable_float(value):
    result = float(value)
    return None if math.isnan(result) else result


def read_csv():
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header: {!r}".format(CSV_PATH))
        fieldnames = list(reader.fieldnames)
        if len(fieldnames) != len(set(fieldnames)):
            raise RuntimeError("CSV contains duplicate column names")
        missing = {"chunk_id", "text_content", "genre"}.difference(fieldnames)
        if missing:
            raise RuntimeError(
                "CSV is missing required columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        if "row_index" in fieldnames:
            raise RuntimeError("CSV already contains reserved column 'row_index'")

        db_fields = [name for name in fieldnames if name != "text_content"]
        rows = []
        chunk_ids = []
        genres = []
        for row_index, row in enumerate(reader):
            chunk_id = row["chunk_id"]
            if not chunk_id:
                raise RuntimeError("empty chunk_id at CSV row {}".format(row_index))
            values = [convert_csv_value(name, row[name]) for name in db_fields]
            values.append(row_index)
            rows.append(tuple(values))
            chunk_ids.append(chunk_id)
            genres.append(row["genre"])

    if len(rows) != 4600:
        raise RuntimeError("expected 4600 CSV rows, found {}".format(len(rows)))
    if len(set(chunk_ids)) != len(chunk_ids):
        raise RuntimeError("CSV chunk_id values are not unique")
    actual_genres = set(genres)
    if actual_genres != set(GENRES):
        raise RuntimeError(
            "expected genres {}, found {}".format(GENRES, sorted(actual_genres))
        )
    return db_fields, rows, chunk_ids, genres


def csv_column_definition(name):
    if name == "chunk_id":
        sql_type = "TEXT PRIMARY KEY"
    elif name in INTEGER_COLUMNS:
        sql_type = "INTEGER"
    elif name in REAL_COLUMNS:
        sql_type = "REAL"
    else:
        sql_type = "TEXT"
    return "{} {}".format(quote_identifier(name), sql_type)


def create_schema(conn, csv_fields):
    chunk_columns = [csv_column_definition(name) for name in csv_fields]
    chunk_columns.append('"row_index" INTEGER NOT NULL')
    conn.execute("CREATE TABLE chunks ({})".format(", ".join(chunk_columns)))

    conn.execute(
        """
        CREATE TABLE feature_summary (
            model TEXT NOT NULL,
            layer INTEGER NOT NULL,
            feature INTEGER NOT NULL,
            n_chunks_fired INTEGER,
            mean_firing_rate REAL,
            max_firing_rate REAL,
            mean_act_when_firing REAL,
            max_mean_act REAL,
            is_marker INTEGER NOT NULL DEFAULT 0,
            mean_fr_comedy REAL,
            mean_fr_history REAL,
            mean_fr_tragedy REAL,
            mean_fr_tragicomedy REAL,
            genre_range REAL,
            PRIMARY KEY (model, layer, feature)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE markers (
            model TEXT NOT NULL,
            layer INTEGER NOT NULL,
            feature INTEGER NOT NULL,
            n_chunks_fired INTEGER,
            mean_firing_rate REAL,
            max_act_overall REAL,
            top_tokens_json TEXT,
            PRIMARY KEY (model, layer)
        )
        """
    )
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")


def insert_chunks(conn, csv_fields, rows):
    columns = csv_fields + ["row_index"]
    sql = "INSERT INTO chunks ({}) VALUES ({})".format(
        ", ".join(quote_identifier(name) for name in columns),
        ", ".join("?" for _ in columns),
    )
    conn.executemany(sql, rows)


def normalized_chunk_ids(values, source):
    if values.ndim != 1:
        raise RuntimeError(
            "{} chunk_ids must be one-dimensional, got shape {}".format(
                source, values.shape
            )
        )
    if values.dtype.kind == "U":
        return values
    if values.dtype.kind == "S":
        try:
            return values.astype("U")
        except UnicodeDecodeError as exc:
            raise RuntimeError("{} chunk_ids are not valid UTF-8".format(source)) from exc
    raise RuntimeError(
        "{} chunk_ids must contain strings, got dtype {}".format(
            source, values.dtype
        )
    )


def assert_chunk_ids(values, expected, source, np):
    actual = normalized_chunk_ids(values, source)
    if actual.shape != expected.shape:
        raise RuntimeError(
            "{} chunk_ids shape mismatch: expected {}, got {}".format(
                source, expected.shape, actual.shape
            )
        )
    unequal = np.flatnonzero(actual != expected)
    if unequal.size:
        index = int(unequal[0])
        raise RuntimeError(
            "{} chunk_ids differ from CSV order at row {}: expected {!r}, "
            "got {!r}".format(source, index, expected[index], actual[index])
        )


def load_npz_array(path, key, expected_ids, expected_shape, np):
    with np.load(path, allow_pickle=False) as archive:
        if "chunk_ids" not in archive or key not in archive:
            raise RuntimeError(
                "{} must contain keys 'chunk_ids' and {!r}".format(path, key)
            )
        chunk_ids = archive["chunk_ids"]
        values = archive[key]
    assert_chunk_ids(chunk_ids, expected_ids, path, np)
    if values.shape != expected_shape:
        raise RuntimeError(
            "{} key {!r} has shape {}; expected {}".format(
                path, key, values.shape, expected_shape
            )
        )
    return values


def feature_rows(
    model,
    layer,
    marker,
    n_chunks_fired,
    mean_firing_rate,
    max_firing_rate,
    mean_act_when_firing,
    max_mean_act,
    genre_means,
    genre_range,
):
    for feature in range(FEATURES):
        yield (
            model,
            layer,
            feature,
            int(n_chunks_fired[feature]),
            nullable_float(mean_firing_rate[feature]),
            nullable_float(max_firing_rate[feature]),
            nullable_float(mean_act_when_firing[feature]),
            nullable_float(max_mean_act[feature]),
            int(feature == marker),
            nullable_float(genre_means[0, feature]),
            nullable_float(genre_means[1, feature]),
            nullable_float(genre_means[2, feature]),
            nullable_float(genre_means[3, feature]),
            nullable_float(genre_range[feature]),
        )


def insert_feature_summaries(conn, chunk_ids, genres, np):
    expected_ids = np.asarray(chunk_ids, dtype="U")
    genre_array = np.asarray(genres, dtype="U")
    expected_shape = (len(chunk_ids), FEATURES)
    insert_sql = """
        INSERT INTO feature_summary (
            model, layer, feature, n_chunks_fired, mean_firing_rate,
            max_firing_rate, mean_act_when_firing, max_mean_act, is_marker,
            mean_fr_comedy, mean_fr_history, mean_fr_tragedy,
            mean_fr_tragicomedy, genre_range
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for model, layers in MODELS.items():
        for layer in layers:
            print("Processing {} layer {}...".format(model, layer), flush=True)
            firing_path = os.path.join(
                ACT_DIR, "{}_L{}_firing.npz".format(model, layer)
            )
            mean_act_path = os.path.join(
                ACT_DIR, "{}_L{}_mean_act.npz".format(model, layer)
            )
            firing_rate = load_npz_array(
                firing_path, "firing_rate", expected_ids, expected_shape, np
            )
            mean_act = load_npz_array(
                mean_act_path, "mean_act", expected_ids, expected_shape, np
            )

            fired = firing_rate > 0
            n_chunks_fired = np.count_nonzero(fired, axis=0)
            mean_firing_rate = np.mean(firing_rate, axis=0, dtype=np.float64)
            max_firing_rate = np.max(firing_rate, axis=0)
            act_sum = np.sum(mean_act, axis=0, dtype=np.float64, where=fired)
            mean_act_when_firing = np.full(FEATURES, np.nan, dtype=np.float64)
            np.divide(
                act_sum,
                n_chunks_fired,
                out=mean_act_when_firing,
                where=n_chunks_fired > 0,
            )
            max_mean_act = np.max(mean_act, axis=0)

            genre_means = []
            for genre in GENRES:
                mask = genre_array == genre
                genre_sum = np.sum(
                    firing_rate,
                    axis=0,
                    dtype=np.float64,
                    where=mask[:, np.newaxis],
                )
                genre_means.append(genre_sum / int(np.count_nonzero(mask)))
            genre_means = np.stack(genre_means, axis=0)
            genre_range = np.max(genre_means, axis=0) - np.min(
                genre_means, axis=0
            )

            conn.executemany(
                insert_sql,
                feature_rows(
                    model,
                    layer,
                    MARKERS[model][layer],
                    n_chunks_fired,
                    mean_firing_rate,
                    max_firing_rate,
                    mean_act_when_firing,
                    max_mean_act,
                    genre_means,
                    genre_range,
                ),
            )
            print("Finished {} layer {}.".format(model, layer), flush=True)

            del firing_rate
            del mean_act
            del fired
            del n_chunks_fired
            del mean_firing_rate
            del max_firing_rate
            del act_sum
            del mean_act_when_firing
            del max_mean_act
            del genre_means
            del genre_range
            gc.collect()


def insert_markers(conn):
    path = os.path.join(ACT_DIR, "markers.json")
    with open(path, "r", encoding="utf-8") as handle:
        marker_data = json.load(handle)

    rows = []
    for model, layers in MODELS.items():
        for layer in layers:
            try:
                entry = marker_data[model][str(layer)]
                summary = entry["summary"]
                feature = int(entry["marker"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "invalid markers.json entry for {} layer {}".format(
                        model, layer
                    )
                ) from exc
            if feature != MARKERS[model][layer]:
                raise RuntimeError(
                    "markers.json feature mismatch for {} layer {}: expected {}, "
                    "got {}".format(model, layer, MARKERS[model][layer], feature)
                )
            rows.append(
                (
                    model,
                    layer,
                    feature,
                    int(summary["n_chunks_fired"]),
                    nullable_float(summary["mean_firing_rate_across_chunks"]),
                    nullable_float(summary["max_act_overall"]),
                    json.dumps(entry["top_tokens"], ensure_ascii=False),
                )
            )

    conn.executemany(
        """
        INSERT INTO markers (
            model, layer, feature, n_chunks_fired, mean_firing_rate,
            max_act_overall, top_tokens_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def newest_run_meta():
    paths = glob.glob(os.path.join(ACT_DIR, "run_meta_*.json"))
    if not paths:
        raise RuntimeError("no run_meta_*.json found in {!r}".format(ACT_DIR))
    newest = max(paths, key=lambda path: (os.path.getmtime(path), path))
    return os.path.basename(newest)


def insert_meta(conn, csv_sha256, n_chunks):
    rows = [
        ("built_at", dt.datetime.now().astimezone().isoformat(timespec="seconds")),
        ("csv_sha256", csv_sha256),
        ("n_chunks", str(n_chunks)),
        ("n_features", str(FEATURES)),
        ("models_layers", json.dumps(MODELS, ensure_ascii=False, sort_keys=True)),
        ("source_run_meta", newest_run_meta()),
        ("script_version", SCRIPT_VERSION),
        ("note", NOTE),
    ]
    conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)


def create_indexes(conn):
    conn.execute("CREATE INDEX idx_chunks_play ON chunks (play)")
    conn.execute("CREATE INDEX idx_chunks_genre ON chunks (genre)")
    conn.execute("CREATE INDEX idx_chunks_row_index ON chunks (row_index)")
    conn.execute(
        "CREATE INDEX idx_feature_model_layer ON feature_summary (model, layer)"
    )
    conn.execute(
        "CREATE INDEX idx_feature_n_chunks_fired "
        "ON feature_summary (n_chunks_fired)"
    )
    conn.execute(
        "CREATE INDEX idx_feature_mean_firing_rate "
        "ON feature_summary (mean_firing_rate)"
    )
    conn.execute(
        "CREATE INDEX idx_feature_genre_range ON feature_summary (genre_range)"
    )
    conn.execute(
        "CREATE INDEX idx_feature_is_marker ON feature_summary (is_marker)"
    )


def build_database(db_path, force, np):
    if os.path.exists(db_path) and not force:
        raise FileExistsError(
            "output database already exists; use --force to replace it: {!r}".format(
                db_path
            )
        )
    output_dir = os.path.dirname(os.path.abspath(db_path))
    if not os.path.isdir(output_dir):
        raise FileNotFoundError("output directory does not exist: {!r}".format(output_dir))

    temp_path = db_path + ".tmp"
    try:
        descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            "temporary database already exists; refusing to modify it: {!r}".format(
                temp_path
            )
        ) from exc
    os.close(descriptor)

    conn = None
    try:
        csv_fields, csv_rows, chunk_ids, genres = read_csv()
        csv_sha256 = sha256_file(CSV_PATH)

        conn = sqlite3.connect(temp_path)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("BEGIN")
        create_schema(conn, csv_fields)
        insert_chunks(conn, csv_fields, csv_rows)
        insert_feature_summaries(conn, chunk_ids, genres, np)
        insert_markers(conn)
        insert_meta(conn, csv_sha256, len(chunk_ids))
        create_indexes(conn)
        conn.commit()

        counts = {}
        for table in ("chunks", "feature_summary", "markers", "meta"):
            counts[table] = conn.execute(
                "SELECT COUNT(*) FROM {}".format(quote_identifier(table))
            ).fetchone()[0]
        conn.close()
        conn = None

        if os.path.exists(db_path) and not force:
            raise FileExistsError(
                "output database appeared during the build; refusing to replace "
                "it without --force: {!r}".format(db_path)
            )
        os.replace(temp_path, db_path)
    except BaseException:
        if conn is not None:
            conn.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    size_mb = os.path.getsize(db_path) / (1024.0 * 1024.0)
    print("Build complete:", flush=True)
    for table in ("chunks", "feature_summary", "markers", "meta"):
        print("  {}: {:,} rows".format(table, counts[table]), flush=True)
    print("  database: {:.2f} MB ({})".format(size_mb, db_path), flush=True)


def main():
    args = parse_args()
    import numpy as np

    build_database(args.db, args.force, np)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create speech-boundary chunks from the 38 Folger Shakespeare TEI plays."""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


TEI_NS = "http://www.tei-c.org/ns/1.0"
TEI = "{" + TEI_NS + "}"
MAX_CHARS = 1400

ABBREVIATIONS = (
    "1H4 1H6 2H4 2H6 3H6 AWW AYL Ado Ant Cor Cym Err H5 H8 Ham JC "
    "Jn LLL Lr MM MND MV Mac Oth Per R2 R3 Rom Shr TGV TN TNK Tim Tit "
    "Tmp Tro WT Wiv"
).split()

TRAGEDIES = set("Ham Mac Lr Oth Ant Cor JC Rom Tim Tit Cym Tro".split())
COMEDIES = set("MND TN AYL Ado Err LLL MV Wiv Shr TGV MM AWW Tmp WT".split())
HISTORIES = set("1H4 2H4 H5 1H6 2H6 3H6 H8 Jn R2 R3".split())
TRAGICOMEDIES = {"Per", "TNK"}

CSV_COLUMNS = [
    "chunk_id",
    "play",
    "folger_abbr",
    "genre",
    "subgenre",
    "authorship_note",
    "act",
    "scene",
    "div_type",
    "ftln_start",
    "ftln_end",
    "n_chars",
    "n_speeches",
    "speech_split",
    "character_set",
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
    "verse_ratio",
    "contains_song",
    "text_content",
]

ALLOWED_DIV_TYPES = {
    "scene",
    "prologue",
    "epilogue",
    "chorus",
    "induction",
    "act",
}
PRIMARY_STAGE_TYPES = {"entrance", "exit", "delivery", "business", "modifier"}
SKIP_INLINE_SUBTREES = {"app", "fw"}


def local_name(element):
    """Return an ElementTree element's local tag name."""
    return element.tag.rsplit("}", 1)[-1]


def normalize_space(text):
    """Normalize Unicode and collapse all whitespace to ordinary spaces."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def semantic_text(element):
    """Extract element text without XML indentation whitespace."""
    pieces = []

    def visit(node):
        tag = local_name(node)
        if node.text:
            if tag == "c" or node.text.strip():
                pieces.append(node.text)
        for child in node:
            visit(child)
            if child.tail and child.tail.strip():
                pieces.append(child.tail)

    visit(element)
    return normalize_space("".join(pieces))


def normalize_speaker(element):
    """Uppercase a speaker label and give it exactly one final period."""
    label = semantic_text(element).strip().upper()
    if not label:
        return ""
    return label.rstrip(".") + "."


def derive_speaker(speech, parent):
    """Derive a normalized label for a speech whose speaker text is empty."""
    ancestor = parent.get(speech)
    while ancestor is not None:
        if local_name(ancestor) in {"div1", "div2"}:
            division_type = (ancestor.get("type") or "").strip().lower()
            if division_type == "epilogue":
                return "EPILOGUE."
            if division_type in {"prologue", "chorus"}:
                return "CHORUS."
            break
        ancestor = parent.get(ancestor)

    who = speech.get("who") or ""
    if " " in who:
        return "ALL."
    if who:
        return who.lstrip("#").rsplit("_", 1)[0].upper() + "."
    return "WITHIN."


def stage_text(element):
    """Render a stage direction using its complete descendant text."""
    return "[" + semantic_text(element) + "]"


def raw_stage_type(element):
    value = (element.get("type") or "none").strip().lower()
    return value or "none"


def stage_counter(element):
    """Count an outer stage plus all nested stage elements by raw type."""
    result = Counter()
    for descendant in element.iter(TEI + "stage"):
        result[raw_stage_type(descendant)] += 1
    return result


def ftlns_in(element):
    return [
        node.get("n", "")
        for node in element.iter(TEI + "milestone")
        if node.get("unit") == "ftln" and node.get("n")
    ]


def empty_meta():
    return {
        "ftlns": [],
        "stage_counts": Counter(),
        "verse_ids": set(),
        "prose_ids": set(),
        "song": False,
        "speakers": [],
        "n_speeches": 0,
    }


def merge_meta(items):
    result = empty_meta()
    for item in items:
        result["ftlns"].extend(item["ftlns"])
        result["stage_counts"].update(item["stage_counts"])
        result["verse_ids"].update(item["verse_ids"])
        result["prose_ids"].update(item["prose_ids"])
        result["song"] = result["song"] or item["song"]
        result["speakers"].extend(item["speakers"])
        result["n_speeches"] += item["n_speeches"]
    return result


def append_text_event(events, value, in_song):
    if value is None:
        return
    if value.strip() or value == " ":
        events.append(("text", value, in_song))


def flatten_inline(element, events, in_song=False):
    """Flatten primary TEI text, stage directions, and FTLN markers in order."""
    tag = local_name(element)
    in_song = in_song or (tag == "seg" and element.get("type") == "song")

    if tag in SKIP_INLINE_SUBTREES:
        return
    if tag == "stage":
        events.append(
            (
                "stage",
                stage_text(element),
                stage_counter(element),
                ftlns_in(element),
                in_song
                or any(
                    node.get("type") == "song"
                    for node in element.iter(TEI + "seg")
                ),
            )
        )
        return
    if tag == "milestone":
        if element.get("unit") == "ftln":
            events.append(
                (
                    "marker",
                    element.get("n", ""),
                    element.get("ana", ""),
                    in_song,
                )
            )
        return

    append_text_event(events, element.text, in_song)
    for child in element:
        flatten_inline(child, events, in_song)
        if child.tail and child.tail.strip():
            append_text_event(events, child.tail, in_song)


def records_from_container(element):
    """Convert an ab/p/l container to FTLN-aligned text records."""
    events = []
    flatten_inline(element, events)
    records = []
    current = {
        "pieces": [],
        "ftlns": [],
        "ana": "",
        "stage_counts": Counter(),
        "stages": [],
        "song": False,
    }

    def has_text(record):
        return bool(normalize_space("".join(record["pieces"])))

    def finish():
        nonlocal current
        text = normalize_space("".join(current["pieces"]))
        if text:
            records.append(
                {
                    "text": text,
                    "ftlns": list(current["ftlns"]),
                    "ana": current["ana"],
                    "stage_counts": Counter(current["stage_counts"]),
                    "stages": list(current["stages"]),
                    "song": current["song"],
                }
            )
        current = {
            "pieces": [],
            "ftlns": [],
            "ana": "",
            "stage_counts": Counter(),
            "stages": [],
            "song": False,
        }

    for event in events:
        if event[0] == "marker":
            _, number, ana, song = event
            if current["ftlns"] or has_text(current):
                if current["ftlns"]:
                    finish()
            current["ftlns"].append(number)
            current["ana"] = ana
            current["song"] = current["song"] or song
        elif event[0] == "stage":
            _, text, counts, ftlns, song = event
            current["pieces"].append(text)
            current["stage_counts"].update(counts)
            current["stages"].append((text, Counter(counts)))
            current["ftlns"].extend(ftlns)
            current["song"] = current["song"] or song
        else:
            _, text, song = event
            current["pieces"].append(text)
            current["song"] = current["song"] or song
    finish()
    return records


def sentence_spans(text):
    """Return prose spans ending after . ! ?, excluding punctuation in stages."""
    spans = []
    start = 0
    square_depth = 0
    for index, char in enumerate(text):
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth:
            square_depth -= 1
        elif char in ".!?" and square_depth == 0:
            end = index + 1
            if text[start:end].strip():
                spans.append((start, end))
            start = end
            while start < len(text) and text[start].isspace():
                start += 1
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def atom_meta(ftlns=None, stages=None, verse_id=None, prose_id=None, song=False):
    meta = empty_meta()
    meta["ftlns"] = list(ftlns or [])
    meta["stage_counts"] = Counter(stages or {})
    if verse_id is not None:
        meta["verse_ids"].add(verse_id)
    if prose_id is not None:
        meta["prose_ids"].add(prose_id)
    meta["song"] = song
    return meta


def prose_atoms(records, source_id, group_index):
    """Turn a consecutive prose record group into sentence-boundary atoms."""
    pieces = []
    record_ranges = []
    offset = 0
    for record in records:
        if pieces:
            pieces.append(" ")
            offset += 1
        start = offset
        pieces.append(record["text"])
        offset += len(record["text"])
        record_ranges.append((start, offset, record))
    text = "".join(pieces)
    paragraph_id = (source_id, group_index)
    marker_positions = []
    stage_positions = []
    for record_start, record_end, record in record_ranges:
        marker_positions.extend((record_start, value) for value in record["ftlns"])
        search_start = 0
        for rendered_stage, counts in record["stages"]:
            relative = record["text"].find(rendered_stage, search_start)
            if relative < 0:
                relative = 0
            stage_positions.append((record_start + relative, counts))
            search_start = relative + len(rendered_stage)
    atoms = []
    for sentence_index, (start, end) in enumerate(sentence_spans(text)):
        sentence = text[start:end].strip()
        overlapping = [
            record
            for record_start, record_end, record in record_ranges
            if record_end > start and record_start < end
        ]
        meta = atom_meta(
            ftlns=[value for position, value in marker_positions if start <= position < end],
            stages=Counter(
                {
                    key: sum(
                        counts[key]
                        for position, counts in stage_positions
                        if start <= position < end
                    )
                    for key in {
                        stage_type
                        for position, counts in stage_positions
                        if start <= position < end
                        for stage_type in counts
                    }
                }
            ),
            prose_id=paragraph_id,
            song=any(record["song"] for record in overlapping),
        )
        atoms.append(
            {
                "text": sentence,
                "sep": "\n" if sentence_index == 0 else " ",
                "meta": meta,
            }
        )
    return atoms


def container_atoms(element, source_id):
    """Map Folger ab/p/l encoding to verse-line and prose-sentence atoms."""
    tag = local_name(element)
    records = records_from_container(element)
    if not records:
        return []
    atoms = []
    prose_group = []
    group_index = 0

    def finish_prose():
        nonlocal prose_group, group_index
        if prose_group:
            group_index += 1
            atoms.extend(prose_atoms(prose_group, source_id, group_index))
            prose_group = []

    for record_index, record in enumerate(records, 1):
        ana = record["ana"]
        is_verse = tag == "l" or ana in {"#verse", "#short"}
        is_prose = tag == "p" or ana == "#prose"
        if is_verse:
            finish_prose()
            atoms.append(
                {
                    "text": record["text"],
                    "sep": "\n",
                    "meta": atom_meta(
                        ftlns=record["ftlns"],
                        stages=record["stage_counts"],
                        verse_id=(source_id, record_index),
                        song=record["song"],
                    ),
                }
            )
        elif is_prose:
            prose_group.append(record)
        else:
            # Text without lineation is prose unless it is an explicit l.
            prose_group.append(record)
    finish_prose()
    return atoms


def render_speech(prefix_lines, atoms):
    prefix = " ".join(line for line in prefix_lines if line)
    body_parts = []
    for index, atom in enumerate(atoms):
        if index:
            body_parts.append(atom["sep"])
        body_parts.append(atom["text"])
    body = "".join(body_parts).strip()
    if prefix and body:
        return prefix + " " + body
    return prefix or body


def make_stage_atom(element, source_id):
    meta = atom_meta(
        ftlns=ftlns_in(element),
        stages=stage_counter(element),
        prose_id=None,
        song=any(
            node.get("type") == "song" for node in element.iter(TEI + "seg")
        ),
    )
    return {"text": stage_text(element), "sep": "\n", "meta": meta}


def build_speech(speech, speech_serial, parent):
    """Render one speech and split it only at verse-line/prose-sentence atoms."""
    children = list(speech)
    speaker_elements = [
        child for child in children if local_name(child) == "speaker"
    ]
    speaker_element = speaker_elements[0] if speaker_elements else None
    speaker = normalize_speaker(speaker_element) if speaker_element is not None else ""
    if not any(semantic_text(element) for element in speaker_elements):
        speaker = derive_speaker(speech, parent)

    head_stages = []
    if speaker_element is not None:
        index = children.index(speaker_element) + 1
        while index < len(children) and local_name(children[index]) == "stage":
            head_stages.append(children[index])
            index += 1
    head_stage_ids = {id(element) for element in head_stages}
    prefix_lines = ([speaker] if speaker else []) + [
        stage_text(element) for element in head_stages
    ]
    prefix_meta = empty_meta()
    for element in head_stages:
        prefix_meta["ftlns"].extend(ftlns_in(element))
        prefix_meta["stage_counts"].update(stage_counter(element))
        prefix_meta["song"] = prefix_meta["song"] or any(
            node.get("type") == "song" for node in element.iter(TEI + "seg")
        )

    atoms = []
    container_number = 0
    for child in children:
        tag = local_name(child)
        if tag == "speaker" or id(child) in head_stage_ids:
            continue
        if tag in {"ab", "p", "l"}:
            container_number += 1
            atoms.extend(
                container_atoms(
                    child, ("speech", speech_serial, container_number, tag)
                )
            )
        elif tag == "stage":
            atoms.append(
                make_stage_atom(
                    child, ("speech-stage", speech_serial, container_number)
                )
            )

    # Cover unusual wrapper elements in a speech without duplicating ab/p/l content.
    if not atoms:
        for child in children:
            tag = local_name(child)
            if tag not in {"speaker", "stage", "lb"}:
                container_number += 1
                atoms.extend(
                    container_atoms(
                        child, ("speech-wrapper", speech_serial, container_number)
                    )
                )

    full_text = render_speech(prefix_lines, atoms)
    base_speech_meta = empty_meta()
    base_speech_meta["speakers"] = [speaker]
    base_speech_meta["n_speeches"] = 1
    full_meta = merge_meta(
        [prefix_meta, base_speech_meta] + [atom["meta"] for atom in atoms]
    )
    source_stage_counts = Counter(full_meta["stage_counts"])

    if len(full_text) <= MAX_CHARS:
        return (
            [{"text": full_text, "meta": full_meta, "speech_split": ""}],
            len(full_text),
            source_stage_counts,
            list(full_meta["ftlns"]),
        )

    fragments_atoms = []
    current = []
    for atom in atoms:
        candidate = current + [atom]
        if len(render_speech(prefix_lines, candidate)) <= MAX_CHARS:
            current = candidate
            continue
        if not current:
            raise ValueError(
                "Atomic verse line/prose sentence exceeds 1400 characters "
                "after the repeated speech prefix"
            )
        fragments_atoms.append(current)
        current = [atom]
        if len(render_speech(prefix_lines, current)) > MAX_CHARS:
            raise ValueError(
                "Atomic verse line/prose sentence exceeds 1400 characters "
                "after the repeated speech prefix"
            )
    if current:
        fragments_atoms.append(current)

    total = len(fragments_atoms)
    fragments = []
    for part_number, fragment_atoms in enumerate(fragments_atoms, 1):
        meta = merge_meta(
            [prefix_meta, base_speech_meta]
            + [atom["meta"] for atom in fragment_atoms]
        )
        fragments.append(
            {
                "text": render_speech(prefix_lines, fragment_atoms),
                "meta": meta,
                "speech_split": f"{part_number}/{total}",
            }
        )
    return (
        fragments,
        len(full_text),
        source_stage_counts,
        list(full_meta["ftlns"]),
    )


def build_external_stage(element):
    meta = empty_meta()
    meta["ftlns"] = ftlns_in(element)
    meta["stage_counts"] = stage_counter(element)
    meta["song"] = any(
        node.get("type") == "song" for node in element.iter(TEI + "seg")
    )
    return {"text": stage_text(element), "meta": meta, "speech_split": ""}


def direct_division_events(division):
    """Yield sp and outermost external stage elements in document order."""
    result = []

    def visit(node):
        for child in node:
            tag = local_name(child)
            if tag in {"div1", "div2"}:
                continue
            if tag == "sp":
                result.append(("speech", child))
            elif tag == "stage":
                result.append(("stage", child))
            elif tag in {"app", "fw"}:
                continue
            else:
                visit(child)

    visit(division)
    return result


def parse_int(value):
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


def division_context(division, parent):
    div_type = (division.get("type") or "").lower()
    act = None
    scene = None
    current = division
    while current is not None:
        current_type = (current.get("type") or "").lower()
        if current_type == "act" and act is None:
            act = parse_int(current.get("n"))
        if current_type == "scene" and scene is None:
            scene = parse_int(current.get("n"))
        current = parent.get(current)
    if div_type == "scene":
        scene = parse_int(division.get("n"))
    return act, scene, div_type


def genre_for(abbreviation):
    if abbreviation in TRAGEDIES:
        return "tragedy"
    if abbreviation in COMEDIES:
        return "comedy"
    if abbreviation in HISTORIES:
        return "history"
    if abbreviation in TRAGICOMEDIES:
        return "tragicomedy"
    raise ValueError("No genre mapping for " + abbreviation)


def subgenre_for(abbreviation):
    return "romance" if abbreviation in {"Per", "TNK"} else ""


def authorship_for(abbreviation, act):
    if abbreviation == "TNK":
        return "Fletcher collaboration"
    if abbreviation == "Per" and act in {1, 2}:
        return "Wilkins collaboration acts 1-2"
    return ""


def ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def row_from_chunk(
    abbreviation,
    title,
    act,
    scene,
    div_type,
    unique_number,
    part_number,
    text,
    meta,
    speech_split,
):
    stage_counts = meta["stage_counts"]
    speakers = meta["speakers"]
    characters = ordered_unique(speakers)
    speaker_changes = sum(
        left != right for left, right in zip(speakers, speakers[1:])
    )
    n_verse = len(meta["verse_ids"])
    n_prose = len(meta["prose_ids"])
    denominator = n_verse + n_prose
    other_stages = sum(
        count
        for stage_type, count in stage_counts.items()
        if stage_type not in PRIMARY_STAGE_TYPES
    )
    act_id = str(act) if act is not None else "x"
    scene_id = str(scene) if scene is not None else "x"
    chunk_id = (
        f"{abbreviation}_{div_type}_a{act_id}_s{scene_id}"
        f"_u{unique_number:03d}_c{part_number:03d}"
    )
    return {
        "chunk_id": chunk_id,
        "play": title,
        "folger_abbr": abbreviation,
        "genre": genre_for(abbreviation),
        "subgenre": subgenre_for(abbreviation),
        "authorship_note": authorship_for(abbreviation, act),
        "act": "" if act is None else act,
        "scene": "" if scene is None else scene,
        "div_type": div_type,
        "ftln_start": meta["ftlns"][0] if meta["ftlns"] else "",
        "ftln_end": meta["ftlns"][-1] if meta["ftlns"] else "",
        "n_chars": len(text),
        "n_speeches": meta["n_speeches"],
        "speech_split": speech_split,
        "character_set": ";".join(characters),
        "n_speakers": len(characters),
        "n_speaker_changes": speaker_changes,
        "n_stage_entrance": stage_counts["entrance"],
        "n_stage_exit": stage_counts["exit"],
        "n_stage_delivery": stage_counts["delivery"],
        "n_stage_business": stage_counts["business"],
        "n_stage_modifier": stage_counts["modifier"],
        "n_stage_other": other_stages,
        "n_verse_lines": n_verse,
        "n_prose_paragraphs": n_prose,
        "verse_ratio": "" if not denominator else n_verse / denominator,
        "contains_song": 1 if meta["song"] else 0,
        "text_content": text,
    }


def parse_play(xml_path, abbreviation):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    if root.tag != TEI + "TEI":
        raise ValueError(f"{xml_path}: not a TEI namespace document")
    parent = {child: node for node in root.iter() for child in node}
    title_element = root.find(
        ".//" + TEI + "titleStmt/" + TEI + "title"
    )
    if title_element is None:
        raise ValueError(f"{xml_path}: missing titleStmt/title")
    title = semantic_text(title_element)
    body = root.find(".//" + TEI + "body")
    if body is None:
        raise ValueError(f"{xml_path}: missing text/body")

    rows = []
    n_speeches = 0
    n_speeches_split = 0
    max_speech_chars = 0
    source_ftlns = []
    source_stage_counts = Counter()
    speech_serial = 0

    for division in body.iter():
        if local_name(division) not in {"div1", "div2"}:
            continue
        if (division.get("type") or "").lower() not in ALLOWED_DIV_TYPES:
            continue
        events = direct_division_events(division)
        if not events:
            continue
        act, scene, div_type = division_context(division, parent)
        unique_number = 0
        buffer = []

        def emit_buffer():
            nonlocal unique_number, buffer
            if not buffer:
                return
            unique_number += 1
            text = "\n\n".join(item["text"] for item in buffer if item["text"])
            meta = merge_meta([item["meta"] for item in buffer])
            if not text:
                buffer = []
                return
            if len(text) > MAX_CHARS:
                raise ValueError(
                    f"{abbreviation} {div_type}: packed chunk exceeds 1400"
                )
            rows.append(
                row_from_chunk(
                    abbreviation,
                    title,
                    act,
                    scene,
                    div_type,
                    unique_number,
                    1,
                    text,
                    meta,
                    "",
                )
            )
            buffer = []

        for event_type, element in events:
            if event_type == "stage":
                item = build_external_stage(element)
                source_stage_counts.update(item["meta"]["stage_counts"])
                source_ftlns.extend(item["meta"]["ftlns"])
                candidate_text = "\n\n".join(
                    value["text"] for value in buffer + [item] if value["text"]
                )
                if len(candidate_text) > MAX_CHARS:
                    emit_buffer()
                if len(item["text"]) > MAX_CHARS:
                    raise ValueError(
                        f"{abbreviation}: external stage exceeds 1400"
                    )
                buffer.append(item)
                continue

            speech_serial += 1
            n_speeches += 1
            (
                fragments,
                full_length,
                speech_stages,
                speech_ftlns,
            ) = build_speech(element, speech_serial, parent)
            max_speech_chars = max(max_speech_chars, full_length)
            source_stage_counts.update(speech_stages)
            source_ftlns.extend(speech_ftlns)

            if len(fragments) > 1:
                n_speeches_split += 1
                emit_buffer()
                unique_number += 1
                for part_number, fragment in enumerate(fragments, 1):
                    rows.append(
                        row_from_chunk(
                            abbreviation,
                            title,
                            act,
                            scene,
                            div_type,
                            unique_number,
                            part_number,
                            fragment["text"],
                            fragment["meta"],
                            fragment["speech_split"],
                        )
                    )
                continue

            item = fragments[0]
            candidate_text = "\n\n".join(
                value["text"] for value in buffer + [item] if value["text"]
            )
            if len(candidate_text) > MAX_CHARS:
                emit_buffer()
            buffer.append(item)
        emit_buffer()

    if not rows:
        raise ValueError(f"{abbreviation}: no chunks generated")
    return {
        "rows": rows,
        "n_speeches": n_speeches,
        "n_speeches_split": n_speeches_split,
        "max_speech_chars": max_speech_chars,
        "source_ftlns": source_ftlns,
        "source_stage_counts": source_stage_counts,
    }


def resolve_input(corpus_root, abbreviation):
    matches = sorted(
        (corpus_root / abbreviation).glob(
            "*_XML_FolgerShakespeare/" + abbreviation + ".xml"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{abbreviation}: expected one Folger XML, found {len(matches)}"
        )
    return matches[0]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(all_rows, per_play_results):
    per_play = {}
    per_genre = {
        genre: {"n_plays": 0, "n_chunks": 0, "total_chars": 0}
        for genre in ("tragedy", "comedy", "history", "tragicomedy")
    }
    stage_totals = Counter()
    for abbreviation in ABBREVIATIONS:
        result = per_play_results[abbreviation]
        rows = result["rows"]
        genre = genre_for(abbreviation)
        ftlns = result["source_ftlns"]
        per_play[abbreviation] = {
            "genre": genre,
            "n_chunks": len(rows),
            "n_speeches": result["n_speeches"],
            "max_speech_chars": result["max_speech_chars"],
            "n_speeches_split": result["n_speeches_split"],
            "ftln_range": [ftlns[0], ftlns[-1]] if ftlns else ["", ""],
        }
        per_genre[genre]["n_plays"] += 1
        per_genre[genre]["n_chunks"] += len(rows)
        per_genre[genre]["total_chars"] += sum(
            int(row["n_chars"]) for row in rows
        )
        stage_totals.update(result["source_stage_counts"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_plays": len(per_play),
        "n_chunks": len(all_rows),
        "n_speeches_total": sum(
            result["n_speeches"] for result in per_play_results.values()
        ),
        "n_speeches_split": sum(
            result["n_speeches_split"]
            for result in per_play_results.values()
        ),
        "max_chunk_chars": max(int(row["n_chars"]) for row in all_rows),
        "chunks_over_1400": sum(
            int(row["n_chars"]) > MAX_CHARS for row in all_rows
        ),
        "per_play": per_play,
        "per_genre": per_genre,
        "stage_type_totals": dict(sorted(stage_totals.items())),
    }


def default_paths():
    repository = Path(__file__).resolve().parents[2]
    return {
        "corpus_root": repository / "じんもんこん" / "corpus_raw",
        "out_csv": repository
        / "じんもんこん2026"
        / "chunks"
        / "shakespeare_chunks_v2.csv",
        "out_summary": repository
        / "じんもんこん2026"
        / "reports"
        / "parse_summary.json",
        "out_length_dist": repository
        / "じんもんこん2026"
        / "reports"
        / "chunk_length_distribution.csv",
    }


def parse_args():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=defaults["corpus_root"])
    parser.add_argument("--out-csv", type=Path, default=defaults["out_csv"])
    parser.add_argument(
        "--out-summary", type=Path, default=defaults["out_summary"]
    )
    parser.add_argument(
        "--out-length-dist", type=Path, default=defaults["out_length_dist"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    per_play_results = {}
    all_rows = []
    for abbreviation in ABBREVIATIONS:
        xml_path = resolve_input(args.corpus_root, abbreviation)
        result = parse_play(xml_path, abbreviation)
        per_play_results[abbreviation] = result
        all_rows.extend(result["rows"])

    write_csv(args.out_csv, all_rows, CSV_COLUMNS)
    length_rows = [
        {
            "folger_abbr": row["folger_abbr"],
            "chunk_id": row["chunk_id"],
            "n_chars": row["n_chars"],
            "n_speeches": row["n_speeches"],
            "speech_split": row["speech_split"],
        }
        for row in all_rows
    ]
    write_csv(
        args.out_length_dist,
        length_rows,
        [
            "folger_abbr",
            "chunk_id",
            "n_chars",
            "n_speeches",
            "speech_split",
        ],
    )
    summary = build_summary(all_rows, per_play_results)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"plays: {summary['n_plays']}")
    print(f"chunks: {summary['n_chunks']}")
    print(f"speeches: {summary['n_speeches_total']}")
    print(f"split speeches: {summary['n_speeches_split']}")
    print(f"max chunk chars: {summary['max_chunk_chars']}")
    print(f"chunks over 1400: {summary['chunks_over_1400']}")


if __name__ == "__main__":
    main()

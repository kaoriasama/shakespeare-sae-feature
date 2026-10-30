# Shakespeare SAE Feature

## Overview

This repository contains data and code for mapping sparse-autoencoder (SAE) feature landscapes across 38 Shakespeare plays. It accompanies a paper for Jinmonkon 2026; the camera-ready paper and its complete citation are forthcoming.

## Repository layout

- `chunks/`: the analysis-ready Folger TEI chunk table.
- `corpus_raw/`: provenance and instructions for acquiring the source TEI; the TEI files are not redistributed.
- `activations_v2/`: placeholder for activation matrices distributed separately.
- `reports/`: derived tables and parser summaries.
- `scripts/`: acquisition, parsing, SAE encoding, analysis, and verification code.
- `docs/`: a concise methods description.

## Data availability

The activation data are distributed through Zenodo: [10.5281/zenodo.22112879](https://doi.org/10.5281/zenodo.22112879). The record holds 19 files: six firing-rate matrices, six mean-activation matrices, the summary database (`features.db`), marker-feature records (`markers.json`), residual-stream samples for ten chunks per model, sanitized run metadata, SHA-256 checksums, and a README that defines every file and field. Download the twelve matrix files, and `features.db` if you want the summary views, and place them in `activations_v2/` before running the analysis scripts.

## Reproduction

1. Fetch the Folger TEI corpus:

   ```bash
   bash scripts/fetch_folger_tei.sh corpus_raw
   ```

2. Verify its structure:

   ```bash
   python scripts/verify_tei_structure.py --corpus-root corpus_raw
   ```

3. Rebuild the chunks and parser reports:

   ```bash
   python scripts/tei_chunk_parser_v2.py --corpus-root corpus_raw --out-csv chunks/shakespeare_chunks_v2.csv --out-summary reports/parse_summary.json --out-length-dist reports/chunk_length_distribution.csv
   ```

4. Run SAE encoding with a GPU, authenticated access to the model repository, the base models, and the Gemma Scope weights. Obtain the G2 weights from Hugging Face repository `google/gemma-scope-2b-pt-res`: `layer_4/width_16k/average_l0_124`, `layer_11/width_16k/average_l0_80`, and `layer_22/width_16k/average_l0_72`. Obtain the G3 weights from `google/gemma-scope-2-4b-pt`: `resid_post/layer_9_width_16k_l0_medium`, `resid_post/layer_17_width_16k_l0_medium`, and `resid_post/layer_29_width_16k_l0_medium`. Place the downloaded weights under `sae_weights/` at the repository root. See the constants in `scripts/sae_pipeline/run_sae_encode.py` for the exact expected paths.

5. Once the matrices are in `activations_v2/`, the scripts under `scripts/analysis/` require only the NPZ matrices and the chunks CSV.

## Integrity

The SHA-256 prefix (first 16 hexadecimal characters) of `chunks/shakespeare_chunks_v2.csv` is `d6c48d0934a30e96`. The corresponding prefix for `tei_chunk_parser_v2.py` is `2735704efd255fd5`.

## Camera-ready additions (October 2026)

The following scripts and tables were added for the camera-ready version of the Jinmonkon 2026 paper. All scripts run with Python 3.9 and NumPy only, read the activation matrices and `chunks/shakespeare_chunks_v2.csv` without modifying them, and are deterministic (fixed seeds where randomness is used).
The activation matrices read by these scripts (`activations_v2/*_firing.npz`) are distributed in the Zenodo record (https://doi.org/10.5281/zenodo.22112879), not in this repository.

### Scripts (`scripts/analysis/`)

| File | Purpose |
|---|---|
| `length_equalized_patterns.py` | Recounts the three-way classification (fires in all plays / in some / in none) and the per-play counts of single-play and complement (all-but-one-play) features after equalizing play length: every play is subsampled to the character budget of the shortest play (88,675 characters) by drawing chunks in random order until the budget is reached; 1,000 draws, seed 20261007. Also reports Spearman correlations between play length and the counts before and after equalization. |
| `genre_proximity_step1.py` | For features firing in 8–30 plays (presence probability ≥ 0.5 under length equalization), measures the Jaccard similarity between the set of firing plays and the First Folio's comedies, histories, and tragedies, counts features with Jaccard ≥ 0.6 / 0.7 / 0.8, and compares the counts with a null distribution of 10,000 random play sets of the same sizes (seed 20261007; p = (k + 1) / 10,001, where k is the number of draws reaching the observed count). |
| `reference_groups.json` | The three reference groupings over the 36 Folio plays (Pericles and The Two Noble Kinsmen excluded). |
| `list_pattern_features.py` | For selected features (near a Folio genre, silent in exactly one play, or firing in exactly one play), lists the three chunks with the highest firing rate, verse/prose statistics from the chunk table, and Neuronpedia metadata (`--neuronpedia fetch` queries the public API; `cache` reads a saved JSON; `off` skips). |
| `single_work_feature_sparsity.py` | For features firing in exactly one play, counts how many fire in only 1 / 2 / 3–5 / 6+ chunks of the whole corpus, and the median firing rate within fired chunks. |
| `verse_index_by_work.py` | Per-play verse/prose leaning from the chunk table: character-weighted mean of `verse_ratio` (= verse lines / (verse lines + prose paragraphs)) and the shares of prose-only and verse-only chunks. |

Run order: `length_equalized_patterns.py` (writes `equalized_presence_*.npz`), then `genre_proximity_step1.py`, then `list_pattern_features.py --neuronpedia cache` (requires `pattern_feature_neuronpedia.json` from the Zenodo record, https://doi.org/10.5281/zenodo.22112879). The other two scripts are independent.

### Tables (`reports/`)

| File | Contents |
|---|---|
| `length_equalized_by_work.csv` | Per layer and play: total chunks and characters, mean chunks used, and single-play / complement counts in full data and under equalization (mean, 2.5th and 97.5th percentiles). |
| `length_equalized_classification.csv` | Per layer: three-way classification in full data and equalized means; Spearman correlations with play length. |
| `length_equalized_samples_first3.json` | Chunk IDs drawn in the first three equalization draws, for verification. |
| `length_equalized_reps_{model}_L{layer}.npz` | Per draw (1,000 × 38): single-play counts, complement counts, classification, chunks used. |
| `equalized_presence_{model}_L{layer}.npz` | Per play and feature (38 × 16,384): probability of firing under equalization (`presence_prob`) and firing in full data (`presence_obs`). |
| `genre_proximity_step1.csv`, `genre_proximity_step1_best.csv` | Proximity counts, null statistics and p-values per layer and genre; the five closest features per cell with missing and extra plays. |
| `pattern_feature_ids.csv` | The 505 selected feature rows with firing statistics, the three strongest chunks, verse/prose statistics and Neuronpedia summary fields. |
| `pattern_feature_excerpts.txt` | The text of the three strongest chunks for each selected feature, quoted verbatim from the Folger Shakespeare (see Source-text attribution). |
| `single_work_feature_sparsity.csv`, `verse_index_by_work.csv` | Outputs of the two scripts above (the latter begins with comment lines defining the index). |

Notes.
- The Neuronpedia API responses used for `pattern_feature_ids.csv` (505 features, about 85 MB) are archived in the Zenodo record (https://doi.org/10.5281/zenodo.22112879) as `pattern_feature_neuronpedia.json`; Neuronpedia content may change over time, so that snapshot, not a fresh fetch, reproduces the paper.
- `verse_ratio` in `chunks/shakespeare_chunks_v2.csv` is written at full double precision. Some spreadsheet applications import the longer decimals as text; use a CSV importer or `VALUE()` before computing with this column.

## License

Code is licensed under the MIT License. Data and derived data are covered by [LICENSE-DATA.md](LICENSE-DATA.md).

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff). The paper citation will be added with the camera-ready release.

## 日本語要約

本リポジトリは、シェイクスピア戯曲38作品を対象にSAE特徴量の発火分布を観察する、じんもんこん2026論文のデータと再現コードです。原典TEIと大容量の発火行列は同梱せず、取得手順と別配布先を示しています。カメラレディ論文と正式な引用情報は追って追加します。

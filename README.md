# Shakespeare SAE Feature Landscape

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

The firing and mean-activation matrices (`*.npz`) and the summary database are distributed through Zenodo: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX). Download the matrix files and place them in `activations_v2/` before running the analysis scripts.

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

## License

Code is licensed under the MIT License. Data and derived data are covered by [LICENSE-DATA.md](LICENSE-DATA.md).

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff). The paper citation will be added with the camera-ready release.

## 日本語要約

本リポジトリは、シェイクスピア戯曲38作品を対象にSAE特徴量の発火分布を観察する、じんもんこん2026論文のデータと再現コードです。原典TEIと大容量の発火行列は同梱せず、取得手順と別配布先を示しています。カメラレディ論文と正式な引用情報は追って追加します。

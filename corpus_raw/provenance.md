# Folger Shakespeare Library TEI XML provenance

- Retrieval date: 2026-07-19
- Source: https://www.folger.edu/explore/shakespeares-works/download/ (current official Folger Shakespeare Library download page; supersedes the legacy folgerdigitaltexts.org site whose direct zip links are broken as of 2026-07-19)
- License notice (verbatim): "These files are free to use for all non-commercial purposes."
- Terms note: The full Terms of Use text was not automatically retrieved (it is linked separately from the page); only the on-page statement quoted above is confirmed.

Each supplied `flgr.sh` URL was downloaded with `curl -L` and a browser-like User-Agent. All 15 requests resolved to `folger-main-site-assets.s3.amazonaws.com`, returned HTTP 200 with `Content-Type: application/zip`, passed `unzip -tq`, and were fully extracted. The primary file below is the largest extracted `.xml` whose parsed root element is `{http://www.tei-c.org/ns/1.0}TEI`; each primary file was independently parsed with Python `xml.etree.ElementTree` and passed well-formedness validation.

The archives also contain `__MACOSX/.../._*.xml` AppleDouble metadata files. These were retained because the full archive contents were requested; despite their filename extension they are not XML documents and are not primary TEI candidates.

| Title | Abbreviation | Genre | Subgenre | flgr.sh URL used | Final resolved download URL | Download status | Extracted primary XML file path | Well-formedness result |
|---|---|---|---|---|---|---|---|---|
| Hamlet | Ham | tragedy | (none) | https://flgr.sh/txtfssHamxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/hamlet_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 764832 bytes; ZIP test and extraction passed | `Ham/hamlet_XML_FolgerShakespeare/Ham.xml` | PASS — ElementTree; TEI namespace root |
| Macbeth | Mac | tragedy | (none) | https://flgr.sh/txtfssMacxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/macbeth_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 466298 bytes; ZIP test and extraction passed | `Mac/macbeth_XML_FolgerShakespeare/Mac.xml` | PASS — ElementTree; TEI namespace root |
| King Lear | Lr | tragedy | (none) | https://flgr.sh/txtfssLr_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/king-lear_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 686069 bytes; ZIP test and extraction passed | `Lr/king-lear_XML_FolgerShakespeare/Lr.xml` | PASS — ElementTree; TEI namespace root |
| Othello | Oth | tragedy | (none) | https://flgr.sh/txtfssOthxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/othello_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 681167 bytes; ZIP test and extraction passed | `Oth/othello_XML_FolgerShakespeare/Oth.xml` | PASS — ElementTree; TEI namespace root |
| A Midsummer Night's Dream | MND | comedy | (none) | https://flgr.sh/txtfssMNDxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/a-midsummer-nights-dream_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 454923 bytes; ZIP test and extraction passed | `MND/a-midsummer-nights-dream_XML_FolgerShakespeare/MND.xml` | PASS — ElementTree; TEI namespace root |
| Twelfth Night | TN | comedy | (none) | https://flgr.sh/txtfssTN_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/twelfth-night_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 524421 bytes; ZIP test and extraction passed | `TN/twelfth-night_XML_FolgerShakespeare/TN.xml` | PASS — ElementTree; TEI namespace root |
| As You Like It | AYL | comedy | (none) | https://flgr.sh/txtfssAYLxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/as-you-like-it_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 555364 bytes; ZIP test and extraction passed | `AYL/as-you-like-it_XML_FolgerShakespeare/AYL.xml` | PASS — ElementTree; TEI namespace root |
| Much Ado About Nothing | Ado | comedy | (none) | https://flgr.sh/txtfssAdoxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/much-ado-about-nothing_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 544410 bytes; ZIP test and extraction passed | `Ado/much-ado-about-nothing_XML_FolgerShakespeare/Ado.xml` | PASS — ElementTree; TEI namespace root |
| The Winter's Tale | WT | tragicomedy | romance | https://flgr.sh/txtfssWT_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-winters-tale_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 625441 bytes; ZIP test and extraction passed | `WT/the-winters-tale_XML_FolgerShakespeare/WT.xml` | PASS — ElementTree; TEI namespace root |
| The Tempest | Tmp | tragicomedy | romance | https://flgr.sh/txtfssTmpxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-tempest_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 455863 bytes; ZIP test and extraction passed | `Tmp/the-tempest_XML_FolgerShakespeare/Tmp.xml` | PASS — ElementTree; TEI namespace root |
| Cymbeline | Cym | tragicomedy | romance | https://flgr.sh/txtfssCymxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/cymbeline_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 695895 bytes; ZIP test and extraction passed | `Cym/cymbeline_XML_FolgerShakespeare/Cym.xml` | PASS — ElementTree; TEI namespace root |
| Pericles | Per | tragicomedy | romance | https://flgr.sh/txtfssPerxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/pericles_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 488819 bytes; ZIP test and extraction passed | `Per/pericles_XML_FolgerShakespeare/Per.xml` | PASS — ElementTree; TEI namespace root |
| Measure for Measure | MM | tragicomedy | problem_play | https://flgr.sh/txtfssMM_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/measure-for-measure_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 567607 bytes; ZIP test and extraction passed | `MM/measure-for-measure_XML_FolgerShakespeare/MM.xml` | PASS — ElementTree; TEI namespace root |
| All's Well That Ends Well | AWW | tragicomedy | problem_play | https://flgr.sh/txtfssAWWxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/alls-well-that-ends-well_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 586825 bytes; ZIP test and extraction passed | `AWW/alls-well-that-ends-well_XML_FolgerShakespeare/AWW.xml` | PASS — ElementTree; TEI namespace root |
| Troilus and Cressida | Tro | tragicomedy | problem_play | https://flgr.sh/txtfssTroxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/troilus-and-cressida_XML_FolgerShakespeare.zip | Success — HTTP 200, application/zip, 678873 bytes; ZIP test and extraction passed | `Tro/troilus-and-cressida_XML_FolgerShakespeare/Tro.xml` | PASS — ElementTree; TEI namespace root |

## License (from TEI headers)

> Distributed under a Creative Commons Attribution-NonCommercial 3.0 Unported License

Target: [http://creativecommons.org/licenses/by-nc/3.0/deed.en_US](http://creativecommons.org/licenses/by-nc/3.0/deed.en_US). The notice comes from `teiHeader/publicationStmt/availability/licence` in each downloaded primary XML and was verified on 2026-07-19. The identical licence text and target were present in 15/15 primary XML files. This is consistent with folger.edu's statement that the files are "free to use for all non-commercial purposes".

## Terms of Use 全文の要点(2026-07-19 追補・Claudeが取得)

当初JSモーダルのため未取得だったが、`https://www.folger.edu/copyright-policy/` から本文を取得。要点は以下(原文引用):

- **ライセンス**: Creative Commons Attribution-NonCommercial 3.0 Unported(TEIヘッダ内の宣言と一致)。
- **許諾範囲**: "anyone who wants to read, search, share, remix, transform, and build on Folger Digital Texts is free to do so" — 改変・派生物作成・再配布が明示的に許可されている(=チャンク化コーパスの作成・共有は条項内)。
- **帰属表示義務**: "Any time you create a new version or distribute Folger Digital Texts ... you must credit Folger Digital Texts as the original source", かつ "provide a link to the Creative Commons license, and state whether any changes were made"。→ 派生データ(本チャンクコーパス)を公開する場合、(a)出典としてFolger Digital Textsを明記、(b)CCライセンスへのリンク、(c)改変(チャンク化・パラテキスト除去等)を行った旨の明示、の3点が必要。
- **非商用条項**: "you may not use the material from Folger Digital Texts for commercial purposes"。non-commercialの厳密な定義は文書内になく、不明な場合は editions@folger.edu へ照会するよう案内。
- **学術利用の特則**: なし(一般のCC BY-NC 3.0に従う)。じんもんこん2026(非商用の学術発表)は条項内と解釈できるが、データ自体をリポジトリ公開する場合は上記の帰属3点を満たすこと。

参照: https://www.folger.edu/copyright-policy/ (2026-07-19取得) / ライセンス本体 http://creativecommons.org/licenses/by-nc/3.0/deed.en_US

---

## 追補: 残り23戯曲の取得 (2026-07-28)

- Retrieval date: 2026-07-28
- Source: https://www.folger.edu/explore/shakespeares-works/download/
- 取得方法・検証手順は上記15作と同一 (`curl -L` + ブラウザ相当UA → `unzip -tq` → TEI名前空間ルートを ElementTree でパース検証)。
- Folger の全ラインナップは戯曲38 + 詩4。Folio 1623 の36作に Pericles と The Two Noble Kinsmen を加えたもので、Edward III は含まれない。Henry VI は3部個別。既存15作を除く23作を本追補で取得した。
- ジャンル欄は First Folio (1623) の Catalogue に拠る。The Two Noble Kinsmen は Folio 未収録のため Folger/現代の通例に従い romance とした (Fletcher との共作。分析時は要注記)。
- 全23作の licence 宣言は既存15作と同一 (`teiHeader/publicationStmt/availability/licence` を全件確認)。

| Abbr | Title | Genre | Subgenre | flgr.sh URL | Resolved URL | zip bytes | zip SHA-256 | Primary XML | Validation |
|---|---|---|---|---|---|---|---|---|---|
| 1H4 | Henry IV, Part I | history | (none) | https://flgr.sh/txtfss1H4xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-iv-part-1_XML_FolgerShakespeare.zip | 604697 | `e175a437080ca775…` | `1H4/henry-iv-part-1_XML_FolgerShakespeare/1H4.xml` | PASS — ElementTree; TEI namespace root |
| 1H6 | Henry VI, Part 1 | history | (none) | https://flgr.sh/txtfss1H6xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-vi-part-1_XML_FolgerShakespeare.zip | 553793 | `8e19f0c889032b35…` | `1H6/henry-vi-part-1_XML_FolgerShakespeare/1H6.xml` | PASS — ElementTree; TEI namespace root |
| 2H4 | Henry IV, Part 2 | history | (none) | https://flgr.sh/txtfss2H4xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-iv-part-2_XML_FolgerShakespeare.zip | 661329 | `b89ecf9758332952…` | `2H4/henry-iv-part-2_XML_FolgerShakespeare/2H4.xml` | PASS — ElementTree; TEI namespace root |
| 2H6 | Henry VI, Part 2 | history | (none) | https://flgr.sh/txtfss2H6xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-vi-part-2_XML_FolgerShakespeare.zip | 641689 | `fb026026a041266d…` | `2H6/henry-vi-part-2_XML_FolgerShakespeare/2H6.xml` | PASS — ElementTree; TEI namespace root |
| 3H6 | Henry VI, Part 3 | history | (none) | https://flgr.sh/txtfss3H6xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-vi-part-3_XML_FolgerShakespeare.zip | 613959 | `9497c812d122224e…` | `3H6/henry-vi-part-3_XML_FolgerShakespeare/3H6.xml` | PASS — ElementTree; TEI namespace root |
| Ant | Antony and Cleopatra | tragedy | (none) | https://flgr.sh/txtfssAntxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/antony-and-cleopatra_XML_FolgerShakespeare.zip | 645087 | `b46e991a8a3f7432…` | `Ant/antony-and-cleopatra_XML_FolgerShakespeare/Ant.xml` | PASS — ElementTree; TEI namespace root |
| Cor | Coriolanus | tragedy | (none) | https://flgr.sh/txtfssCorxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/coriolanus_XML_FolgerShakespeare.zip | 688856 | `f5327f722d0ead6d…` | `Cor/coriolanus_XML_FolgerShakespeare/Cor.xml` | PASS — ElementTree; TEI namespace root |
| Err | The Comedy of Errors | comedy | (none) | https://flgr.sh/txtfssErrxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-comedy-of-errors_XML_FolgerShakespeare.zip | 416327 | `42f84e3ee96bbab8…` | `Err/the-comedy-of-errors_XML_FolgerShakespeare/Err.xml` | PASS — ElementTree; TEI namespace root |
| H5 | Henry V | history | (none) | https://flgr.sh/txtfssH5_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-v_XML_FolgerShakespeare.zip | 641060 | `1b8f2442d0bf402b…` | `H5/henry-v_XML_FolgerShakespeare/H5.xml` | PASS — ElementTree; TEI namespace root |
| H8 | Henry VIII | history | (none) | https://flgr.sh/txtfssH8_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/henry-viii_XML_FolgerShakespeare.zip | 611630 | `3096aa06d8e07070…` | `H8/henry-viii_XML_FolgerShakespeare/H8.xml` | PASS — ElementTree; TEI namespace root |
| JC | Julius Caesar | tragedy | (none) | https://flgr.sh/txtfssJC_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/julius-caesar_XML_FolgerShakespeare.zip | 514106 | `ac9a97ee3d8bb617…` | `JC/julius-caesar_XML_FolgerShakespeare/JC.xml` | PASS — ElementTree; TEI namespace root |
| Jn | King John | history | (none) | https://flgr.sh/txtfssJn_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/king-john_XML_FolgerShakespeare.zip | 531658 | `34537ea2d74f213b…` | `Jn/king-john_XML_FolgerShakespeare/Jn.xml` | PASS — ElementTree; TEI namespace root |
| LLL | Love’s Labor’s Lost | comedy | (none) | https://flgr.sh/txtfssLLLxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/loves-labors-lost_XML_FolgerShakespeare.zip | 570779 | `6775cb522f1cf834…` | `LLL/loves-labors-lost_XML_FolgerShakespeare/LLL.xml` | PASS — ElementTree; TEI namespace root |
| MV | The Merchant of Venice | comedy | (none) | https://flgr.sh/txtfssMV_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-merchant-of-venice_XML_FolgerShakespeare.zip | 541140 | `ab5ab956b6d93572…` | `MV/the-merchant-of-venice_XML_FolgerShakespeare/MV.xml` | PASS — ElementTree; TEI namespace root |
| R2 | Richard II | history | (none) | https://flgr.sh/txtfssR2_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/richard-ii_XML_FolgerShakespeare.zip | 570757 | `17c566aac50a46fa…` | `R2/richard-ii_XML_FolgerShakespeare/R2.xml` | PASS — ElementTree; TEI namespace root |
| R3 | Richard III | history | (none) | https://flgr.sh/txtfssR3_xml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/richard-iii_XML_FolgerShakespeare.zip | 721677 | `17e1b8dd1e188168…` | `R3/richard-iii_XML_FolgerShakespeare/R3.xml` | PASS — ElementTree; TEI namespace root |
| Rom | Romeo and Juliet | tragedy | (none) | https://flgr.sh/txtfssRomxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/romeo-and-juliet_XML_FolgerShakespeare.zip | 617576 | `e41c01df88757b07…` | `Rom/romeo-and-juliet_XML_FolgerShakespeare/Rom.xml` | PASS — ElementTree; TEI namespace root |
| Shr | The Taming of the Shrew | comedy | (none) | https://flgr.sh/txtfssShrxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-taming-of-the-shrew_XML_FolgerShakespeare.zip | 557907 | `5c86ced0101f8d35…` | `Shr/the-taming-of-the-shrew_XML_FolgerShakespeare/Shr.xml` | PASS — ElementTree; TEI namespace root |
| TGV | The Two Gentlemen of Verona | comedy | (none) | https://flgr.sh/txtfssTGVxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-two-gentlemen-of-verona_XML_FolgerShakespeare.zip | 465582 | `f12223c657180e93…` | `TGV/the-two-gentlemen-of-verona_XML_FolgerShakespeare/TGV.xml` | PASS — ElementTree; TEI namespace root |
| TNK | The Two Noble Kinsmen | tragicomedy | romance | https://flgr.sh/txtfssTNKxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-two-noble-kinsmen_XML_FolgerShakespeare.zip | 618858 | `d078a9d2366409b2…` | `TNK/the-two-noble-kinsmen_XML_FolgerShakespeare/TNK.xml` | PASS — ElementTree; TEI namespace root |
| Tim | Timon of Athens | tragedy | (none) | https://flgr.sh/txtfssTimxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/timon-of-athens_XML_FolgerShakespeare.zip | 494885 | `6c1b9ad467c7338f…` | `Tim/timon-of-athens_XML_FolgerShakespeare/Tim.xml` | PASS — ElementTree; TEI namespace root |
| Tit | Titus Andronicus | tragedy | (none) | https://flgr.sh/txtfssTitxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/titus-andronicus_XML_FolgerShakespeare.zip | 530614 | `c93051c189b72791…` | `Tit/titus-andronicus_XML_FolgerShakespeare/Tit.xml` | PASS — ElementTree; TEI namespace root |
| Wiv | The Merry Wives of Windsor | comedy | (none) | https://flgr.sh/txtfssWivxml | https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/the-merry-wives-of-windsor_XML_FolgerShakespeare.zip | 581126 | `a926505b391d2ab6…` | `Wiv/the-merry-wives-of-windsor_XML_FolgerShakespeare/Wiv.xml` | PASS — ElementTree; TEI namespace root |

### ライセンス (全23作の TEI ヘッダから)

> Distributed under a Creative Commons Attribution-NonCommercial 3.0 Unported License

Target: http://creativecommons.org/licenses/by-nc/3.0/deed.en_US — 23/23 で同一文言・同一ターゲットを確認 (2026-07-28)。
既存15作と合わせ、Folger 戯曲38作すべてが同一ライセンス下にある。

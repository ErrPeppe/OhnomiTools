# OhnomiTools

**OhnomiTools** is a modular Python library developed to investigate the evolutionary, topological, and multi-omic signatures of **1R/2R Whole Genome Duplications (WGD)** in human **long non-coding RNAs (lncRNAs)**, comparing WGD-associated lncRNAs against ancient Small Scale Duplications (SSD) and phylostratigraphic controls.

Built with **Polars**, **Numba JIT**, **SciPy**, and **Apache Parquet**, the pipeline is designed for reproducible execution on Google Colab or local Linux/WSL environments using a Medallion data architecture (`bronze` -> `silver` -> `gold`).

---

## Scientific Scope

Extending the quantitative framework developed for human coding ohnologs and intragenic ohno-miRNAs (Mottes et al., 2021; Agasso et al., 2025) to lncRNAs, `OhnomiTools` evaluates three complementary analytical cohorts in parallel across all phases:

1. **Lane 1 — Intragenic Passenger Model (Branch A):** lncRNA pairs hosted within WGD vs. ancient SSD protein-coding gene pairs (last common ancestor older than *Sarcopterygii*) with conserved relative strand orientation (`sense`/`antisense`).
2. **Lane 2 — Phylostratigraphically Filtered Intragenic Pairs ($A \cap B$):** Lane 1 pairs filtered by evolutionary age in LncBook v2.0 (reference threshold: both members $\ge 15.$ *Amniota*) with explicit host-length confounding controls (Mantel--Haenszel stratification and multivariate logistic regression).
3. **Lane 3 — Ancestral Phylostratigraphic Cohort (`17. Euteleostomi`, Branch B):** all human lncRNAs emerging at the bony-vertebrate root, stratified by genomic context (`intragenic_WGD_host`, `intragenic_other_host`, `intergenic_lincRNA`) against younger controls (`12. Eutheria`, `1. Homo`).

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ErrPeppe/OhnomiTools.git
```

For Google Colab usage (automatic storage setup and status summary):

```python
from ohnomitools import setup_colab
PATHS = setup_colab()
```

---

## Repository Structure

```text
OhnomiTools/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── ohnomitools/
│       ├── __init__.py      # Public API: PATHS, ProjectPaths, setup_colab
│       ├── config.py        # Medallion paths (bronze/silver/gold) and Colab/Drive setup
│       ├── fetcher.py       # ETL pipeline: LncBook v2.0, OHNOLOGS, Ensembl BioMart, STRING v12.0
│       ├── evolution.py     # Intragenic mapping, host-length controls, A ∩ B sensitivity curve
│       ├── stats.py         # Numba-accelerated bipartite Maslov-Sneppen null model & motif counters
│       ├── network.py       # Degree distributions, Sorensen-Dice, V-motifs, Bifans, PPI-Delta
│       ├── expression.py    # 9-context expression divergence, differential methylation, SmProt
│       └── reporting.py     # Automated LaTeX metrics and booktabs table export (gold/latex_auto/)
└── notebooks/
    ├── 01_data_ingestion.ipynb
    ├── 02_wgd_vs_ssd_evolution.ipynb
    ├── 03_network_motifs_null_models.ipynb
    └── 04_expression_epigenetics_robustness.ipynb
```

---

## Analysis Notebooks

| Notebook | Phase | Main Contents |
| :--- | :---: | :--- |
| `01_data_ingestion.ipynb` | Phase 1 | Normalization of LncBook v2.0, Ohnologs.com, Ensembl BioMart (pure SSD via anti-join), and STRING v12.0 into Zstandard-compressed Parquet files. |
| `02_wgd_vs_ssd_evolution.ipynb` | Phase 2 | Intragenic WGD vs. SSD Fisher tests, host-length bias controls (Mann--Whitney U, Mantel--Haenszel, Logistic Regression), and $A \cap B$ sensitivity curve. |
| `03_network_motifs_null_models.ipynb` | Phase 3 | Bipartite network topology on LncBook RBP (+ STRING PPI) and miRNA layers, Sørensen--Dice similarity, V-motifs, Bifans, PPI-Delta, and Maslov--Sneppen null models. |
| `04_expression_epigenetics_robustness.ipynb` | Phase 4 | Multi-context expression breadth, pairwise Cosine & Manhattan divergence, disease differential methylation, and `SmProt` micropeptide profiling. |

---

## Data Sources & Key References

* **Agasso, L., Molineris, I., & Caselle, M. (2025).** *Ohno-miRNAs: miRNA pairs derived from whole-genome duplication*. PLOS Computational Biology, 21(12), e1013766.
* **Mottes, F., Villa, C., Osella, M., & Caselle, M. (2021).** *The impact of whole genome duplications on the human gene regulatory networks*. PLOS Computational Biology, 17(12), e1009638.
* **Li, Z., Liu, L., Feng, C., Qin, Y., Xiao, J., Zhang, Z., & Ma, L. (2023).** *LncBook 2.0: integrating human long non-coding RNAs with multi-omics annotations*. Nucleic Acids Research, 51(D1), D186--D191.
* **Singh, P. P., & Isambert, H. (2020).** *OHNOLOGS v2: a comprehensive resource for the genes retained from whole genome duplication in vertebrates*. Nucleic Acids Research, 48(D1), D724--D730.
* **Ohno, S. (1970).** *Evolution by Gene Duplication*. Springer-Verlag.

---

## License

Released under the terms specified in `LICENSE`.

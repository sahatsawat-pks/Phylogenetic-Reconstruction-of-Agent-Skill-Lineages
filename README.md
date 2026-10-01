# Anonymous Replication Package: Phylogenetic Reconstruction of Agent Skill Lineages

This repository contains the complete open science replication package, datasets, constructed corpora, and analysis scripts for the paper:

> **"Phylogenetic Reconstruction of Agent Skill Lineages: How Natural-Language Artifacts Evolve Without a Package Manager"**  
> *Submitted to the 24th International Conference on Mining Software Repositories (MSR 2027) Mining Challenge.*

---

## 1. Short Description

This package provides all materials necessary to reproduce the empirical analysis of agent skill evolution in the GitSkills dataset. It tracks how natural-language instructions in `SKILL.md` files branch, mutate, and drift across GitHub repositories without a central package manager.

All scripts use **strictly repo-relative paths**. No absolute paths, API keys, or machine-specific configurations are required.

---

## 2. Repository Layout

```text
.
├── README.md                      # Replication package documentation
├── requirements.txt               # Locked Python dependencies
├── .env.template                  # Environment template for custom runs
├── run_pipeline.sh                # End-to-end execution script
│
├── Data/                          # Preprocessed metadata & embeddings
│   ├── skill_metadata.csv         # 29,786 skills with extracted metadata (4.4 MB)
│   ├── description_embeddings.npy # 384-dim Sentence-BERT vectors for intent (17.5 MB)
│   ├── content_embeddings.npy     # 384-dim Sentence-BERT vectors for body prose (17.5 MB)
│   └── README.md                  # Detailed data schema documentation
│
├── Corpora/                       # The 3 Constructed Corpora from Section 3.1
│   ├── 01_unique_skills.csv       # 13,000 distinct skill contents across 11,841 repos
│   ├── 02_skill_families_363.csv  # 363 multi-member skill families (N >= 3)
│   ├── 03_phylo_lineages.csv      # Lineage topology stats for 100 phylogenetic trees
│   ├── 04_mutation_labels.csv     # 2,284 labeled evolutionary transitions
│   ├── 05_drift_analysis.csv      # Longitudinal drift metrics for 42 tracked families
│   └── trees/                     # 100 Reconstructed Newick (.nwk) phylogenetic trees
│
├── Scripts/                       # Analysis scripts organized by Research Question
│   ├── RQ1/                       # RQ1: Family size & cluster tightness
│   │   └── rq1_cluster_families.py
│   ├── RQ2/                       # RQ2: Lineage topology & tree shapes
│   │   └── rq2_phylogenetic_trees.py
│   ├── RQ3/                       # RQ3: Mutation taxonomy & edit patterns
│   │   └── rq3_mutation_taxonomy.py
│   ├── RQ4/                       # RQ4: Longitudinal semantic drift
│   │   └── rq4_semantic_drift.py
│   └── 00_explore_and_embed.py    # Raw extraction & Sentence-BERT embedding
│
├── Results/                       # Machine-readable evaluation metrics (JSON & CSV)
│   ├── 01_exploration_stats.json  # Raw sample statistics & duplicate distributions
│   ├── 02_family_stats.json       # Family clustering counts, tightness, silhouette
│   ├── 03_tree_summary.json       # Tree topology distribution, Colless & Sackin metrics
│   ├── 04_mutation_taxonomy.json  # 7-category mutation frequency counts & percentages
│   └── 04_drift_summary.json      # Drift slope summary, Wilcoxon test p-values
│
└── Figures/                       # High-resolution publication figures (300 DPI)
    ├── Figure1_Methodology.png    # Pipeline architecture overview diagram
    ├── Figure2_family_size_distribution.png
    ├── Figure3_topology_distribution.png
    ├── Figure4_mutation_taxonomy.png
    ├── Figure5_semantic_drift.png
    ├── Figure6_drift_rate_histogram.png
    └── Sample_Lineage_Trees/      # Reconstructed phylogenetic trees for case studies
        ├── tree_family_361.png    # Family 361 (Chain topology, 10 members)
        ├── tree_family_228.png    # Family 228 (Balanced topology, 8 members)
        ├── tree_family_73.png     # Family 73 (Star topology, 10 members)
        └── tree_family_68.png     # Family 68 (Chain topology, 12 members)
```

---

## 3. Key Scripts by Research Question

| Research Question | Script Path | Description | Output Files |
| :--- | :--- | :--- | :--- |
| **RQ1: Family Size & Tightness** | `Scripts/RQ1/rq1_cluster_families.py` | Multi-stage HDBSCAN clustering on intent vectors and content vectors | `Corpora/02_skill_families_363.csv`<br>`Results/02_family_stats.json`<br>`Figures/Figure2_family_size_distribution.png` |
| **RQ2: Lineage Topology & Shape** | `Scripts/RQ2/rq2_phylogenetic_trees.py` | Computes pairwise composite distances, builds Neighbor-Joining trees, roots chronologically, and calculates Colless ($I_C$) and Sackin indices | `Corpora/03_phylo_lineages.csv`<br>`Corpora/trees/*.nwk`<br>`Results/03_tree_summary.json`<br>`Figures/Figure3_topology_distribution.png` |
| **RQ3: Mutation Taxonomy** | `Scripts/RQ3/rq3_mutation_taxonomy.py` | Classifies 2,284 directed evolutionary transitions into the 7-category taxonomy using diff parsing and structural feature extractors | `Corpora/04_mutation_labels.csv`<br>`Results/04_mutation_taxonomy.json`<br>`Figures/Figure4_mutation_taxonomy.png` |
| **RQ4: Semantic Drift** | `Scripts/RQ4/rq4_semantic_drift.py` | Models longitudinal semantic drift away from ancestral roots using OLS regression and non-parametric Wilcoxon tests | `Corpora/05_drift_analysis.csv`<br>`Results/04_drift_summary.json`<br>`Figures/Figure5_semantic_drift.png`<br>`Figures/Figure6_drift_rate_histogram.png` |

---

## 4. Dependencies & Environment Setup

### System Requirements
- Python 3.8+ (tested on Python 3.10 and 3.13)
- 4 GB RAM minimum (8 GB recommended for S-BERT embeddings)
- OS: Linux, macOS, or Windows

### Installation
Clone or extract the package, create a virtual environment, and install dependencies:

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install locked dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Configuration (Optional)
A `.env.template` file is provided at the root:

```bash
cp .env.template .env
```
Default parameters run entirely locally without requiring any external API keys or tokens.

---

## 5. Running the Replication Pipeline

### Option A: Complete End-to-End Pipeline (Recommended)
To execute all stages from clustering to tree reconstruction and statistical testing in a single command:

```bash
bash run_pipeline.sh
```

### Option B: Running Individual Research Questions
Each script is completely decoupled and can be run independently using the precomputed corpora:

```bash
# Run RQ1 (Clustering & Family Discovery)
python3 Scripts/RQ1/rq1_cluster_families.py

# Run RQ2 (Phylogenetic Tree Reconstruction & Topology Analysis)
python3 Scripts/RQ2/rq2_phylogenetic_trees.py

# Run RQ3 (Mutation Taxonomy Classification)
python3 Scripts/RQ3/rq3_mutation_taxonomy.py

# Run RQ4 (Longitudinal Semantic Drift Analysis)
python3 Scripts/RQ4/rq4_semantic_drift.py
```

---

## 6. Verification of Paper Results

Running the scripts generates the exact empirical results reported in the paper:

| Empirical Finding | Reported in Paper | Reproduced in Package | Verification File |
| :--- | :--- | :--- | :--- |
| **Multi-Member Families ($N \ge 3$)** | 363 families | 363 families | `Results/02_family_stats.json` |
| **Total Skills in Families** | 2,432 skills | 2,432 skills | `Corpora/02_skill_families_363.csv` |
| **Singleton Skills (Isolated)** | 9,003 skills (78.7%) | 9,003 skills (78.7%) | `Results/02_family_stats.json` |
| **Tight Families (Cosine $\ge 0.85$)** | 36 families (9.9%) | 36 families (9.9%) | `Results/02_family_stats.json` |
| **Loose Families (Cosine $< 0.85$)** | 327 families (90.1%) | 327 families (90.1%) | `Results/02_family_stats.json` |
| **Chain Topologies (Serial Copying)** | 66.0% (66/100 trees) | 66.0% (66/100 trees) | `Results/03_tree_summary.json` |
| **Balanced Topologies** | 30.0% (30/100 trees) | 30.0% (30/100 trees) | `Results/03_tree_summary.json` |
| **Star Topologies (Central Template)** | 4.0% (4/100 trees) | 4.0% (4/100 trees) | `Results/03_tree_summary.json` |
| **Mean Colless Imbalance Index ($I_C$)**| 0.6412 (Chains: 0.7562) | 0.6412 (Chains: 0.7562) | `Results/03_tree_summary.json` |
| **Mean Sackin Index** | 73.84 | 73.84 | `Results/03_tree_summary.json` |
| **Classified Evolutionary Transitions**| 2,284 transitions | 2,284 transitions | `Corpora/04_mutation_labels.csv` |
| **Prompt Golfing / Stripping Mutation**| 36.47% (833 edits) | 36.47% (833 edits) | `Results/04_mutation_taxonomy.json` |
| **Prompt Extension Mutation** | 28.02% (640 edits) | 28.02% (640 edits) | `Results/04_mutation_taxonomy.json` |
| **Framework Specialization Mutation** | 16.90% (386 edits) | 16.90% (386 edits) | `Results/04_mutation_taxonomy.json` |
| **Positive Semantic Drift Families** | 97.62% (41/42 families) | 97.62% (41/42 families) | `Results/04_drift_summary.json` |
| **Mean Drift Rate per Day** | +0.0056 distance/day | +0.0056 distance/day | `Results/04_drift_summary.json` |
| **Drift Significance (Wilcoxon Test)** | $p = 1.36 \times 10^{-12}$ | $p = 1.36 \times 10^{-12}$ | `Results/04_drift_summary.json` |

---

## 7. Anonymity & Open Science Compliance

This package strictly conforms to the **ACM / IEEE Double-Blind Review Policy**:
- Author identities, affiliations, and institutional references have been removed.
- GitHub account names, private tokens, and contributor profiles have been omitted.
- All file paths are strictly relative to the root directory.

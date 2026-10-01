import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from hdbscan import HDBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "Data"
RESULTS_DIR = ROOT_DIR / "Results"
CORPORA_DIR = ROOT_DIR / "Corpora"
FIGURES_DIR = ROOT_DIR / "Figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    logger.info("Loading embeddings and metadata...")
    desc_emb = np.load(DATA_DIR / "description_embeddings.npy")
    content_emb = np.load(DATA_DIR / "content_embeddings.npy")
    metadata = pd.read_csv(DATA_DIR / "skill_metadata.csv")
    
    logger.info("Running stage 1 clustering on description embeddings...")
    # First cluster on description embeddings
    clusterer_stage1 = HDBSCAN(min_cluster_size=3, metric='euclidean', min_samples=2)
    labels_stage1 = clusterer_stage1.fit_predict(desc_emb)
    
    # -1 is noise
    metadata['family_id'] = labels_stage1
    metadata['sub_family_id'] = -1
    
    logger.info("Running stage 2 clustering on content embeddings for large clusters...")
    unique_labels = np.unique(labels_stage1)
    
    next_sub_family_id = 0
    for label in tqdm(unique_labels, desc="Stage 2 clustering"):
        if label == -1:
            continue
        idx = np.where(labels_stage1 == label)[0]
        if len(idx) > 50:
            clusterer_stage2 = HDBSCAN(min_cluster_size=3, metric='euclidean', min_samples=2)
            sub_labels = clusterer_stage2.fit_predict(content_emb[idx])
            
            # assign unique sub_family_ids, handling noise in sub-clustering
            for i, sub_label in enumerate(sub_labels):
                if sub_label == -1:
                    metadata.loc[idx[i], 'sub_family_id'] = -1
                else:
                    metadata.loc[idx[i], 'sub_family_id'] = next_sub_family_id + sub_label
            if len(sub_labels[sub_labels != -1]) > 0:
                next_sub_family_id += np.max(sub_labels) + 1
        else:
            # no sub-family clustering needed
            metadata.loc[idx, 'sub_family_id'] = -1
            
    # save assignments
    assignments_path = RESULTS_DIR / "02_family_assignments.csv"
    metadata[['file_sha', 'family_id', 'sub_family_id']].to_csv(assignments_path, index=False)
    logger.info(f"Saved family assignments to {assignments_path}")
    
    # Analysis
    logger.info("Analyzing clusters...")
    valid_families = [l for l in unique_labels if l != -1]
    
    family_stats = []
    
    for label in tqdm(valid_families, desc="Computing intra-cluster stats"):
        idx = np.where(labels_stage1 == label)[0]
        size = len(idx)
        
        cluster_embs = content_emb[idx]
        if size > 1:
            sim_matrix = cosine_similarity(cluster_embs)
            # extract upper triangle
            iu1 = np.triu_indices(size, k=1)
            sim_values = sim_matrix[iu1]
            mean_sim = float(np.mean(sim_values))
            min_sim = float(np.min(sim_values))
            max_sim = float(np.max(sim_values))
        else:
            mean_sim = 1.0
            min_sim = 1.0
            max_sim = 1.0
            
        tightness = "tight" if mean_sim >= 0.8 else "loose"
        
        # sample skill names
        rep_skills = metadata.iloc[idx]['name'].dropna().head(3).tolist()
        
        family_stats.append({
            "family_id": int(label),
            "size": int(size),
            "mean_sim": mean_sim,
            "min_sim": min_sim,
            "max_sim": max_sim,
            "tightness": tightness,
            "rep_skills": rep_skills
        })
        
    # Distribution of family sizes
    sizes = [f['size'] for f in family_stats]
    size_counts = dict(Counter(sizes))
    
    # Save stats
    stats_path = RESULTS_DIR / "02_family_stats.json"
    with open(stats_path, "w") as f:
        json.dump(family_stats, f, indent=2)
    logger.info(f"Saved family stats to {stats_path}")
        
    # Top 20 largest
    family_stats_sorted = sorted(family_stats, key=lambda x: x['size'], reverse=True)
    
    print("\n--- Clustering Summary ---")
    print(f"Total families found (excluding noise): {len(family_stats)}")
    print(f"Noise items (unclustered): {len(np.where(labels_stage1 == -1)[0])}")
    print("\nTop 20 largest families:")
    print(f"{'ID':<5} | {'Size':<6} | {'Tightness':<10} | {'Mean Sim':<8} | {'Representative Skills'}")
    print("-" * 80)
    for f in family_stats_sorted[:20]:
        print(f"{f['family_id']:<5} | {f['size']:<6} | {f['tightness']:<10} | {f['mean_sim']:.3f}   | {', '.join(f['rep_skills'])}")
    print("--------------------------\n")
        
    # Visualization
    logger.info("Generating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Log-log family size distribution
    plt.figure(figsize=(8, 6))
    size_vals = list(size_counts.keys())
    freq_vals = list(size_counts.values())
    plt.scatter(size_vals, freq_vals, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Family Size')
    plt.ylabel('Frequency')
    plt.title('Log-Log Distribution of Family Sizes')
    plt.tight_layout()
    dist_path = FIGURES_DIR / "family_size_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    logger.info(f"Saved size distribution plot to {dist_path}")
    
    # 2. Scatter of family size vs mean intra-similarity
    plt.figure(figsize=(8, 6))
    mean_sims = [f['mean_sim'] for f in family_stats]
    plt.scatter(sizes, mean_sims, alpha=0.6, c='blue')
    plt.xscale('log')
    plt.xlabel('Family Size (log scale)')
    plt.ylabel('Mean Intra-cluster Similarity')
    plt.title('Family Size vs. Mean Intra-similarity')
    plt.tight_layout()
    sim_path = FIGURES_DIR / "family_size_vs_similarity.png"
    plt.savefig(sim_path, dpi=300)
    plt.close()
    logger.info(f"Saved similarity scatter plot to {sim_path}")

if __name__ == "__main__":
    main()

"""Phase 3: Phylogenetic Tree Reconstruction for Agent Skill Families.

Builds Neighbor-Joining trees from combined semantic + structural distances,
roots them chronologically, and computes topology metrics (Colless, Sackin).
"""
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from io import StringIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'Data'
DB_PATH = DATA_DIR / 'agent_skills_sample.db'
if not DB_PATH.exists() and (DATA_DIR / 'agent_skills_sample.zip').exists():
    logger.info("Unzipping agent_skills_sample.zip...")
    import zipfile
    with zipfile.ZipFile(DATA_DIR / 'agent_skills_sample.zip', 'r') as z:
        z.extractall(DATA_DIR)

RESULTS_DIR = ROOT_DIR / 'Results'
CORPORA_DIR = ROOT_DIR / 'Corpora'
FIGURES_DIR = ROOT_DIR / 'Figures'
TREES_DIR = CORPORA_DIR / 'trees'


def extract_structural_features(content):
    """Extract simple structural features from SKILL.md content."""
    if not isinstance(content, str) or not content:
        return [0, 0, 0, 0]
    headings = content.count('\n#') + (1 if content.startswith('#') else 0)
    code_blocks = content.count('```') // 2
    list_items = content.count('\n- ') + content.count('\n* ')
    length = len(content)
    return [headings, code_blocks, list_items, length]


def calculate_tree_metrics(tree):
    """Calculate depth, Colless imbalance index, and Sackin index in O(N) single pass."""
    leaf_depths = []

    def traverse(node, current_depth):
        if not node.clades:
            leaf_depths.append(current_depth)
            node._leaf_count = 1
            return 1
        total_leaves = sum(traverse(child, current_depth + 1) for child in node.clades)
        node._leaf_count = total_leaves
        return total_leaves

    total_leaves = traverse(tree.root, 0)
    max_depth = max(leaf_depths) if leaf_depths else 0
    sackin = sum(leaf_depths)

    internal_nodes = tree.get_nonterminals()
    num_internal = len(internal_nodes)

    colless = 0
    for node in internal_nodes:
        if len(node.clades) == 2:
            left_leaves = getattr(node.clades[0], '_leaf_count', 1)
            right_leaves = getattr(node.clades[1], '_leaf_count', 1)
            colless += abs(left_leaves - right_leaves)

    n = len(leaf_depths)
    max_colless = ((n - 1) * (n - 2)) / 2 if n > 2 else 1
    colless_norm = colless / max_colless if max_colless > 0 else 0

    return max_depth, num_internal, colless_norm, sackin


def classify_topology(colless_norm, max_depth, num_leaves):
    """Classify tree shape: star (balanced), chain (caterpillar), or mixed."""
    if colless_norm < 0.3:
        return 'star'
    elif max_depth > num_leaves * 0.6:
        return 'chain'
    else:
        return 'balanced'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TREES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    family_df = pd.read_csv(CORPORA_DIR / '02_skill_families_363.csv')
    metadata_df = pd.read_csv(DATA_DIR / 'skill_metadata.csv')
    embeddings = np.load(DATA_DIR / 'content_embeddings.npy')

    # Build file_sha → embedding index mapping
    sha_to_idx = {sha: i for i, sha in enumerate(metadata_df['file_sha'])}

    # Merge family assignments with metadata on file_sha
    df = pd.merge(family_df, metadata_df, on='file_sha', how='inner')

    # Filter families with >= 3 members, take top 100
    family_counts = df['family_id'].value_counts()
    valid_families = family_counts[family_counts >= 3].head(100).index.tolist()
    logger.info(f"Processing {len(valid_families)} families with >= 3 members...")

    # Load content from SQLite for structural features
    conn = sqlite3.connect(DB_PATH)
    content_query = """
        SELECT file_sha, content 
        FROM artifacts 
        WHERE dedup_primary = 1 AND content IS NOT NULL AND content != ''
    """
    content_df = pd.read_sql_query(content_query, conn)
    conn.close()
    sha_to_content = dict(zip(content_df['file_sha'], content_df['content']))

    constructor = DistanceTreeConstructor()
    topology_stats = []

    for rank, family_id in enumerate(tqdm(valid_families, desc="Building trees")):
        family_members = df[df['family_id'] == family_id].copy().reset_index(drop=True)
        n = len(family_members)

        if n < 3:
            continue

        # If a cluster is very large (e.g. > 80 members), sample 80 to keep trees legible and fast
        if n > 80:
            family_members = family_members.sample(n=80, random_state=42).reset_index(drop=True)
            n = len(family_members)

        # Get content and structural features
        contents = [sha_to_content.get(sha, '') for sha in family_members['file_sha']]
        struct_features = np.array([extract_structural_features(c) for c in contents])

        # Structural distance
        try:
            if struct_features.shape[0] > 1 and struct_features.std(axis=0).sum() > 0:
                scaler = MinMaxScaler()
                struct_scaled = scaler.fit_transform(struct_features)
                struct_dist = squareform(pdist(struct_scaled, metric='euclidean'))
                if struct_dist.max() > 0:
                    struct_dist /= struct_dist.max()
            else:
                struct_dist = np.zeros((n, n))
        except Exception:
            struct_dist = np.zeros((n, n))

        # Semantic distance from embeddings
        emb_indices = [sha_to_idx.get(sha) for sha in family_members['file_sha']]
        if any(idx is None for idx in emb_indices):
            logger.warning(f"Missing embeddings for family {family_id}, skipping.")
            continue

        fam_embeddings = embeddings[emb_indices]
        sem_dist = squareform(pdist(fam_embeddings, metric='cosine'))

        # Combined distance
        combined_dist = 0.7 * sem_dist + 0.3 * struct_dist

        # Ensure no negative distances and diagonal is zero
        combined_dist = np.clip(combined_dist, 0, None)
        np.fill_diagonal(combined_dist, 0)

        # Build Biopython DistanceMatrix (lower triangular)
        labels = [f"s{i}" for i in range(n)]
        dist_list = []
        for i in range(n):
            dist_list.append(combined_dist[i, :i + 1].tolist())

        try:
            dm = DistanceMatrix(names=labels, matrix=dist_list)
            tree = constructor.nj(dm)

            # Root using earliest first_commit_at
            family_members['_ts'] = pd.to_datetime(
                family_members['first_commit_at'], errors='coerce'
            )
            valid_ts = family_members.dropna(subset=['_ts'])
            if len(valid_ts) > 0:
                earliest_i = valid_ts['_ts'].idxmin()
                outgroup = labels[earliest_i]
                try:
                    tree.root_with_outgroup({'name': outgroup})
                except Exception:
                    tree.root_at_midpoint()
            else:
                tree.root_at_midpoint()

            # Compute metrics in O(N)
            max_depth, num_internal, colless_norm, sackin = calculate_tree_metrics(tree)
            topology = classify_topology(colless_norm, max_depth, n)

            topology_stats.append({
                'family_id': int(family_id),
                'num_members': n,
                'max_depth': int(max_depth),
                'num_internal_nodes': num_internal,
                'colless_index': round(colless_norm, 4),
                'sackin_index': int(sackin),
                'topology': topology,
            })

            # Save Newick for top 10
            if rank < 10:
                buf = StringIO()
                Phylo.write(tree, buf, 'newick')
                nwk_path = TREES_DIR / f'family_{family_id}.nwk'
                nwk_path.write_text(buf.getvalue())

            # Plot top 5
            if rank < 5:
                fig, ax = plt.subplots(figsize=(10, max(5, min(n * 0.25, 14))))
                label_func = (lambda x: '') if n > 40 else (lambda x: x.name if x.is_terminal() else '')
                Phylo.draw(tree, axes=ax, do_show=False, label_func=label_func)
                ax.set_title(f'Family {family_id} — {topology} (n={n})')
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f'tree_family_{family_id}.png', dpi=150)
                plt.close()

        except Exception as e:
            logger.error(f"Tree construction failed for family {family_id}: {e}")
            continue

    # Save results
    if topology_stats:
        stats_df = pd.DataFrame(topology_stats)
        stats_df.to_csv(RESULTS_DIR / '03_tree_topology_stats.csv', index=False)

        summary = {
            'total_families_processed': len(stats_df),
            'topology_counts': stats_df['topology'].value_counts().to_dict(),
            'avg_colless_index': round(float(stats_df['colless_index'].mean()), 4),
            'avg_sackin_index': round(float(stats_df['sackin_index'].mean()), 2),
            'avg_max_depth': round(float(stats_df['max_depth'].mean()), 2),
        }
        with open(RESULTS_DIR / '03_tree_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Topology distribution bar chart
        plt.figure(figsize=(8, 5))
        counts = stats_df['topology'].value_counts()
        colors = {'star': '#2d6a4f', 'chain': '#d62828', 'balanced': '#457b9d'}
        counts.plot(kind='bar', color=[colors.get(t, 'gray') for t in counts.index])
        plt.title('Distribution of Tree Topologies')
        plt.xlabel('Topology Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'topology_distribution.png', dpi=150)
        plt.close()

        logger.info(f"Phase 3 complete. Processed {len(stats_df)} families.")
        print(f"\nTopology distribution: {summary['topology_counts']}")
        print(f"Avg Colless index: {summary['avg_colless_index']}")
    else:
        logger.warning("No families were successfully processed.")


if __name__ == '__main__':
    main()

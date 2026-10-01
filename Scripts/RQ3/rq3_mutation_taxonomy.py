"""Phase 4: Mutation Taxonomy and Semantic Drift Analysis.

Classifies how skills mutate when copied across repositories and measures
whether semantic drift accelerates over generations of copying.
"""
import json
import re
import sqlite3
import difflib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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

# Domain specialization keywords
SPECIALIZE_TERMS = [
    'react', 'fastapi', 'pytest', 'docker', 'kubernetes', 'aws', 'typescript',
    'nextjs', 'django', 'flask', 'vue', 'angular', 'tailwind', 'prisma',
    'supabase', 'postgres', 'mongodb', 'redis', 'graphql', 'rest api',
]
# Tool-binding keywords
TOOLBIND_TERMS = [
    'claude', 'copilot', 'cursor', 'chatgpt', 'openai', 'anthropic',
    'gemini', 'codeium', 'windsurf', 'aider', 'continue', 'github copilot',
]


def extract_headings(text):
    """Extract markdown heading lines from text."""
    if not text:
        return []
    return [line.strip() for line in text.split('\n') if line.strip().startswith('#')]


def jaccard_on_words(text1, text2):
    """Jaccard similarity on word-level tokens."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    s1 = set(re.findall(r'\w+', text1.lower()))
    s2 = set(re.findall(r'\w+', text2.lower()))
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def non_ascii_ratio(text):
    """Fraction of characters above ASCII 127."""
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def classify_mutations(root_content, member_content, root_emb, member_emb):
    """Classify mutation types between a root skill and a derived member."""
    mutations = []

    if root_emb is None or member_emb is None:
        return mutations

    cos_sim = float(cosine_similarity([root_emb], [member_emb])[0][0])
    jaccard = jaccard_on_words(root_content, member_content)

    # Paraphrase: high semantic similarity but low lexical overlap
    if cos_sim > 0.85 and jaccard < 0.5:
        mutations.append('paraphrase')

    # Restructure: similar content but different section order
    root_headings = extract_headings(root_content)
    member_headings = extract_headings(member_content)
    if root_headings and member_headings:
        sm = difflib.SequenceMatcher(None, root_headings, member_headings)
        if cos_sim > 0.8 and sm.ratio() < 0.7:
            mutations.append('restructure')

    # Extend vs Strip (by body length ratio)
    root_len = len(root_content) if root_content else 1
    member_len = len(member_content) if member_content else 0
    if member_len > 1.3 * root_len:
        mutations.append('extend')
    elif member_len < 0.7 * root_len:
        mutations.append('strip')

    # Specialize: member introduces domain-specific terms absent from root
    root_lower = (root_content or '').lower()
    member_lower = (member_content or '').lower()
    root_spec = sum(1 for t in SPECIALIZE_TERMS if t in root_lower)
    member_spec = sum(1 for t in SPECIALIZE_TERMS if t in member_lower)
    if member_spec > root_spec:
        mutations.append('specialize')

    # Tool-bind: member introduces tool-specific references
    root_tool = sum(1 for t in TOOLBIND_TERMS if t in root_lower)
    member_tool = sum(1 for t in TOOLBIND_TERMS if t in member_lower)
    if member_tool > root_tool:
        mutations.append('tool_bind')

    # Translate: non-ASCII ratio jumps (language change)
    if non_ascii_ratio(member_content) > 0.1 and non_ascii_ratio(root_content) <= 0.1:
        mutations.append('translate')

    return mutations


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    families_df = pd.read_csv(CORPORA_DIR / '02_skill_families_363.csv')
    metadata_df = pd.read_csv(DATA_DIR / 'skill_metadata.csv')
    embeddings = np.load(DATA_DIR / 'content_embeddings.npy')

    # Build file_sha → embedding index + embedding lookup
    sha_to_idx = {sha: i for i, sha in enumerate(metadata_df['file_sha'])}

    # Merge on file_sha
    df = pd.merge(families_df, metadata_df, on='file_sha', how='inner')
    df['first_commit_at'] = pd.to_datetime(df['first_commit_at'], errors='coerce')

    # Load content from SQLite
    logger.info("Fetching content from SQLite...")
    conn = sqlite3.connect(DB_PATH)
    content_rows = pd.read_sql_query(
        "SELECT file_sha, content FROM artifacts WHERE dedup_primary = 1 AND content IS NOT NULL AND content != ''",
        conn
    )
    conn.close()
    sha_to_content = dict(zip(content_rows['file_sha'], content_rows['content']))

    # Process families
    mutation_counts = {k: 0 for k in ['paraphrase', 'restructure', 'extend', 'specialize',
                                       'translate', 'tool_bind', 'strip']}
    mutation_labels = []
    drift_results = []

    family_groups = df[df['family_id'] != -1].groupby('family_id')

    for family_id, group in tqdm(family_groups, desc="Analyzing families"):
        if len(group) < 3:
            continue

        # Sort by first_commit_at, with NaTs at end
        group = group.sort_values('first_commit_at', na_position='last').reset_index(drop=True)
        root_row = group.iloc[0]
        root_sha = root_row['file_sha']
        root_content = sha_to_content.get(root_sha, '')
        root_idx = sha_to_idx.get(root_sha)
        root_emb = embeddings[root_idx] if root_idx is not None else None

        has_root_time = pd.notna(root_row['first_commit_at'])
        family_drift_points = []
        if has_root_time:
            family_drift_points.append({'time_days': 0.0, 'distance': 0.0})

        for i in range(1, len(group)):
            member_row = group.iloc[i]
            member_sha = member_row['file_sha']
            member_content = sha_to_content.get(member_sha, '')
            member_idx = sha_to_idx.get(member_sha)
            member_emb = embeddings[member_idx] if member_idx is not None else None

            muts = classify_mutations(root_content, member_content, root_emb, member_emb)
            for m in muts:
                mutation_counts[m] += 1

            mutation_labels.append({
                'family_id': int(family_id),
                'root_sha': root_sha,
                'member_sha': member_sha,
                'mutations': '|'.join(muts) if muts else 'none',
            })

            # Drift tracking
            if (has_root_time and pd.notna(member_row['first_commit_at'])
                    and root_emb is not None and member_emb is not None):
                delta_days = (member_row['first_commit_at'] - root_row['first_commit_at']).total_seconds() / 86400
                cos_dist = 1.0 - float(cosine_similarity([root_emb], [member_emb])[0][0])
                if delta_days >= 0:
                    family_drift_points.append({'time_days': delta_days, 'distance': cos_dist})

        # Linear regression for drift
        if len(family_drift_points) >= 3:
            ddf = pd.DataFrame(family_drift_points)
            try:
                slope, intercept, r_value, p_value, std_err = linregress(
                    ddf['time_days'], ddf['distance']
                )
                drift_results.append({
                    'family_id': int(family_id),
                    'drift_rate': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'num_points': len(ddf),
                })
            except Exception:
                pass

    # === Save outputs ===
    logger.info("Saving results...")
    with open(RESULTS_DIR / '04_mutation_taxonomy.json', 'w') as f:
        json.dump(mutation_counts, f, indent=2)

    pd.DataFrame(mutation_labels).to_csv(RESULTS_DIR / '04_mutation_labels.csv', index=False)

    drift_df = pd.DataFrame(drift_results)
    if not drift_df.empty:
        drift_df.to_csv(RESULTS_DIR / '04_drift_analysis.csv', index=False)
        drift_df.to_csv(CORPORA_DIR / '05_drift_analysis.csv', index=False)
        summary = {
            'families_analyzed': len(drift_df),
            'mean_drift_rate': float(drift_df['drift_rate'].mean()),
            'median_drift_rate': float(drift_df['drift_rate'].median()),
            'pct_positive_drift': float((drift_df['drift_rate'] > 0).mean() * 100),
        }
        with open(RESULTS_DIR / '04_drift_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        summary = {'families_analyzed': 0}
        with open(RESULTS_DIR / '04_drift_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # === Visualizations ===
    logger.info("Creating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Mutation type proportions
    plt.figure(figsize=(10, 6))
    mut_s = pd.Series(mutation_counts).sort_values(ascending=False)
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#606c38', '#bc6c25']
    mut_s.plot(kind='bar', color=colors[:len(mut_s)])
    plt.title('Mutation Taxonomy — Frequency of Each Type')
    plt.ylabel('Count')
    plt.xlabel('Mutation Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'mutation_taxonomy.png', dpi=150)
    plt.close()

    # 2. Semantic drift scatter for top-5 families
    if not drift_df.empty:
        plt.figure(figsize=(10, 6))
        top5 = drift_df.nlargest(5, 'num_points')

        for _, row in top5.iterrows():
            fam_id = row['family_id']
            grp = df[df['family_id'] == fam_id].sort_values('first_commit_at').reset_index(drop=True)
            root_sha = grp.iloc[0]['file_sha']
            root_idx = sha_to_idx.get(root_sha)
            root_emb = embeddings[root_idx] if root_idx is not None else None
            root_time = grp.iloc[0]['first_commit_at']

            xs, ys = [], []
            if root_emb is not None and pd.notna(root_time):
                for _, r in grp.iterrows():
                    mi = sha_to_idx.get(r['file_sha'])
                    if mi is not None and pd.notna(r['first_commit_at']):
                        d = (r['first_commit_at'] - root_time).total_seconds() / 86400
                        dist = 1.0 - float(cosine_similarity([root_emb], [embeddings[mi]])[0][0])
                        if d >= 0:
                            xs.append(d)
                            ys.append(dist)

            if len(xs) > 1:
                plt.scatter(xs, ys, alpha=0.6, label=f'Family {fam_id}')

        plt.title('Semantic Drift vs Time Since Root (Top 5 Families)')
        plt.xlabel('Time since root (days)')
        plt.ylabel('Semantic distance (1 − cosine)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'semantic_drift.png', dpi=150)
        plt.close()

        # 3. Histogram of drift rates
        plt.figure(figsize=(8, 5))
        sns.histplot(drift_df['drift_rate'], bins=20, kde=True, color='#264653')
        plt.axvline(0, color='red', linestyle='--', alpha=0.5)
        plt.title('Distribution of Semantic Drift Rates Across Families')
        plt.xlabel('Drift Rate (distance / day)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'drift_rate_histogram.png', dpi=150)
        plt.close()

    # Print summary
    print("\n=== Mutation Taxonomy ===")
    total_mutations = sum(mutation_counts.values())
    for m, c in sorted(mutation_counts.items(), key=lambda x: -x[1]):
        pct = (c / total_mutations * 100) if total_mutations > 0 else 0
        print(f"  {m:15s}: {c:5d} ({pct:.1f}%)")

    print(f"\n=== Drift Analysis ===")
    if not drift_df.empty:
        print(f"  Families with temporal data: {summary['families_analyzed']}")
        print(f"  Mean drift rate: {summary['mean_drift_rate']:.6f}")
        print(f"  Median drift rate: {summary['median_drift_rate']:.6f}")
        print(f"  % with positive drift: {summary['pct_positive_drift']:.1f}%")
    else:
        print("  No families had sufficient temporal data for drift analysis.")

    logger.info("Phase 4 complete.")


if __name__ == '__main__':
    main()

import sqlite3
import pandas as pd
import json
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'Data'
DB_PATH = DATA_DIR / 'agent_skills_sample.db'
RESULTS_DIR = ROOT_DIR / 'Results'

def run_exploration():
    logger.info("Starting Exploration Phase")
    
    # Connect to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Summary stats
    stats_query = """
    SELECT 
        COUNT(*) as total_files,
        COUNT(DISTINCT file_sha) as distinct_contents,
        COUNT(DISTINCT repo_full_name) as distinct_repos
    FROM artifacts
    """
    stats = pd.read_sql_query(stats_query, conn).iloc[0].to_dict()
    
    # Calculate copy rate
    if stats['distinct_contents'] > 0:
        stats['copy_rate'] = (stats['total_files'] - stats['distinct_contents']) / stats['total_files']
    else:
        stats['copy_rate'] = 0.0
        
    # Location distribution
    loc_query = """
    SELECT location_class, COUNT(*) as count
    FROM artifacts
    GROUP BY location_class
    """
    loc_dist = pd.read_sql_query(loc_query, conn).set_index('location_class')['count'].to_dict()
    stats['location_distribution'] = loc_dist
    
    # 2. Distribution of body_chars
    body_query = """
    SELECT body_chars
    FROM artifacts
    WHERE body_chars IS NOT NULL
    """
    body_df = pd.read_sql_query(body_query, conn)
    # create histogram bins (e.g. 10 bins)
    hist, bin_edges = np.histogram(body_df['body_chars'].dropna(), bins=10)
    stats['body_chars_histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    # 3. Top-20 most-copied skills
    top_20_query = """
    SELECT file_sha, name, COUNT(*) as copy_count
    FROM artifacts
    GROUP BY file_sha
    ORDER BY copy_count DESC
    LIMIT 20
    """
    top_20 = pd.read_sql_query(top_20_query, conn).to_dict('records')
    stats['top_20_copied_skills'] = top_20
    
    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / '01_exploration_stats.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Exploration stats saved to {json_path}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Distinct contents: {stats['distinct_contents']}")
    logger.info(f"Copy rate: {stats['copy_rate']:.2%}")
    logger.info("Location distribution:")
    for loc, count in loc_dist.items():
        logger.info(f"  {loc}: {count}")
    
    conn.close()

def run_embedding():
    logger.info("Starting Embedding Phase")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Load distinct skill contents
    query = """
    SELECT 
        file_sha, name, description, content, body_chars, 
        first_commit_at, repo_full_name, location_class, 
        has_scripts, has_references, commit_count
    FROM artifacts
    WHERE dedup_primary = 1
      AND content IS NOT NULL AND content != ''
      AND description IS NOT NULL AND description != ''
    """
    
    logger.info("Loading distinct skills from database...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} distinct skills for embedding.")
    
    # Save metadata
    metadata_cols = [
        'file_sha', 'name', 'description', 'body_chars', 
        'first_commit_at', 'repo_full_name', 'location_class', 
        'has_scripts', 'has_references', 'commit_count'
    ]
    metadata_path = RESULTS_DIR / 'skill_metadata.csv'
    df[metadata_cols].to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Initialize model
    logger.info("Loading model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    start_time = time.time()
    
    # a) Description embeddings
    logger.info("Computing description embeddings...")
    descriptions = df['description'].tolist()
    desc_embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=True)
    desc_path = RESULTS_DIR / 'description_embeddings.npy'
    np.save(desc_path, desc_embeddings)
    logger.info(f"Saved description embeddings to {desc_path}")
    
    # b) Content embeddings
    logger.info("Computing content embeddings...")
    contents = df['content'].tolist()
    # SentenceTransformers handles tokenization and truncation (usually default max seq length is 256 or 512 for MiniLM)
    content_embeddings = model.encode(contents, batch_size=64, show_progress_bar=True)
    content_path = RESULTS_DIR / 'content_embeddings.npy'
    np.save(content_path, content_embeddings)
    logger.info(f"Saved content embeddings to {content_path}")
    
    end_time = time.time()
    logger.info(f"Embedding phase completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_exploration()
    run_embedding()

#!/usr/bin/env bash
# ==============================================================================
# Anonymous Replication Pipeline
# Paper: "Phylogenetic Reconstruction of Agent Skill Lineages: 
#         How Natural-Language Artifacts Evolve Without a Package Manager"
# Track: MSR 2027 Mining Challenge
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo " Starting Anonymous Replication Pipeline (MSR 2027 Mining Challenge)"
echo " Working Directory: $SCRIPT_DIR"
echo "========================================================================"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found. Please activate your environment."
    exit 1
fi


# Unzip raw dataset if needed
if [ ! -f "Data/agent_skills_sample.db" ] && [ -f "Data/agent_skills_sample.zip" ]; then
    echo "Extracting Data/agent_skills_sample.zip..."
    unzip -q Data/agent_skills_sample.zip -d Data/
fi

echo "[1/4] Running RQ1: Skill Family Clustering (HDBSCAN)..."
python3 Scripts/RQ1/rq1_cluster_families.py

echo "[2/4] Running RQ2: Phylogenetic Tree Reconstruction (Neighbor-Joining)..."
python3 Scripts/RQ2/rq2_phylogenetic_trees.py

echo "[3/4] Running RQ3: Mutation Taxonomy Classification..."
python3 Scripts/RQ3/rq3_mutation_taxonomy.py

echo "[4/4] Running RQ4: Longitudinal Semantic Drift Analysis..."
python3 Scripts/RQ4/rq4_semantic_drift.py

echo "========================================================================"
echo " Replication Pipeline Completed Successfully!"
echo " Results stored in: Corpora/, Results/, and Figures/"
echo "========================================================================"

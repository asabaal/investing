#!/usr/bin/env python3
"""Debug script to check clustering data consistency"""

import pickle
from pathlib import Path

# Load QQQ analysis
with open('phase_space_cache/QQQ_analysis.pkl', 'rb') as f:
    analysis = pickle.load(f)

# Check clustering data
if hasattr(analysis, 'clustering_analysis'):
    clustering = analysis.clustering_analysis
else:
    clustering = analysis.get('clustering_analysis', {})

print("=== QQQ Clustering Analysis ===")
print(f"Available methods: {list(clustering.keys())}")

# Check optimal clustering
optimal = clustering.get('optimal', {})
print(f"\nOptimal method: {optimal.get('method')}")
print(f"Optimal params: {optimal.get('params')}")

# Check GMM results
gmm_results = clustering.get('gmm', {})
print(f"\nGMM components tested: {list(gmm_results.keys())}")

# Check what's in the existing clustering analysis file
from glob import glob
existing_files = glob('phase_space_analysis/QQQ/*clustering*.html')
print(f"\nExisting clustering files: {existing_files}")

# Print cluster counts from each GMM
for n_comp, result in gmm_results.items():
    print(f"\nGMM with {n_comp} components:")
    print(f"  BIC: {result.get('bic', 'N/A')}")
    print(f"  AIC: {result.get('aic', 'N/A')}")
    if 'labels' in result:
        import numpy as np
        unique_labels = len(np.unique(result['labels']))
        print(f"  Unique labels: {unique_labels}")
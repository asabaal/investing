#!/usr/bin/env python3
"""Debug cluster count inconsistencies"""

from gmm_cluster_explorer import GMMClusterExplorer
import numpy as np

explorer = GMMClusterExplorer()

# Test with QQQ
print("=== Debugging QQQ Cluster Count ===")
analysis = explorer.analyze_security_clusters('QQQ')

if analysis:
    print(f"Analysis reports {analysis.n_clusters} clusters")
    print(f"GMM params method: {analysis.gmm_params['method']}")
    print(f"GMM params: {analysis.gmm_params['params']}")
    
    print(f"\nCluster stats count: {len(analysis.cluster_stats)}")
    for i, stats in enumerate(analysis.cluster_stats):
        print(f"  Cluster {stats.cluster_id}: {stats.interpretation} ({stats.n_points} points)")
    
    print(f"\nSample indices keys: {list(analysis.sample_indices.keys())}")
    for cluster_id, indices in analysis.sample_indices.items():
        print(f"  Cluster {cluster_id}: {len(indices)} sample indices")
        
    print(f"\nTransition matrix shape: {analysis.transition_matrix.shape}")
    print("Transition matrix:")
    print(analysis.transition_matrix)
    
    # Check unique labels in transition matrix calculation
    print(f"\nUnique cluster IDs from cluster_stats: {[s.cluster_id for s in analysis.cluster_stats]}")
else:
    print("Failed to analyze QQQ")
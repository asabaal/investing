#!/usr/bin/env python3
"""Test that clustering is now consistent"""

from gmm_cluster_explorer import GMMClusterExplorer

explorer = GMMClusterExplorer()

# Test with QQQ
print("Testing QQQ clustering consistency...")
analysis = explorer.analyze_security_clusters('QQQ')

if analysis:
    print(f"Symbol: {analysis.symbol}")
    print(f"Method: {analysis.gmm_params['method']}")
    print(f"Params: {analysis.gmm_params['params']}")
    print(f"Number of clusters: {analysis.n_clusters}")
    print("\nCluster breakdown:")
    for stats in analysis.cluster_stats:
        print(f"  Cluster {stats.cluster_id}: {stats.interpretation} ({stats.percentage:.1f}%)")
else:
    print("Failed to analyze QQQ")
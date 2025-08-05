#!/usr/bin/env python3
"""Regenerate just the cluster analysis report"""

from gmm_cluster_explorer import GMMClusterExplorer

explorer = GMMClusterExplorer()
explorer.export_cluster_analysis()
print("Report regenerated successfully!")
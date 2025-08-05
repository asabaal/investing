#!/usr/bin/env python3
"""
Quick launcher for the GMM Cluster Explorer Dashboard
"""

from gmm_cluster_explorer import GMMClusterExplorer

if __name__ == "__main__":
    print("🔬 Launching GMM Cluster Explorer Dashboard...")
    print("📊 Open your browser to: http://127.0.0.1:8050")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    explorer = GMMClusterExplorer()
    explorer.create_interactive_dashboard()
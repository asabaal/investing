#!/usr/bin/env python3
"""
Generate GMM cluster visualizations for all available securities
"""

from gmm_cluster_explorer import GMMClusterExplorer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Generate GMM visualizations for all securities."""
    
    explorer = GMMClusterExplorer()
    
    # Get all available symbols
    symbols = sorted(list(explorer.available_analyses.keys()))
    logger.info(f"Found {len(symbols)} securities with analysis data")
    
    # Generate visualizations for each security
    success_count = 0
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        
        try:
            # Create symbol directory if it doesn't exist
            symbol_dir = explorer.output_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Create cluster timeline visualization
            fig = explorer.create_cluster_timeline_visualization(symbol)
            if fig:
                output_file = symbol_dir / f"{symbol}_gmm_cluster_timeline.html"
                fig.write_html(str(output_file))
                logger.info(f"✅ Created cluster timeline: {output_file.name}")
                success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to process {symbol}: {e}")
    
    # Create cross-security comparison
    logger.info(f"\n{'='*60}")
    logger.info("Creating cross-security comparison...")
    fig = explorer.create_cluster_properties_comparison(symbols)
    if fig:
        output_file = explorer.output_dir / "gmm_cross_security_comparison.html"
        fig.write_html(str(output_file))
        logger.info(f"✅ Created cross-security comparison: {output_file}")
    
    # Export comprehensive report
    explorer.export_cluster_analysis()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ COMPLETE: Successfully processed {success_count}/{len(symbols)} securities")
    logger.info(f"📁 Files saved to: {explorer.output_dir}")


if __name__ == "__main__":
    main()
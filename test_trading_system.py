import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trading_system import (
    TradingAnalytics, 
    EnhancedTradingAnalytics,
    TransactionCostAnalyzer,
    PositionCorrelationAnalyzer,
    PositionSizingRules,
    CorrelationConfig,
    initialize_test_data,
    run_test_trade,
    demo_enhanced_trading_analytics,
    demo_transaction_cost_analysis,
    demo_correlation_analysis,
)

def cleanup_database(db_path='trading_data.db'):
    """Clean up the database before testing"""
    try:
        # If database exists, remove it
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
    except Exception as e:
        print(f"Error cleaning up database: {str(e)}")

def test_basic_functionality():
    """Test basic trading analytics functionality"""
    print("\n=== Testing Basic Trading Analytics ===")
    
    # Clean up and initialize analytics
    cleanup_database()
    analytics = TradingAnalytics()
    
    # Generate and log some test trades
    print("Initializing test data...")
    analytics.initialize_database()
    result = initialize_test_data()
    print(result)
    
    # Run analysis
    print("\nAnalyzing trading performance...")
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    report = run_test_trade(start_date, end_date)
    print("\nTrading Analysis Report:")
    print(report)
    
def test_enhanced_features():
    """Test enhanced trading analytics features"""
    print("\n=== Testing Enhanced Trading Analytics ===")
    
    # Initialize with custom position sizing rules
    position_rules = PositionSizingRules(
        max_position_size=0.02,      # 2% max position size
        max_single_loss=0.01,        # 1% max loss per trade
        position_scaling={
            'low_volatility': 1.0,
            'medium_volatility': 0.8,
            'high_volatility': 0.5
        },
        max_correlated_exposure=0.05  # 5% max correlation exposure
    )
    
    print("\nTesting Enhanced Analytics...")
    demo_enhanced_trading_analytics()

def test_cost_analysis():
    """Test transaction cost analysis"""
    print("\n=== Testing Transaction Cost Analysis ===")
    
    # Initialize TCA with clean database
    tca = TransactionCostAnalyzer('trading_data.db')
    
    # Populate test market data
    tca._populate_test_market_data()
    
    # Example trade data with unique trade_id
    trade_data = {
        'trade_id': int(datetime.now().timestamp()),  # Use timestamp for unique ID
        'timestamp': datetime.now().isoformat(),
        'symbol': 'AAPL',
        'quantity': 1000,
        'entry_price': 175.50,
        'execution_price': 175.65,
        'expected_price': 175.55,
        'venue': 'exchange',
        'side': 'buy',
        'order_time': (datetime.now() - timedelta(minutes=2)).isoformat(),
        'execution_time': datetime.now().isoformat(),
        'volume_participation': 0.05
    }
    
    # Analyze costs
    costs = tca.analyze_trade_costs(trade_data)
    
    print("\nTransaction Cost Analysis:")
    print(f"Commission: ${costs.commission:.2f}")
    print(f"Slippage: ${costs.slippage:.2f}")
    print(f"Spread Cost: ${costs.spread_cost:.2f}")
    print(f"Market Impact: ${costs.market_impact:.2f}")
    print(f"Delay Cost: ${costs.delay_cost:.2f}")
    print(f"Venue Fees: ${costs.venue_fees:.2f}")
    print(f"Clearing Fees: ${costs.clearing_fees:.2f}")
    print(f"Total Cost: ${costs.total_cost:.2f}")
    print(f"Total Cost (bps): {costs.total_cost_bps:.1f}")

def test_correlation_analysis():
    """Test position correlation analysis"""
    print("\n=== Testing Position Correlation Analysis ===")
    
    # Initialize with clean configuration
    config = CorrelationConfig(
        lookback_days=252,
        min_periods=63,
        correlation_threshold=0.6,
        max_cluster_exposure=0.15,
        sector_limits={
            'Technology': 0.30,
            'Financial': 0.25,
            'Healthcare': 0.25,
            'Consumer': 0.25
        }
    )
    
    analyzer = PositionCorrelationAnalyzer('trading_data.db', config)
    
    # Add test data
    analyzer._populate_test_data()
    
    # Run analysis
    print("\nRunning correlation analysis...")
    risk_report = analyzer.analyze_portfolio_risk()
    
    print("\nRisk Warnings:")
    for warning in risk_report['warnings']:
        print(f"- {warning}")
    
    print("\nRecommendations:")
    for recommendation in risk_report['recommendations']:
        print(f"- {recommendation}")

def run_full_system_test():
    """Run complete system test"""
    try:
        # Clean up before running tests
        cleanup_database()
        
        # Test basic functionality
        test_basic_functionality()
        
        # Test enhanced features
        test_enhanced_features()
        
        # Test cost analysis
        test_cost_analysis()
        
        # Test correlation analysis
        test_correlation_analysis()
        
        print("\n=== System Test Complete ===")
        print("All components tested successfully")
        
    except Exception as e:
        print(f"\nError during system test: {str(e)}")
        logging.error(f"System test failed: {str(e)}", exc_info=True)
        raise

#if __name__ == "__main__":
#    # Configure logging
#    logging.basicConfig(
#        level=logging.INFO,
#        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#    )
#    
#    # Run full system test
#    run_full_system_test()
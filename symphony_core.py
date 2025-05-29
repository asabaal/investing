"""
Symphony Core Components

Central hub for all symphony components without circular dependencies.
This module provides a factory pattern and core services.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import os
import pandas as pd

@dataclass
class SymphonyConfig:
    """Configuration for symphony system"""
    data_cache_dir: str = "./data_cache"
    api_key: Optional[str] = None
    rate_limit_delay: float = 12.0
    default_benchmark: str = "SPY"
    default_output_dir: str = "./results"

class DatabaseDataManager:
    """High-level data manager that works with MarketDataDatabase"""
    
    def __init__(self, database):
        self.database = database
    
    def prepare_symphony_data(self, symphony_config: dict, start_date: str = None, 
                            end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare all data needed for a symphony using database
        
        Args:
            symphony_config: Symphony configuration dictionary
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbol -> DataFrame with market data
        """
        
        universe = symphony_config.get('universe', [])
        
        if not universe:
            raise ValueError("No universe defined in symphony configuration")
        
        print(f"📊 Preparing data for symphony: {symphony_config.get('name', 'Unknown')}")
        print(f"🎯 Universe: {universe}")
        print(f"📅 Date range: {start_date} to {end_date}")
        
        # Fetch data for all symbols using database (NO rate limiting!)
        market_data = {}
        
        for symbol in universe:
            try:
                print(f"📈 Getting {symbol} data from database...")
                data = self.database.get_data(symbol, start_date, end_date, 'daily')
                
                if not data.empty:
                    market_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} records")
                else:
                    print(f"⚠️ {symbol}: No data available")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
                continue
        
        # Validate we have sufficient data  
        min_required_days = 100  # Back to normal now that we have historical data
        valid_symbols = []
        
        for symbol, data in market_data.items():
            if len(data) >= min_required_days:
                valid_symbols.append(symbol)
            else:
                print(f"⚠️ {symbol} has only {len(data)} days of data (min {min_required_days})")
        
        print(f"✅ Valid symbols with sufficient data: {valid_symbols}")
        
        return {symbol: data for symbol, data in market_data.items() if symbol in valid_symbols}

class SymphonyComponentFactory:
    """Factory for creating symphony components without circular dependencies"""
    
    def __init__(self, config: SymphonyConfig = None):
        self.config = config or SymphonyConfig()
        self._components = {}
    
    def get_data_pipeline(self):
        """Get or create data pipeline component - now using database"""
        if 'data_pipeline' not in self._components:
            from market_data_database import MarketDataDatabase
            
            self._components['data_pipeline'] = MarketDataDatabase(
                api_key=self.config.api_key
            )
        
        return self._components['data_pipeline']
    
    def get_data_manager(self):
        """Get or create data manager component"""
        if 'data_manager' not in self._components:
            # Create a new data manager that works with our database
            pipeline = self.get_data_pipeline()
            self._components['data_manager'] = DatabaseDataManager(pipeline)
        
        return self._components['data_manager']
    
    def get_engine(self):
        """Get or create symphony engine component"""
        if 'engine' not in self._components:
            from symphony_engine import SymphonyEngine
            self._components['engine'] = SymphonyEngine()
        
        return self._components['engine']
    
    def get_backtester(self):
        """Get or create backtester component"""
        if 'backtester' not in self._components:
            from symphony_engine import SymphonyBacktester
            self._components['backtester'] = SymphonyBacktester()
        
        return self._components['backtester']
    
    def get_visualizer(self):
        """Get or create visualizer component"""
        if 'visualizer' not in self._components:
            from symphony_visualizer import SymphonyVisualizer
            self._components['visualizer'] = SymphonyVisualizer()
        
        return self._components['visualizer']
    
    def get_forecaster(self):
        """Get or create forecaster component"""
        if 'forecaster' not in self._components:
            from symphony_forecaster import SymphonyForecaster
            self._components['forecaster'] = SymphonyForecaster()
        
        return self._components['forecaster']
    
    def get_optimizer(self):
        """Get or create optimizer component"""
        if 'optimizer' not in self._components:
            from symphony_optimizer import SymphonyOptimizer
            self._components['optimizer'] = SymphonyOptimizer()
        
        return self._components['optimizer']
    
    def get_composer_parser(self):
        """Get or create Composer DSL parser"""
        if 'composer_parser' not in self._components:
            from composer_compatibility import ComposerDSLParser
            self._components['composer_parser'] = ComposerDSLParser()
        
        return self._components['composer_parser']
    
    def get_composer_reconciler(self):
        """Get or create Composer reconciler"""  
        if 'composer_reconciler' not in self._components:
            from composer_compatibility import ComposerResultsReconciliation
            self._components['composer_reconciler'] = ComposerResultsReconciliation()
        
        return self._components['composer_reconciler']

class SymphonyService:
    """Core service layer for symphony operations"""
    
    def __init__(self, config: SymphonyConfig = None):
        self.config = config or SymphonyConfig()
        self.factory = SymphonyComponentFactory(self.config)
        
        # Set environment variables if needed
        if self.config.api_key:
            os.environ['ALPHA_VANTAGE_API_KEY'] = self.config.api_key
    
    def create_sample_symphony(self, output_path: str = "sample_symphony_v2.json") -> dict:
        """Create a sample symphony configuration"""
        
        sample_config = {
            "name": "Momentum Quality Strategy",
            "description": "Buy top 3 momentum stocks when market is bullish, defensive when bearish",
            "universe": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ", "TLT"],
            "rebalance_frequency": "monthly",
            
            "logic": {
                "conditions": [
                    {
                        "id": "market_momentum_check",
                        "type": "if_statement", 
                        "condition": {
                            "metric": "cumulative_return",
                            "asset_1": "SPY",
                            "operator": "greater_than",
                            "asset_2": {"type": "fixed_value", "value": 0.0},
                            "lookback_days": 60
                        },
                        "if_true": "momentum_allocation",
                        "if_false": "defensive_allocation"
                    }
                ],
                
                "allocations": {
                    "momentum_allocation": {
                        "type": "sort_and_weight",
                        "sort": {
                            "metric": "cumulative_return", 
                            "lookback_days": 90,
                            "direction": "top",
                            "count": 3
                        },
                        "weighting": {
                            "method": "inverse_volatility",
                            "lookback_days": 30
                        }
                    },
                    
                    "defensive_allocation": {
                        "type": "fixed_allocation",
                        "weights": {
                            "TLT": 0.7,
                            "SPY": 0.3
                        }
                    }
                }
            }
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"📝 Created sample symphony: {output_path}")
        return sample_config
    
    def validate_symphony(self, symphony_config: dict) -> bool:
        """Validate symphony configuration"""
        
        print("🔍 Validating symphony configuration...")
        
        required_fields = ['name', 'universe', 'logic']
        for field in required_fields:
            if field not in symphony_config:
                print(f"❌ Missing required field: {field}")
                return False
        
        universe = symphony_config['universe']
        if not isinstance(universe, list) or len(universe) == 0:
            print("❌ Universe must be a non-empty list")
            return False
        
        logic = symphony_config['logic']
        if 'allocations' not in logic:
            print("❌ No allocations defined in logic")
            return False
        
        print("✅ Symphony configuration is valid")
        return True
    
    def load_symphony_config(self, config_path: str) -> dict:
        """Load symphony configuration from JSON file"""
        import json
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Loaded symphony: {config.get('name', 'Unknown')}")
            return config
        except Exception as e:
            raise ValueError(f"Error loading symphony config: {e}")

# Global service instance
_service_instance = None

def get_symphony_service(config: SymphonyConfig = None) -> SymphonyService:
    """Get global symphony service instance"""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = SymphonyService(config)
    
    return _service_instance

def reset_symphony_service():
    """Reset global service instance (for testing)"""
    global _service_instance
    _service_instance = None

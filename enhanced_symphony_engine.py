"""
Enhanced Symphony Engine with Black Swan Support
Extends base symphony engine with crisis detection and advanced allocation methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from symphony_engine import SymphonyEngine
from market_data_database import MarketDataDatabase
from simple_data_access import get_simple_data

logger = logging.getLogger(__name__)

class EnhancedSymphonyEngine(SymphonyEngine):
    """Extended symphony engine with crisis indicators and dynamic allocation"""
    
    def __init__(self, symphony_config: dict, db: MarketDataDatabase):
        self.config = symphony_config
        self.db = db
        self.name = symphony_config['name']
        self.universe = symphony_config['universe']
        self.logic = symphony_config['logic']
        self.rebalance_frequency = symphony_config.get('rebalance_frequency', 'monthly')
        
    def calculate_allocations(self, date: datetime) -> Dict[str, float]:
        """Calculate portfolio allocations for given date"""
        try:
            # Execute conditional logic
            allocation_key = self._evaluate_conditions(date)
            
            # Get allocation based on condition result
            allocation_config = self.logic['allocations'][allocation_key]
            
            # Calculate actual allocations
            allocations = self._calculate_allocation(allocation_config, date)
            
            # Normalize weights
            total_weight = sum(allocations.values())
            if total_weight > 0:
                allocations = {k: v/total_weight for k, v in allocations.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating allocations for {self.name}: {e}")
            # Return equal weight default
            equal_weight = 1.0 / len(self.universe)
            return {symbol: equal_weight for symbol in self.universe[:3]}  # Top 3 assets
    
    def _evaluate_conditions(self, date: datetime) -> str:
        """Evaluate conditional logic and return allocation key"""
        conditions = self.logic.get('conditions', [])
        
        if not conditions:
            # No conditions, use first allocation
            return list(self.logic['allocations'].keys())[0]
        
        # Build a map of condition IDs for chained evaluation
        condition_map = {cond['id']: cond for cond in conditions}
        
        # Start with first condition
        current_condition_id = conditions[0]['id']
        
        while current_condition_id:
            if current_condition_id not in condition_map:
                # If it's not a condition ID, it must be an allocation key
                return current_condition_id
            
            condition = condition_map[current_condition_id]
            
            if condition['type'] == 'if_statement':
                if self._evaluate_condition(condition['condition'], date):
                    result = condition['if_true']
                else:
                    result = condition['if_false']
                
                # Check if result is another condition or final allocation
                if result in condition_map:
                    current_condition_id = result
                else:
                    return result
                    
            elif condition['type'] == 'market_regime':
                return self._evaluate_market_regime(condition, date)
            else:
                break
        
        # Default to first allocation
        return list(self.logic['allocations'].keys())[0]
    
    def _evaluate_condition(self, condition: dict, date: datetime) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get('type', 'simple')
        
        if condition_type == 'or':
            return any(self._evaluate_condition(c, date) for c in condition['conditions'])
        elif condition_type == 'and':
            return all(self._evaluate_condition(c, date) for c in condition['conditions'])
        else:
            # Simple condition
            return self._evaluate_simple_condition(condition, date)
    
    def _evaluate_simple_condition(self, condition: dict, date: datetime) -> bool:
        """Evaluate a simple metric-based condition"""
        metric = condition['metric']
        asset_1 = condition['asset_1']
        operator = condition['operator']
        
        if metric == 'vix_level':
            vix_value = self._get_vix_level(date)
            threshold = condition['asset_2']['value']
            return self._compare_values(vix_value, threshold, operator)
        
        elif metric == 'credit_spread':
            spread = self._calculate_credit_spread(date)
            threshold = condition['threshold']['value']
            return self._compare_values(spread, threshold, operator)
        
        elif metric == 'moving_average':
            return self._check_moving_average(asset_1, condition['asset_2'], date)
        
        elif metric == 'cumulative_return':
            return_val = self._calculate_cumulative_return(
                asset_1, 
                condition['lookback_days'], 
                date
            )
            threshold = condition['asset_2']['value']
            return self._compare_values(return_val, threshold, operator)
        
        elif metric == 'trend_score':
            score = self._calculate_trend_score(asset_1, condition['lookback_days'], date)
            threshold = condition['asset_2']['value']
            return self._compare_values(score, threshold, operator)
        
        else:
            logger.warning(f"Unknown metric: {metric}")
            return False
    
    def _compare_values(self, value1: float, value2: float, operator: str) -> bool:
        """Compare two values based on operator"""
        if operator == 'greater_than':
            return value1 > value2
        elif operator == 'less_than':
            return value1 < value2
        elif operator == 'equal_to':
            return abs(value1 - value2) < 0.0001
        else:
            return False
    
    def _get_vix_level(self, date: datetime) -> float:
        """Get VIX level (using VXX as proxy)"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=5)).strftime('%Y-%m-%d')
            data = get_simple_data('VXX', start_date, end_date)
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return 20.0  # Default VIX
    
    def _calculate_credit_spread(self, date: datetime) -> float:
        """Calculate credit spread between HYG and TLT"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=20)).strftime('%Y-%m-%d')
            hyg_data = get_simple_data('HYG', start_date, end_date)
            tlt_data = get_simple_data('TLT', start_date, end_date)
            
            if not hyg_data.empty and not tlt_data.empty:
                hyg_yield = -np.log(hyg_data['Close'] / hyg_data['Close'].shift(1)).mean() * 252
                tlt_yield = -np.log(tlt_data['Close'] / tlt_data['Close'].shift(1)).mean() * 252
                return (hyg_yield - tlt_yield) * 10000  # basis points
        except:
            pass
        return 300  # Default spread
    
    def _check_moving_average(self, symbol: str, ma_config: dict, date: datetime) -> bool:
        """Check if price is above/below moving average"""
        try:
            period = ma_config.get('period', 200)
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=period+50)).strftime('%Y-%m-%d')
            data = get_simple_data(symbol, start_date, end_date)
            
            if not data.empty:
                ma = data['Close'].rolling(period).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                return current_price > ma
        except:
            pass
        return True  # Default to bullish
    
    def _calculate_cumulative_return(self, symbol: str, lookback_days: int, 
                                    date: datetime) -> float:
        """Calculate cumulative return over lookback period"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=lookback_days+5)).strftime('%Y-%m-%d')
            data = get_simple_data(symbol, start_date, end_date)
            
            if len(data) >= lookback_days:
                return (data['Close'].iloc[-1] / data['Close'].iloc[-lookback_days]) - 1
        except:
            pass
        return 0.0
    
    def _calculate_trend_score(self, symbol: str, lookback_days: int, 
                              date: datetime) -> float:
        """Calculate trend strength score (0-1)"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=lookback_days*2)).strftime('%Y-%m-%d')
            data = get_simple_data(symbol, start_date, end_date)
            
            if not data.empty:
                # Multiple trend indicators
                sma_20 = data['Close'].rolling(20).mean()
                sma_50 = data['Close'].rolling(50).mean()
                
                # Trend score components
                price_above_sma20 = (data['Close'] > sma_20).iloc[-lookback_days:].mean()
                price_above_sma50 = (data['Close'] > sma_50).iloc[-lookback_days:].mean()
                sma20_above_sma50 = (sma_20 > sma_50).iloc[-lookback_days:].mean()
                
                # Momentum
                returns = data['Close'].pct_change()
                positive_days = (returns > 0).iloc[-lookback_days:].mean()
                
                # Combined score
                trend_score = (
                    price_above_sma20 * 0.3 +
                    price_above_sma50 * 0.3 +
                    sma20_above_sma50 * 0.2 +
                    positive_days * 0.2
                )
                
                return trend_score
        except:
            pass
        return 0.5  # Neutral trend
    
    def _evaluate_market_regime(self, condition: dict, date: datetime) -> str:
        """Evaluate market regime and return appropriate allocation"""
        metrics = condition['metrics']
        thresholds = condition['thresholds']
        allocations = condition['allocations']
        
        # Calculate regime scores
        vix_level = self._get_vix_level(date)
        vix_percentile = self._calculate_vix_percentile(date)
        trend_strength = self._calculate_trend_score('SPY', 30, date)
        credit_health = 1.0 - (self._calculate_credit_spread(date) / 1000)  # Normalize
        
        # Determine regime
        if (vix_percentile < thresholds['bull']['vix_percentile'] and 
            trend_strength > thresholds['bull']['trend_strength']):
            return allocations['bull']
        elif (vix_percentile > thresholds['bear']['vix_percentile'] and 
              trend_strength < thresholds['bear']['trend_strength']):
            return allocations['bear']
        else:
            return allocations['uncertainty']
    
    def _calculate_vix_percentile(self, date: datetime) -> float:
        """Calculate VIX percentile rank"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=252)).strftime('%Y-%m-%d')
            data = get_simple_data('VXX', start_date, end_date)
            if not data.empty:
                current_vix = data['Close'].iloc[-1]
                percentile = (data['Close'] < current_vix).mean()
                return percentile
        except:
            pass
        return 0.5
    
    def _calculate_allocation(self, allocation_config: dict, date: datetime) -> Dict[str, float]:
        """Calculate actual allocation based on allocation type"""
        allocation_type = allocation_config['type']
        
        if allocation_type == 'fixed_allocation':
            return allocation_config['weights'].copy()
        
        elif allocation_type == 'sort_and_weight':
            return self._sort_and_weight_allocation(allocation_config, date)
        
        elif allocation_type == 'dynamic_allocation':
            return self._dynamic_allocation(allocation_config, date)
        
        elif allocation_type == 'momentum_weighted':
            return self._momentum_weighted_allocation(allocation_config, date)
        
        elif allocation_type == 'risk_parity':
            return self._risk_parity_allocation(allocation_config, date)
        
        else:
            logger.warning(f"Unknown allocation type: {allocation_type}")
            return {}
    
    def _sort_and_weight_allocation(self, config: dict, date: datetime) -> Dict[str, float]:
        """Sort assets and apply weighting"""
        sort_config = config['sort']
        weight_config = config['weighting']
        
        # Get metric values for all assets
        metric_values = {}
        for symbol in self.universe:
            if symbol in ['CASH', 'UVXY', 'VXX']:  # Skip special assets
                continue
            
            if sort_config['metric'] == 'cumulative_return':
                value = self._calculate_cumulative_return(
                    symbol, 
                    sort_config['lookback_days'], 
                    date
                )
            elif sort_config['metric'] == 'sharpe_ratio':
                value = self._calculate_sharpe_ratio(
                    symbol,
                    sort_config['lookback_days'],
                    date
                )
            else:
                value = 0.0
            
            metric_values[symbol] = value
        
        # Sort and select top/bottom N
        sorted_assets = sorted(metric_values.items(), key=lambda x: x[1], 
                             reverse=(sort_config['direction'] == 'top'))
        selected_assets = sorted_assets[:sort_config['count']]
        
        # Apply weighting
        allocations = {}
        if weight_config['method'] == 'equal_weight':
            weight = 1.0 / len(selected_assets)
            for asset, _ in selected_assets:
                allocations[asset] = weight
        
        elif weight_config['method'] == 'momentum_weighted':
            total_momentum = sum(max(0, value) for _, value in selected_assets)
            if total_momentum > 0:
                for asset, value in selected_assets:
                    allocations[asset] = max(0, value) / total_momentum
        
        # Apply overlay if exists
        if 'overlay' in config:
            for asset, weight in config['overlay'].items():
                allocations[asset] = weight
        
        return allocations
    
    def _calculate_sharpe_ratio(self, symbol: str, lookback_days: int, 
                               date: datetime) -> float:
        """Calculate Sharpe ratio"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=lookback_days+5)).strftime('%Y-%m-%d')
            data = get_simple_data(symbol, start_date, end_date)
            
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 20:
                    return returns.mean() / returns.std() * np.sqrt(252)
        except:
            pass
        return 0.0
    
    def _dynamic_allocation(self, config: dict, date: datetime) -> Dict[str, float]:
        """Dynamic allocation based on crisis score"""
        base_weights = config['base_weights'].copy()
        
        # Calculate crisis score if specified
        if config.get('scaling_factor') == 'crisis_score':
            crisis_score = self._calculate_crisis_score(date)
            scale = min(2.0, 1.0 + (crisis_score - 2.0) * 0.3)
            
            # Scale volatility positions
            for asset in ['UVXY', 'VXX']:
                if asset in base_weights:
                    base_weights[asset] *= scale
        
        return base_weights
    
    def _calculate_crisis_score(self, date: datetime) -> float:
        """Calculate comprehensive crisis score"""
        vix = self._get_vix_level(date)
        credit_spread = self._calculate_credit_spread(date)
        
        crisis_score = (vix / 20) * 2 + (credit_spread / 300)
        return crisis_score
    
    def _momentum_weighted_allocation(self, config: dict, date: datetime) -> Dict[str, float]:
        """Momentum-based dynamic weighting"""
        base_weights = config.get('base_weights', {})
        lookback = config.get('momentum_lookback', 60)
        
        momentum_scores = {}
        for asset in base_weights:
            momentum = self._calculate_cumulative_return(asset, lookback, date)
            momentum_scores[asset] = max(0, momentum)  # Only positive momentum
        
        # Normalize by momentum
        total_momentum = sum(momentum_scores.values())
        if total_momentum > 0:
            allocations = {}
            for asset, base_weight in base_weights.items():
                momentum_weight = momentum_scores[asset] / total_momentum
                # Blend base weight with momentum weight
                allocations[asset] = base_weight * 0.5 + momentum_weight * 0.5
            return allocations
        else:
            return base_weights
    
    def _risk_parity_allocation(self, config: dict, date: datetime) -> Dict[str, float]:
        """Risk parity allocation"""
        assets = config['assets']
        target_risk = config.get('target_risk', 0.10)
        
        # Calculate volatilities
        volatilities = {}
        for asset in assets:
            vol = self._calculate_volatility(asset, 60, date)
            volatilities[asset] = vol
        
        # Inverse volatility weighting
        inv_vols = {asset: 1.0/vol for asset, vol in volatilities.items() if vol > 0}
        total_inv_vol = sum(inv_vols.values())
        
        allocations = {}
        for asset in assets:
            if asset in inv_vols:
                weight = inv_vols[asset] / total_inv_vol
                # Apply constraints
                min_weight = config['constraints'].get('min_weight', 0)
                max_weight = config['constraints'].get('max_weight', 1)
                weight = max(min_weight, min(max_weight, weight))
                allocations[asset] = weight
        
        return allocations
    
    def _calculate_volatility(self, symbol: str, lookback_days: int, 
                             date: datetime) -> float:
        """Calculate annualized volatility"""
        try:
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=lookback_days+5)).strftime('%Y-%m-%d')
            data = get_simple_data(symbol, start_date, end_date)
            
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 20:
                    return returns.std() * np.sqrt(252)
        except:
            pass
        return 0.15  # Default 15% volatility
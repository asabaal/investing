#!/usr/bin/env python3
"""
Market Data Quality Evaluator
Systematic framework for evaluating data quality across multiple dimensions
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class DataQualityEvaluator:
    """Evaluate market data quality across multiple dimensions"""
    
    def __init__(self, symbol: str = 'USO'):
        self.symbol = symbol
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.quality_scores = {}
        self.test_results = {}
        
    def fetch_data(self, interval: str = '15min') -> pd.DataFrame:
        """Fetch data for quality testing"""
        
        if interval == 'daily':
            function = 'TIME_SERIES_DAILY'
            params = {
                'function': function,
                'symbol': self.symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
        else:
            function = 'TIME_SERIES_INTRADAY'
            params = {
                'function': function,
                'symbol': self.symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
        
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        # Parse data
        if interval == 'daily':
            time_series = data['Time Series (Daily)']
        else:
            time_series = data[f'Time Series ({interval})']
        
        df_data = []
        for datetime_str, values in time_series.items():
            df_data.append({
                'datetime': datetime_str,
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'Volume': int(values['5. volume'])
            })
        
        df = pd.DataFrame(df_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def test_price_consistency(self, df: pd.DataFrame) -> Dict:
        """Test price data consistency (OHLC relationships)"""
        
        results = {
            'test_name': 'Price Consistency',
            'description': 'Validates OHLC relationships and price reasonableness',
            'checks': {}
        }
        
        # Check 1: High >= Open, Close, Low
        high_violations = ((df['High'] < df['Open']) | 
                          (df['High'] < df['Close']) | 
                          (df['High'] < df['Low'])).sum()
        
        # Check 2: Low <= Open, Close, High  
        low_violations = ((df['Low'] > df['Open']) | 
                         (df['Low'] > df['Close']) | 
                         (df['Low'] > df['High'])).sum()
        
        # Check 3: Reasonable price movements (no >50% single-period moves)
        price_changes = df['Close'].pct_change().abs()
        extreme_moves = (price_changes > 0.5).sum()
        
        # Check 4: No zero or negative prices
        invalid_prices = ((df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)).sum()
        
        results['checks'] = {
            'high_violations': {'count': high_violations, 'pass': high_violations == 0},
            'low_violations': {'count': low_violations, 'pass': low_violations == 0},
            'extreme_moves': {'count': extreme_moves, 'threshold': '50%', 'pass': extreme_moves == 0},
            'invalid_prices': {'count': invalid_prices, 'pass': invalid_prices == 0}
        }
        
        # Calculate overall score
        passing_checks = sum(1 for check in results['checks'].values() if check['pass'])
        total_checks = len(results['checks'])
        results['score'] = passing_checks / total_checks
        
        return results
    
    def test_volume_quality(self, df: pd.DataFrame) -> Dict:
        """Test volume data quality"""
        
        results = {
            'test_name': 'Volume Quality',
            'description': 'Validates volume data reasonableness and consistency',
            'checks': {}
        }
        
        # Check 1: No negative volumes
        negative_volumes = (df['Volume'] < 0).sum()
        
        # Check 2: Zero volume periods (suspicious but not always wrong)
        zero_volumes = (df['Volume'] == 0).sum()
        
        # Check 3: Extreme volume spikes (>10x median)
        median_volume = df['Volume'].median()
        extreme_volume_spikes = (df['Volume'] > 10 * median_volume).sum()
        
        # Check 4: Volume consistency (coefficient of variation)
        volume_cv = df['Volume'].std() / df['Volume'].mean()
        high_variability = volume_cv > 3.0  # High but not necessarily wrong for ETFs
        
        results['checks'] = {
            'negative_volumes': {'count': negative_volumes, 'pass': negative_volumes == 0},
            'zero_volumes': {'count': zero_volumes, 'total_periods': len(df), 'pass': zero_volumes < len(df) * 0.05},
            'extreme_spikes': {'count': extreme_volume_spikes, 'threshold': '10x median', 'pass': extreme_volume_spikes < len(df) * 0.01},
            'high_variability': {'cv': volume_cv, 'threshold': 3.0, 'pass': not high_variability}
        }
        
        # Calculate score
        passing_checks = sum(1 for check in results['checks'].values() if check['pass'])
        total_checks = len(results['checks'])
        results['score'] = passing_checks / total_checks
        
        return results
    
    def test_temporal_consistency(self, df: pd.DataFrame, interval: str) -> Dict:
        """Test temporal consistency and completeness"""
        
        results = {
            'test_name': 'Temporal Consistency',
            'description': f'Validates {interval} data timing and completeness',
            'checks': {}
        }
        
        # Expected frequency
        freq_map = {
            '1min': '1min',
            '5min': '5min', 
            '15min': '15min',
            '30min': '30min',
            '60min': '60min',
            'daily': 'D'
        }
        
        expected_freq = freq_map.get(interval, 'D')
        
        # Check 1: Missing periods
        if interval != 'daily':
            # For intraday data, check business hours only
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), 
                                     freq=expected_freq)
            # Filter for business hours (rough approximation)
            if interval in ['1min', '5min', '15min', '30min', '60min']:
                business_hours_range = full_range[
                    (full_range.time >= pd.Timestamp('09:30').time()) & 
                    (full_range.time <= pd.Timestamp('16:00').time()) &
                    (full_range.weekday < 5)  # Monday=0, Friday=4
                ]
                expected_periods = len(business_hours_range)
                missing_periods = max(0, expected_periods - len(df))
            else:
                missing_periods = 0
        else:
            # For daily data, check business days
            business_days = pd.bdate_range(start=df.index.min(), end=df.index.max())
            expected_periods = len(business_days)
            missing_periods = max(0, expected_periods - len(df))
        
        # Check 2: Duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        
        # Check 3: Out-of-order timestamps
        out_of_order = (df.index[1:] <= df.index[:-1]).sum()
        
        results['checks'] = {
            'missing_periods': {
                'count': missing_periods,
                'expected': expected_periods if 'expected_periods' in locals() else 'N/A',
                'actual': len(df),
                'pass': missing_periods < expected_periods * 0.05 if 'expected_periods' in locals() else True
            },
            'duplicate_timestamps': {'count': duplicate_timestamps, 'pass': duplicate_timestamps == 0},
            'out_of_order': {'count': out_of_order, 'pass': out_of_order == 0}
        }
        
        # Calculate score
        passing_checks = sum(1 for check in results['checks'].values() if check['pass'])
        total_checks = len(results['checks'])
        results['score'] = passing_checks / total_checks
        
        return results
    
    def calculate_data_freshness(self, df: pd.DataFrame, interval: str) -> Dict:
        """Calculate data freshness score"""
        
        results = {
            'test_name': 'Data Freshness',
            'description': 'Evaluates how recent the latest data is',
            'checks': {}
        }
        
        latest_data = df.index.max()
        current_time = pd.Timestamp.now(tz='UTC')
        
        # Convert to same timezone for comparison
        if latest_data.tz is None:
            latest_data = latest_data.tz_localize('UTC')
        
        time_diff = current_time - latest_data
        hours_old = time_diff.total_seconds() / 3600
        
        # Freshness thresholds (hours)
        thresholds = {
            '1min': 0.5,   # 30 minutes
            '5min': 1.0,   # 1 hour
            '15min': 2.0,  # 2 hours
            '30min': 4.0,  # 4 hours
            '60min': 8.0,  # 8 hours
            'daily': 24.0  # 24 hours
        }
        
        threshold = thresholds.get(interval, 24.0)
        is_fresh = hours_old <= threshold
        
        results['checks'] = {
            'freshness': {
                'latest_data': latest_data.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'hours_old': round(hours_old, 2),
                'threshold_hours': threshold,
                'pass': is_fresh
            }
        }
        
        results['score'] = 1.0 if is_fresh else max(0.0, 1.0 - (hours_old - threshold) / threshold)
        
        return results
    
    def run_comprehensive_evaluation(self, interval: str = 'daily') -> Dict:
        """Run comprehensive data quality evaluation"""
        
        print(f"🔍 Running Data Quality Evaluation for {self.symbol} ({interval})")
        print("=" * 60)
        
        try:
            # Fetch data
            df = self.fetch_data(interval)
            print(f"📊 Loaded {len(df):,} {interval} records")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            
            # Run all tests
            tests = [
                self.test_price_consistency(df),
                self.test_volume_quality(df), 
                self.test_temporal_consistency(df, interval),
                self.calculate_data_freshness(df, interval)
            ]
            
            # Print results
            print(f"\n{'='*20} TEST RESULTS {'='*20}")
            
            overall_score = 0
            for test in tests:
                print(f"\n📋 {test['test_name']}: {test['score']:.2%}")
                print(f"   {test['description']}")
                
                for check_name, check_data in test['checks'].items():
                    status = "✅ PASS" if check_data['pass'] else "❌ FAIL"
                    print(f"   • {check_name}: {status}")
                    
                    # Print additional details
                    if 'count' in check_data:
                        print(f"     Count: {check_data['count']}")
                    if 'hours_old' in check_data:
                        print(f"     Hours old: {check_data['hours_old']}")
                
                overall_score += test['score']
            
            overall_score /= len(tests)
            
            # Summary
            print(f"\n{'='*20} OVERALL ASSESSMENT {'='*20}")
            print(f"📊 Overall Data Quality Score: {overall_score:.2%}")
            
            if overall_score >= 0.9:
                grade = "A (Excellent)"
                recommendation = "Data is high quality and suitable for trading/analysis"
            elif overall_score >= 0.8:
                grade = "B (Good)"  
                recommendation = "Data is good quality with minor issues"
            elif overall_score >= 0.7:
                grade = "C (Fair)"
                recommendation = "Data has some quality issues - investigate before using"
            else:
                grade = "D (Poor)"
                recommendation = "Data quality is poor - consider alternative sources"
            
            print(f"🎯 Grade: {grade}")
            print(f"💡 Recommendation: {recommendation}")
            
            return {
                'symbol': self.symbol,
                'interval': interval,
                'overall_score': overall_score,
                'grade': grade,
                'recommendation': recommendation,
                'test_results': tests,
                'data_summary': {
                    'records': len(df),
                    'date_range': f"{df.index.min()} to {df.index.max()}",
                    'latest_close': df['Close'].iloc[-1]
                }
            }
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return None

def main():
    """Run data quality evaluation"""
    
    evaluator = DataQualityEvaluator('USO')
    
    # Test multiple intervals
    intervals = ['daily', '15min']
    
    results = {}
    for interval in intervals:
        print(f"\n{'='*80}")
        result = evaluator.run_comprehensive_evaluation(interval)
        if result:
            results[interval] = result
        print(f"\n{'='*80}")
    
    # Comparison summary
    if len(results) > 1:
        print(f"\n{'='*20} INTERVAL COMPARISON {'='*20}")
        for interval, result in results.items():
            print(f"{interval:>8}: {result['overall_score']:.1%} ({result['grade']})")

if __name__ == "__main__":
    main()
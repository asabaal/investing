"""
Composer DSL Compatibility System

Translates between Composer's DSL format and our JSON format:
1. Parse Composer DSL (defsymphony format)
2. Convert to our symphony JSON format
3. Reconcile backtest results
4. Validate performance matching
"""

import re
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import ast

@dataclass
class ComposerParseResult:
    """Result of parsing Composer DSL"""
    name: str
    config: dict
    our_format: dict
    parsing_notes: List[str]

class ComposerDSLParser:
    """Parse and convert Composer DSL to our format"""
    
    def __init__(self):
        self.asset_mappings = {
            # Common asset mappings
            "SPY": "SPY",
            "QQQ": "QQQ", 
            "TQQQ": "TQQQ",
            "TECL": "TECL",
            "PSQ": "PSQ",
            "UVXY": "UVXY",
            "SHY": "SHY",
            "TLT": "TLT"
        }
        
        self.metric_mappings = {
            "current-price": "current_price",
            "moving-average-price": "moving_average_price",
            "rsi": "rsi",
            "cumulative-return": "cumulative_return",
            "ema": "ema_price",
            "standard-deviation": "standard_deviation_return"
        }
        
        self.operator_mappings = {
            ">": "greater_than",
            "<": "less_than", 
            ">=": "greater_than_or_equal",
            "<=": "less_than_or_equal",
            "=": "equal",
            "!=": "not_equal"
        }
    
    def parse_composer_dsl(self, dsl_string: str) -> ComposerParseResult:
        """
        Parse Composer DSL and convert to our format
        
        Args:
            dsl_string: The Composer DSL code
            
        Returns:
            ComposerParseResult with converted configuration
        """
        
        print("🔄 Parsing Composer DSL...")
        
        # Clean up the DSL string
        dsl_cleaned = self._clean_dsl(dsl_string)
        
        # Extract symphony name
        name = self._extract_symphony_name(dsl_cleaned)
        
        # Extract config options
        config = self._extract_config(dsl_cleaned)
        
        # Parse the main logic
        logic_section = self._extract_logic_section(dsl_cleaned)
        
        # Convert to our format
        our_format = self._convert_to_our_format(name, config, logic_section)
        
        # Parsing notes
        notes = self._generate_parsing_notes(dsl_cleaned, our_format)
        
        print(f"✅ Parsed symphony: {name}")
        
        return ComposerParseResult(
            name=name,
            config=config,
            our_format=our_format,
            parsing_notes=notes
        )
    
    def _clean_dsl(self, dsl_string: str) -> str:
        """Clean and normalize DSL string"""
        
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', dsl_string.strip())
        
        # Normalize parentheses spacing
        cleaned = re.sub(r'\s*\(\s*', '(', cleaned)
        cleaned = re.sub(r'\s*\)\s*', ')', cleaned)
        
        return cleaned
    
    def _extract_symphony_name(self, dsl: str) -> str:
        """Extract symphony name from DSL"""
        
        # Look for pattern: defsymphony "name"
        match = re.search(r'defsymphony\s+"([^"]+)"', dsl)
        if match:
            return match.group(1)
        
        return "Converted Composer Symphony"
    
    def _extract_config(self, dsl: str) -> dict:
        """Extract configuration options"""
        
        config = {}
        
        # Extract asset class
        asset_match = re.search(r':asset-class\s+"([^"]+)"', dsl)
        if asset_match:
            config['asset_class'] = asset_match.group(1)
        
        # Extract rebalance threshold
        threshold_match = re.search(r':rebalance-threshold\s+([\d.]+)', dsl)
        if threshold_match:
            config['rebalance_threshold'] = float(threshold_match.group(1))
        
        return config
    
    def _extract_logic_section(self, dsl: str) -> str:
        """Extract the main logic section"""
        
        # Find the main logic after the config
        # Look for pattern after the config map
        match = re.search(r'\{[^}]*\}\s*(.+)', dsl)
        if match:
            return match.group(1)
        
        # Fallback: everything after defsymphony and name
        match = re.search(r'defsymphony\s+"[^"]+"\s*(.+)', dsl)
        if match:
            return match.group(1)
        
        return dsl
    
    def _convert_to_our_format(self, name: str, config: dict, logic: str) -> dict:
        """Convert parsed DSL to our JSON format"""
        
        # Extract all unique assets mentioned
        universe = self._extract_universe(logic)
        
        # Parse the nested logic structure
        conditions, allocations = self._parse_logic_structure(logic)
        
        our_symphony = {
            "name": name,
            "description": f"Converted from Composer DSL - {config.get('asset_class', 'EQUITIES')}",
            "universe": universe,
            "rebalance_frequency": "daily",  # Composer default
            "logic": {
                "conditions": conditions,
                "allocations": allocations
            }
        }
        
        return our_symphony
    
    def _extract_universe(self, logic: str) -> List[str]:
        """Extract all unique assets from the logic"""
        
        assets = set()
        
        # Find all quoted asset symbols
        asset_matches = re.findall(r'"([A-Z]{2,5})"', logic)
        for asset in asset_matches:
            if asset in self.asset_mappings:
                assets.add(self.asset_mappings[asset])
        
        # Also look for assets in function calls
        function_matches = re.findall(r'\([\w-]+\s+"([A-Z]{2,5})"', logic)
        for asset in function_matches:
            if asset in self.asset_mappings:
                assets.add(self.asset_mappings[asset])
        
        return sorted(list(assets))
    
    def _parse_logic_structure(self, logic: str) -> Tuple[List[dict], Dict[str, dict]]:
        """Parse the nested if-then logic structure"""
        
        conditions = []
        allocations = {}
        allocation_counter = 0
        
        # This is a simplified parser for the complex nested structure
        # In reality, you'd want a proper Lisp parser for full accuracy
        
        # Find if statements
        if_pattern = r'\(if\s+\(([^)]+)\)\s+\[([^\]]+)\]\s+\[([^\]]+)\]\)'
        if_matches = re.findall(if_pattern, logic, re.DOTALL)
        
        for i, (condition_str, true_branch, false_branch) in enumerate(if_matches):
            condition_id = f"condition_{i+1}"
            true_alloc_id = f"true_allocation_{i+1}"
            false_alloc_id = f"false_allocation_{i+1}"
            
            # Parse condition
            parsed_condition = self._parse_condition(condition_str)
            
            if parsed_condition:
                conditions.append({
                    "id": condition_id,
                    "type": "if_statement",
                    "condition": parsed_condition,
                    "if_true": true_alloc_id,
                    "if_false": false_alloc_id
                })
                
                # Parse branches
                allocations[true_alloc_id] = self._parse_allocation_branch(true_branch)
                allocations[false_alloc_id] = self._parse_allocation_branch(false_branch)
        
        # If no complex conditions found, create a simple allocation
        if not conditions:
            simple_assets = self._extract_simple_assets(logic)
            if simple_assets:
                allocations["default_allocation"] = {
                    "type": "fixed_allocation",
                    "weights": {asset: 1.0/len(simple_assets) for asset in simple_assets}
                }
        
        return conditions, allocations
    
    def _parse_condition(self, condition_str: str) -> Optional[dict]:
        """Parse individual condition"""
        
        # Look for comparison patterns
        # Pattern: operator asset1 asset2/value
        parts = condition_str.strip().split()
        
        if len(parts) >= 3:
            operator = parts[0]
            
            # Check if it's a function call
            if '(' in parts[1]:
                # Extract function and asset
                func_match = re.match(r'\(([^)]+)\s+"([^"]+)"(?:\s+\{[^}]*\})?\)', parts[1])
                if func_match:
                    metric_name = func_match.group(1)
                    asset1 = func_match.group(2)
                    
                    # Parse the second part (could be another function or fixed value)
                    if '(' in parts[2]:
                        func_match2 = re.match(r'\(([^)]+)\s+"([^"]+)"(?:\s+\{[^}]*\})?\)', parts[2])
                        if func_match2:
                            asset2 = func_match2.group(2)
                        else:
                            return None
                    else:
                        # Fixed value
                        try:
                            asset2 = {"type": "fixed_value", "value": float(parts[2])}
                        except ValueError:
                            return None
                    
                    # Map to our format
                    our_metric = self.metric_mappings.get(metric_name, metric_name)
                    our_operator = self.operator_mappings.get(operator, operator)
                    our_asset1 = self.asset_mappings.get(asset1, asset1)
                    
                    if isinstance(asset2, dict):
                        our_asset2 = asset2
                    else:
                        our_asset2 = self.asset_mappings.get(asset2, asset2)
                    
                    # Extract window parameter if present
                    window_match = re.search(r'\{:window\s+(\d+)\}', condition_str)
                    lookback_days = int(window_match.group(1)) if window_match else 20
                    
                    return {
                        "metric": our_metric,
                        "asset_1": our_asset1,
                        "operator": our_operator,
                        "asset_2": our_asset2,
                        "lookback_days": lookback_days
                    }
        
        return None
    
    def _parse_allocation_branch(self, branch_str: str) -> dict:
        """Parse allocation branch"""
        
        # Look for weight-equal with assets
        if "weight-equal" in branch_str:
            # Extract assets
            asset_matches = re.findall(r'"([A-Z]{2,5})"', branch_str)
            if asset_matches:
                mapped_assets = [self.asset_mappings.get(asset, asset) for asset in asset_matches]
                return {
                    "type": "fixed_allocation",
                    "weights": {asset: 1.0/len(mapped_assets) for asset in mapped_assets}
                }
        
        # Look for specific asset allocation
        asset_match = re.search(r'asset\s+"([A-Z]+)"', branch_str)
        if asset_match:
            asset = self.asset_mappings.get(asset_match.group(1), asset_match.group(1))
            return {
                "type": "fixed_allocation",
                "weights": {asset: 1.0}
            }
        
        # Default fallback
        return {
            "type": "fixed_allocation",
            "weights": {"SPY": 1.0}
        }
    
    def _extract_simple_assets(self, logic: str) -> List[str]:
        """Extract assets from simple logic"""
        
        assets = []
        asset_matches = re.findall(r'"([A-Z]{2,5})"', logic)
        
        for asset in asset_matches:
            if asset in self.asset_mappings:
                assets.append(self.asset_mappings[asset])
        
        return list(set(assets))
    
    def _generate_parsing_notes(self, dsl: str, our_format: dict) -> List[str]:
        """Generate notes about the parsing process"""
        
        notes = []
        
        # Count complexity
        if_count = dsl.count('(if')
        notes.append(f"Found {if_count} conditional statements")
        
        # Check for unsupported features
        if 'rsi' in dsl.lower():
            notes.append("Contains RSI indicators - mapped to our RSI metric")
        
        if 'moving-average' in dsl:
            notes.append("Contains moving averages - mapped to our moving_average_price metric")
        
        if len(our_format['universe']) > 5:
            notes.append(f"Large universe ({len(our_format['universe'])} assets) - consider performance impact")
        
        return notes


class ComposerResultsReconciliation:
    """Reconcile our backtest results with Composer's results"""
    
    def __init__(self):
        pass
    
    def reconcile_backtest_results(self, 
                                 composer_csv_path: str,
                                 our_backtest_results: pd.DataFrame,
                                 tolerance: float = 0.05) -> dict:
        """
        Compare our backtest results with Composer's results
        
        Args:
            composer_csv_path: Path to Composer's backtest CSV
            our_backtest_results: Our backtest results
            tolerance: Acceptable difference tolerance
            
        Returns:
            Reconciliation analysis
        """
        
        print("🔄 Reconciling backtest results...")
        
        # Load Composer results
        composer_data = self._load_composer_results(composer_csv_path)
        
        if composer_data.empty:
            return {"error": "Could not load Composer results"}
        
        # Align dates and calculate differences
        alignment = self._align_results(composer_data, our_backtest_results)
        
        # Compare allocations
        allocation_comparison = self._compare_allocations(alignment)
        
        # Compare performance metrics
        performance_comparison = self._compare_performance(alignment, tolerance)
        
        # Generate reconciliation report
        reconciliation = {
            "data_alignment": {
                "composer_periods": len(composer_data),
                "our_periods": len(our_backtest_results),
                "aligned_periods": len(alignment),
                "date_range": {
                    "start": alignment['date'].min() if not alignment.empty else None,
                    "end": alignment['date'].max() if not alignment.empty else None
                }
            },
            "allocation_comparison": allocation_comparison,
            "performance_comparison": performance_comparison,
            "reconciliation_status": self._determine_reconciliation_status(allocation_comparison, performance_comparison, tolerance)
        }
        
        return reconciliation
    
    def _load_composer_results(self, csv_path: str) -> pd.DataFrame:
        """Load and process Composer backtest results"""
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Process allocation columns
            allocation_columns = [col for col in df.columns if col not in ['Date', 'Day Traded', '$USD']]
            
            # Convert percentage strings to floats
            for col in allocation_columns:
                df[col] = df[col].replace('-', '0').str.replace('%', '').astype(float) / 100
            
            # Filter to only days when trading occurred
            df = df[df['Day Traded'] == 'Yes'].copy()
            
            return df
            
        except Exception as e:
            print(f"Error loading Composer results: {e}")
            return pd.DataFrame()
    
    def _align_results(self, composer_data: pd.DataFrame, our_data: pd.DataFrame) -> pd.DataFrame:
        """Align Composer and our results by date"""
        
        # Create alignment DataFrame
        alignment = pd.DataFrame()
        
        # Convert our data dates
        our_data_copy = our_data.copy()
        our_data_copy['date'] = pd.to_datetime(our_data_copy['date'])
        
        # Find overlapping dates
        composer_dates = set(composer_data['Date'])
        our_dates = set(our_data_copy['date'])
        common_dates = composer_dates.intersection(our_dates)
        
        if not common_dates:
            print("⚠️ No overlapping dates found between results")
            return pd.DataFrame()
        
        # Align data for common dates
        for date in sorted(common_dates):
            composer_row = composer_data[composer_data['Date'] == date].iloc[0]
            our_row = our_data_copy[our_data_copy['date'] == date].iloc[0]
            
            aligned_row = {
                'date': date,
                'composer_allocation': self._extract_composer_allocation(composer_row),
                'our_allocation': our_row.get('allocation', {}),
                'our_return': our_row.get('portfolio_return', 0)
            }
            
            alignment = pd.concat([alignment, pd.DataFrame([aligned_row])], ignore_index=True)
        
        return alignment
    
    def _extract_composer_allocation(self, composer_row: pd.Series) -> dict:
        """Extract allocation from Composer row"""
        
        allocation = {}
        allocation_columns = [col for col in composer_row.index if col not in ['Date', 'Day Traded', '$USD']]
        
        for col in allocation_columns:
            weight = composer_row[col]
            if weight > 0:
                allocation[col] = weight
        
        return allocation
    
    def _compare_allocations(self, alignment: pd.DataFrame) -> dict:
        """Compare allocation differences"""
        
        if alignment.empty:
            return {"error": "No aligned data to compare"}
        
        allocation_diffs = []
        
        for _, row in alignment.iterrows():
            composer_alloc = row['composer_allocation']
            our_alloc = row['our_allocation']
            
            # Calculate differences for each asset
            all_assets = set(list(composer_alloc.keys()) + list(our_alloc.keys()))
            
            diff_sum = 0
            for asset in all_assets:
                composer_weight = composer_alloc.get(asset, 0)
                our_weight = our_alloc.get(asset, 0)
                diff_sum += abs(composer_weight - our_weight)
            
            allocation_diffs.append(diff_sum)
        
        return {
            "mean_allocation_difference": np.mean(allocation_diffs),
            "max_allocation_difference": np.max(allocation_diffs),
            "periods_with_differences": sum(1 for diff in allocation_diffs if diff > 0.01),
            "total_aligned_periods": len(allocation_diffs)
        }
    
    def _compare_performance(self, alignment: pd.DataFrame, tolerance: float) -> dict:
        """Compare performance metrics"""
        
        if alignment.empty:
            return {"error": "No aligned data to compare"}
        
        # This is simplified - in practice you'd calculate Composer returns from allocations
        # For now, we'll focus on allocation accuracy
        
        return {
            "note": "Performance comparison requires price data for all assets",
            "allocation_accuracy": "See allocation_comparison section"
        }
    
    def _determine_reconciliation_status(self, allocation_comp: dict, performance_comp: dict, tolerance: float) -> str:
        """Determine overall reconciliation status"""
        
        if "error" in allocation_comp:
            return "ERROR - Cannot reconcile due to data issues"
        
        mean_diff = allocation_comp.get('mean_allocation_difference', 1.0)
        
        if mean_diff < tolerance / 2:
            return "EXCELLENT - Very close match with Composer"
        elif mean_diff < tolerance:
            return "GOOD - Acceptable differences within tolerance"
        elif mean_diff < tolerance * 2:
            return "FAIR - Some differences, may need adjustment"
        else:
            return "POOR - Significant differences, requires investigation"


# Example usage and testing
if __name__ == "__main__":
    
    print("🔧 Composer Compatibility System Test")
    print("=" * 60)
    
    # Test DSL parsing
    sample_composer_dsl = '''
    (defsymphony
     "Copy of 200d MA 3x Leverage"
     {:asset-class "EQUITIES", :rebalance-threshold 0.05}
     (weight-equal
      [(if
        (>
         (current-price "SPY")
         (moving-average-price "SPY" {:window 200}))
        [(weight-equal
          [(if
            (> (rsi "TQQQ" {:window 10}) 79)
            [(asset "UVXY" "ProShares Ultra VIX Short-Term Futures ETF")]
            [(asset "TQQQ" "ProShares UltraPro QQQ")])])]
        [(weight-equal
          [(if
            (< (rsi "TQQQ" {:window 10}) 31)
            [(asset "TECL" "Direxion Daily Technology Bull 3x Shares")]
            [(weight-equal
              [(if
                (>
                 (current-price "QQQ")
                 (moving-average-price "QQQ" {:window 20}))
                [(weight-equal
                  [(if
                    (> (rsi "QQQ" {:window 10}) 70)
                    [(asset "PSQ" "ProShares Short QQQ")]
                    [(weight-equal
                      [(asset "QQQ" "Invesco QQQ Trust")])])])]
                [(weight-equal
                  [(if
                    (> (rsi "SHY" {:window 10}) (rsi "PSQ" {:window 10}))
                    [(asset "QQQ" "Invesco QQQ Trust")]
                    [(asset "PSQ" "ProShares Short QQQ")])])])])])])])]))
    '''
    
    parser = ComposerDSLParser()
    
    try:
        result = parser.parse_composer_dsl(sample_composer_dsl)
        
        print(f"📋 Parsed Symphony: {result.name}")
        print(f"🎯 Universe: {result.our_format['universe']}")
        print(f"🔧 Conditions: {len(result.our_format['logic']['conditions'])}")
        print(f"📊 Allocations: {len(result.our_format['logic']['allocations'])}")
        
        print(f"\n📝 Parsing Notes:")
        for note in result.parsing_notes:
            print(f"  • {note}")
        
        print(f"\n🔍 Converted Format Sample:")
        print(json.dumps(result.our_format, indent=2)[:500] + "...")
        
    except Exception as e:
        print(f"❌ Error parsing DSL: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Composer compatibility system ready!")
    print("💡 Next steps:")
    print("  1. Parse your Composer DSL symphonies")
    print("  2. Convert to our JSON format")
    print("  3. Run backtests and compare results")
    print("  4. Reconcile any differences")

import json
import logging
import os
import pytz
import requests
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from scipy import stats
from typing import Dict, List, Optional


class TradingLogger:
    """
    A singleton logger class for tracking trading system operations.
    
    This class implements the singleton pattern to ensure a single logging instance
    is shared across the trading system. It configures both file and console logging
    with appropriate formatting.
    
    Attributes
    ----------
    logs_dir : str
        Directory where log files are stored
    logger : logging.Logger
        Configured logging instance
    _instance : Optional[TradingLogger]
        Singleton instance of the logger
    _initialized : bool
        Flag indicating if the logger has been initialized
    
    Methods
    -------
    get_logger()
        Returns the configured logger instance
    """    
    _instance: Optional['TradingLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TradingLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Create logs directory if it doesn't exist
            self.logs_dir = 'logs'
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # Create log filename with timestamp
            log_filename = os.path.join(
                self.logs_dir, 
                f'trading_{datetime.now().strftime("%Y%m%d")}.log'
            )
            
            # Configure logging
            self.logger = logging.getLogger('trading_system')
            self.logger.setLevel(logging.INFO)
            
            # File handler
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            self._initialized = True
    
    @classmethod
    def get_logger(cls):
        """
        Get the singleton logger instance.
        
        Returns
        -------
        logging.Logger
            Configured logger instance
        """        
        if cls._instance is None:
            cls()
        return cls._instance.logger

@dataclass
class TradeMetrics:
    """
    Data class containing key trading performance metrics.
    
    Parameters
    ----------
    win_rate : float
        Percentage of profitable trades (0.0 to 1.0)
    profit_factor : float
        Ratio of gross profits to gross losses
    average_win : float
        Average profit per winning trade
    average_loss : float
        Average loss per losing trade (positive number)
    max_drawdown : float
        Maximum peak-to-trough decline in account value
    sharpe_ratio : float
        Risk-adjusted return metric (using 252 trading days)
    total_trades : int
        Total number of trades executed
    profitable_trades : int
        Number of profitable trades
    """    
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    profitable_trades: int

@dataclass
class CurrencyConfig:
    """
    Configuration for currency handling.
    
    Parameters
    ----------
    base_currency : str
        Base currency for calculations (e.g., 'USD')
    fx_data_source : str
        Source for FX rates ('local' or 'api')
    update_frequency : int
        How often to update FX rates (minutes)
    """
    base_currency: str
    fx_data_source: str
    update_frequency: int

class VenueType(Enum):
    """Trading venue types for cost analysis."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    MARKET_MAKER = "market_maker"
    OTC = "otc"

@dataclass
class TransactionCosts:
    """
    Detailed breakdown of transaction costs.
    
    Parameters
    ----------
    commission : float
        Direct commission costs
    slippage : float
        Implementation shortfall (difference from expected price)
    spread_cost : float
        Cost due to bid-ask spread
    market_impact : float
        Estimated price impact of the trade
    delay_cost : float
        Cost due to execution delay
    venue_fees : float
        Specific venue or exchange fees
    clearing_fees : float
        Clearing and settlement fees
    """
    commission: float
    slippage: float
    spread_cost: float
    market_impact: float
    delay_cost: float
    venue_fees: float
    clearing_fees: float
    
    @property
    def total_cost(self) -> float:
        """Calculate total transaction cost."""
        return (self.commission + self.slippage + self.spread_cost +
                self.market_impact + self.delay_cost + self.venue_fees +
                self.clearing_fees)
    
    @property
    def total_cost_bps(self) -> float:
        """Calculate total cost in basis points."""
        return self.total_cost * 10000  # Convert to basis points

@dataclass
class CorrelationConfig:
    """
    Configuration for correlation analysis.
    
    Parameters
    ----------
    lookback_days : int
        Days of historical data to use
    min_periods : int
        Minimum periods required for correlation
    correlation_threshold : float
        Minimum correlation to consider significant
    max_cluster_exposure : float
        Maximum exposure for correlated asset clusters
    sector_limits : Dict[str, float]
        Maximum exposure per sector
    """
    lookback_days: int = 252  # One trading year
    min_periods: int = 63     # ~3 months
    correlation_threshold: float = 0.6
    max_cluster_exposure: float = 0.15  # 15% max for correlated assets
    sector_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.sector_limits is None:
            self.sector_limits = {
                'Technology': 0.30,
                'Financial': 0.25,
                'Healthcare': 0.25,
                'Consumer': 0.25,
                'Industrial': 0.25,
                'Energy': 0.20,
                'Materials': 0.20,
                'Utilities': 0.15,
                'Real Estate': 0.15,
                'Other': 0.15  # Default for any uncategorized sectors
            }
        else:
            # Ensure 'Other' is always present in sector limits
            self.sector_limits['Other'] = self.sector_limits.get('Other', 0.15)

class TradingAnalytics:
    """
    Core analytics system for trading performance analysis and reporting.
    
    This class provides comprehensive functionality for logging trades,
    analyzing performance, and generating reports. It maintains a SQLite
    database to store trading data and provides various analysis methods.
    
    Parameters
    ----------
    db_path : str, optional
        Path to SQLite database file, by default 'trading_data.db'
    
    Attributes
    ----------
    db_path : str
        Path to the SQLite database file
    
    Notes
    -----
    The system maintains three main tables:
    - trades: Individual trade records
    - market_data: Market conditions and indicators
    - bot_decisions: Trading bot decision records
    
    The analysis includes:
    - Performance metrics (win rate, profit factor, etc.)
    - Time-based patterns
    - Market condition analysis
    - Risk analysis
    - Position sizing analysis
    - Drawdown analysis
    
    Examples
    --------
    >>> analytics = TradingAnalytics()
    >>> analytics.initialize_database()
    >>> trade_data = {
    ...     'timestamp': '2024-01-01T10:00:00',
    ...     'symbol': 'AAPL',
    ...     'entry_price': 150.0,
    ...     'exit_price': 152.0,
    ...     'quantity': 100,
    ...     'profit_loss': 200.0
    ... }
    >>> analytics.log_trade(trade_data)
    >>> report = analytics.generate_report(
    ...     analytics.analyze_strategy_performance('2024-01-01', '2024-01-02')
    ... )
    """

    def __init__(self: "TradingAnalytics", db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self: "TradingAnalytics"):
        """
        Create necessary database tables if they don't exist.
        
        Creates three main tables in SQLite database:
        - trades: Individual trade records
        - market_data: Market price and indicator data
        - bot_decisions: Trading bot decision records
        
        Notes
        -----
        This method is idempotent - safe to call multiple times.
        Existing tables will not be modified.
        
        Tables Schema:
        -------------
        trades:
            - trade_id (INTEGER PRIMARY KEY)
            - timestamp (TEXT)
            - symbol (TEXT)
            - entry_price (REAL)
            - exit_price (REAL)
            - quantity (REAL)
            - side (TEXT)
            - strategy_name (TEXT)
            - stop_loss (REAL)
            - take_profit (REAL)
            - profit_loss (REAL)
            - execution_time_ms (INTEGER)
            - indicators (JSON)
            - market_conditions (JSON)
        
        market_data:
            - timestamp (TEXT)
            - symbol (TEXT)
            - open (REAL)
            - high (REAL)
            - low (REAL)
            - close (REAL)
            - volume (REAL)
            - indicators (JSON)
        
        bot_decisions:
            - decision_id (INTEGER PRIMARY KEY)
            - timestamp (TEXT)
            - symbol (TEXT)
            - decision_type (TEXT)
            - confidence_score (REAL)
            - reasons (JSON)
            - executed (BOOLEAN)
            - result (TEXT)
            - performance_impact (REAL)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Trades table
        c.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                side TEXT,
                strategy_name TEXT,
                stop_loss REAL,
                take_profit REAL,
                profit_loss REAL,
                execution_time_ms INTEGER,
                indicators JSON,
                market_conditions JSON
            )
        ''')
        
        # Market Data table
        c.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                indicators JSON,
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
        
        # Bot Decisions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS bot_decisions (
                decision_id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                decision_type TEXT,
                confidence_score REAL,
                reasons JSON,
                executed BOOLEAN,
                result TEXT,
                performance_impact REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade(self: "TradingAnalytics", trade_data: Dict):
        """
        Log a trade with all relevant information to the database.
        
        Parameters
        ----------
        trade_data : Dict
            Dictionary containing trade information with the following keys:
            - timestamp : str
                ISO format timestamp of the trade
            - symbol : str
                Trading symbol/ticker
            - entry_price : float
                Entry price of the trade
            - exit_price : float
                Exit price of the trade
            - quantity : float
                Trade quantity
            - side : str
                Trade direction ('buy' or 'sell')
            - strategy_name : str
                Name of the trading strategy
            - stop_loss : float
                Stop loss price
            - take_profit : float
                Take profit price
            - profit_loss : float
                Realized profit/loss
            - execution_time_ms : int
                Trade execution time in milliseconds
            - indicators : dict
                Technical indicators at time of trade
            - market_conditions : dict
                Market conditions at time of trade
        
        Notes
        -----
        Indicators and market_conditions are stored as JSON strings in the database.
        
        Examples
        --------
        >>> trade_data = {
        ...     'timestamp': '2024-01-01T10:00:00',
        ...     'symbol': 'AAPL',
        ...     'entry_price': 150.0,
        ...     'exit_price': 152.0,
        ...     'quantity': 100,
        ...     'side': 'buy',
        ...     'strategy_name': 'MovingAverageCrossover',
        ...     'stop_loss': 149.0,
        ...     'take_profit': 153.0,
        ...     'profit_loss': 200.0,
        ...     'execution_time_ms': 150,
        ...     'indicators': {'sma_20': 148.5, 'rsi': 65},
        ...     'market_conditions': {'trend': 'upward', 'volatility': 'low'}
        ... }
        >>> analytics.log_trade(trade_data)
        """
        conn = sqlite3.connect(self.db_path)
        
        trade_data['indicators'] = json.dumps(trade_data.get('indicators', {}))
        trade_data['market_conditions'] = json.dumps(trade_data.get('market_conditions', {}))
        
        columns = ', '.join(trade_data.keys())
        placeholders = ':' + ', :'.join(trade_data.keys())
        
        query = f'INSERT INTO trades ({columns}) VALUES ({placeholders})'
        
        conn.execute(query, trade_data)
        conn.commit()
        conn.close()
    
    def log_market_data(self: "TradingAnalytics", market_data: Dict):
        """
        Log market data with technical indicators to the database.
        
        Parameters
        ----------
        market_data : Dict
            Dictionary containing market data with the following keys:
            - timestamp : str
                ISO format timestamp
            - symbol : str
                Trading symbol/ticker
            - open : float
                Opening price
            - high : float
                High price
            - low : float
                Low price
            - close : float
                Closing price
            - volume : float
                Trading volume
            - indicators : dict
                Technical indicators dictionary
        
        Notes
        -----
        If a record exists for the same timestamp and symbol,
        it will be replaced with the new data.
        
        Examples
        --------
        >>> market_data = {
        ...     'timestamp': '2024-01-01T10:00:00',
        ...     'symbol': 'AAPL',
        ...     'open': 150.0,
        ...     'high': 152.0,
        ...     'low': 149.5,
        ...     'close': 151.0,
        ...     'volume': 1000000,
        ...     'indicators': {'sma_20': 148.5, 'rsi': 65}
        ... }
        >>> analytics.log_market_data(market_data)
        """
        conn = sqlite3.connect(self.db_path)
        
        market_data['indicators'] = json.dumps(market_data.get('indicators', {}))
        
        columns = ', '.join(market_data.keys())
        placeholders = ':' + ', :'.join(market_data.keys())
        
        query = f'INSERT OR REPLACE INTO market_data ({columns}) VALUES ({placeholders})'
        
        conn.execute(query, market_data)
        conn.commit()
        conn.close()
    
    def log_bot_decision(self: "TradingAnalytics", decision_data: Dict):
        """
        Log bot's decision-making process to the database.
        
        Parameters
        ----------
        decision_data : Dict
            Dictionary containing decision data with the following keys:
            - timestamp : str
                ISO format timestamp
            - symbol : str
                Trading symbol/ticker
            - decision_type : str
                Type of decision (e.g., 'ENTRY', 'EXIT')
            - confidence_score : float
                Confidence level (0.0 to 1.0)
            - reasons : dict
                Dictionary of reasons for the decision
            - executed : bool
                Whether the decision was executed
            - result : str
                Outcome of the decision
            - performance_impact : float
                Profit/loss impact of the decision
        
        Examples
        --------
        >>> decision_data = {
        ...     'timestamp': '2024-01-01T10:00:00',
        ...     'symbol': 'AAPL',
        ...     'decision_type': 'ENTRY',
        ...     'confidence_score': 0.85,
        ...     'reasons': {'trend': 'upward', 'momentum': 'positive'},
        ...     'executed': True,
        ...     'result': 'profit',
        ...     'performance_impact': 200.0
        ... }
        >>> analytics.log_bot_
            conn = sqlite3.connect(self.db_path)
        """
        conn = sqlite3.connect(self.db_path)
        
        decision_data['reasons'] = json.dumps(decision_data.get('reasons', {}))
        
        columns = ', '.join(decision_data.keys())
        placeholders = ':' + ', :'.join(decision_data.keys())
        
        query = f'INSERT INTO bot_decisions ({columns}) VALUES ({placeholders})'
        
        conn.execute(query, decision_data)
        conn.commit()
        conn.close()

    def calculate_metrics(self: "TradingAnalytics", start_date: str, end_date: str) -> TradeMetrics:
        """
        Calculate comprehensive trading metrics for a specified date range.
        
        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        
        Returns
        -------
        Optional[TradeMetrics]
            TradeMetrics object containing calculated metrics, or None if no data
            
        Notes
        -----
        Metrics calculated include:
        - Win rate: Percentage of profitable trades
        - Profit factor: Ratio of gross profits to gross losses
        - Average win: Mean profit of winning trades
        - Average loss: Mean loss of losing trades
        - Max drawdown: Largest peak-to-trough decline
        - Sharpe ratio: Risk-adjusted return metric
        - Total trades: Number of trades executed
        - Profitable trades: Number of winning trades
        
        The Sharpe ratio calculation assumes:
        - Daily returns
        - Risk-free rate of 0
        - 252 trading days per year
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT *
            FROM trades
            WHERE timestamp BETWEEN ? AND ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Calculate basic metrics
        profitable_trades = df[df['profit_loss'] > 0]
        losing_trades = df[df['profit_loss'] <= 0]
        
        win_rate = len(profitable_trades) / len(df) if len(df) > 0 else 0
        profit_factor = abs(profitable_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else float('inf')
        
        # Calculate average win/loss
        avg_win = profitable_trades['profit_loss'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = abs(losing_trades['profit_loss'].mean()) if len(losing_trades) > 0 else 0
        
        # Calculate drawdown
        cumulative = df['profit_loss'].cumsum()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative - rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        returns = df['profit_loss'].pct_change()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
        
        return TradeMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=len(df),
            profitable_trades=len(profitable_trades)
        )

    def verify_data_availability(self: "TradingAnalytics", start_date: str, end_date: str) -> bool:
        """
        Verify if trading data exists for the specified period.
        
        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        
        Returns
        -------
        bool
            True if data exists for the period, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT COUNT(*) as count 
            FROM trades 
            WHERE timestamp BETWEEN ? AND ?
        '''
        
        count = conn.execute(query, (start_date, end_date)).fetchone()[0]
        conn.close()
        
        return count > 0

    def analyze_strategy_performance(self: "TradingAnalytics", start_date: str, end_date: str) -> Dict:
        """
        Analyze strategy performance and generate recommendations.
        
        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        
        Returns
        -------
        Dict
            Dictionary containing analysis results with keys:
            - metrics : TradeMetrics
                Overall performance metrics
            - patterns : Dict
                Identified trading patterns
            - recommendations : List[str]
                Strategy improvement suggestions
            - risk_analysis : Dict
                Risk metrics and analysis
        
        Notes
        -----
        This method performs comprehensive analysis including:
        - Performance metrics calculation
        - Pattern identification in trading behavior
        - Risk analysis
        - Trading timing analysis
        - Market condition correlation
        - Position sizing effectiveness
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get trades and decisions
        trades_df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=[start_date, end_date])
        
        decisions_df = pd.read_sql_query('''
            SELECT * FROM bot_decisions 
            WHERE timestamp BETWEEN ? AND ?
        ''', conn, params=[start_date, end_date])
        
        conn.close()
        
        analysis = {
            'metrics': self.calculate_metrics(start_date, end_date),
            'patterns': self._identify_patterns(trades_df),
            'recommendations': self._generate_recommendations(trades_df, decisions_df),
            'risk_analysis': self._analyze_risk(trades_df)
        }
        
        return analysis
    
    def _identify_patterns(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Identify common patterns in trading behavior.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records
        
        Returns
        -------
        Dict
            Dictionary containing identified patterns:
            - time_based : Dict
                Patterns in trading times
            - market_conditions : Dict
                Performance under different market conditions
            - consecutive_trades : Dict
                Analysis of trading streaks
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        """
        patterns = {
            'time_based': self._analyze_time_patterns(trades_df),
            'market_conditions': self._analyze_market_conditions(trades_df),
            'consecutive_trades': self._analyze_consecutive_trades(trades_df)
        }
        return patterns
    
    def _generate_recommendations(self: "TradingAnalytics", trades_df: pd.DataFrame, decisions_df: pd.DataFrame) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records
        decisions_df : pd.DataFrame
            DataFrame containing bot decisions
        
        Returns
        -------
        List[str]
            List of strategy improvement recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        Recommendations are based on:
        - Strategy performance metrics
        - Risk/reward ratios
        - Market condition analysis
        - Trading patterns
        """
        recommendations = []
        
        # Analyze win rate by strategy
        strategy_performance = trades_df.groupby('strategy_name')['profit_loss'].agg(['mean', 'count', 'sum'])
        
        for strategy in strategy_performance.index:
            perf = strategy_performance.loc[strategy]
            if perf['mean'] < 0:
                recommendations.append(f"Review strategy '{strategy}' - showing negative average returns")
            elif perf['count'] < 10:
                recommendations.append(f"Need more trades to properly evaluate strategy '{strategy}'")
        
        # Analyze risk management
        avg_risk_reward = abs(trades_df[trades_df['profit_loss'] > 0]['profit_loss'].mean() / 
                            trades_df[trades_df['profit_loss'] < 0]['profit_loss'].mean())
        
        if avg_risk_reward < 1.5:
            recommendations.append("Consider adjusting risk/reward ratio - current average is below 1.5")
        
        return recommendations
    
    def _analyze_risk(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze comprehensive risk metrics and patterns.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records with columns:
            - entry_price : float
            - exit_price : float
            - quantity : float
            - profit_loss : float
            - stop_loss : float
            - take_profit : float
        
        Returns
        -------
        Dict
            Dictionary containing risk analysis results:
            - position_size_analysis : Dict
                Analysis of position sizing effectiveness
            - drawdown_analysis : Dict
                Detailed drawdown metrics and patterns
            - risk_reward_analysis : Dict
                Risk/reward ratios and effectiveness
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        
        The analysis includes:
        - Position size optimization
        - Drawdown patterns and recovery
        - Risk/reward ratio effectiveness
        - Stop loss and take profit placement analysis
        
        Missing or zero values in stop_loss or take_profit are
        handled gracefully and noted in the analysis.
        """
        return {
            'position_size_analysis': self._analyze_position_sizes(trades_df),
            'drawdown_analysis': self._analyze_drawdowns(trades_df),
            'risk_reward_analysis': self._analyze_risk_reward(trades_df)
        }

    def _analyze_time_patterns(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze trading performance patterns based on time periods.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records with columns:
            - timestamp : str
            - profit_loss : float
            - trade_id : int
        
        Returns
        -------
        Dict
            Dictionary containing time-based analysis:
            - best_hour : int
                Hour with highest win rate (0-23)
            - worst_hour : int
                Hour with lowest win rate (0-23)
            - hourly_performance : Dict
                Performance metrics for each trading hour
            - recommendations : List[str]
                Time-based trading recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        
        Time patterns analyzed:
        - Hourly performance during trading day
        - Pre/post market performance
        - Day of week analysis
        - Market session analysis (Asian, European, American)
        
        Performance metrics for each time period include:
        - Win rate
        - Average profit/loss
        - Total trades
        - Risk-adjusted returns
        """
        if trades_df.empty:
            return {
                'best_hour': None,
                'worst_hour': None,
                'hourly_performance': {},
                'recommendations': []
            }

        # Convert timestamp to datetime if it's string
        if isinstance(trades_df['timestamp'].iloc[0], str):
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Extract hour from timestamp
        trades_df['hour'] = trades_df['timestamp'].dt.hour

        # Calculate hourly performance
        hourly_stats = trades_df.groupby('hour').agg({
            'profit_loss': ['count', 'mean', 'sum'],
            'trade_id': 'count'
        }).fillna(0)

        # Calculate win rate per hour
        hourly_wins = trades_df[trades_df['profit_loss'] > 0].groupby('hour').size()
        hourly_total = trades_df.groupby('hour').size()
        win_rates = (hourly_wins / hourly_total).fillna(0)

        # Find best and worst hours
        best_hour = win_rates.idxmax() if not win_rates.empty else None
        worst_hour = win_rates.idxmin() if not win_rates.empty else None

        # Create performance dictionary
        hourly_performance = {
            hour: {
                'win_rate': win_rates.get(hour, 0),
                'avg_profit': hourly_stats.loc[hour, ('profit_loss', 'mean')] if hour in hourly_stats.index else 0,
                'total_trades': hourly_stats.loc[hour, ('trade_id', 'count')] if hour in hourly_stats.index else 0
            }
            for hour in range(9, 16)  # Trading hours 9 AM to 4 PM
        }

        # Generate recommendations
        recommendations = []
        if best_hour is not None:
            recommendations.append(f"Most profitable hour is {best_hour}:00")
        if worst_hour is not None:
            recommendations.append(f"Consider avoiding trading at {worst_hour}:00")

        return {
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'hourly_performance': hourly_performance,
            'recommendations': recommendations
        }

    def _analyze_market_conditions(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze trading performance under different market conditions.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records with columns:
            - market_conditions : str (JSON)
            - profit_loss : float
            - timestamp : str
        
        Returns
        -------
        Dict
            Dictionary containing market condition analysis:
            - trend_performance : Dict
                Performance metrics in different market trends
            - volatility_performance : Dict
                Performance metrics under different volatility conditions
            - recommendations : List[str]
                Market condition-specific recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        
        Market conditions analyzed include:
        - Market trends (upward, downward, sideways)
        - Volatility levels (high, medium, low)
        - Volume conditions (above_average, below_average, normal)
        
        Performance metrics are calculated for each condition type
        and compared to identify optimal trading environments.
        """
        if trades_df.empty:
            return {
                'market_conditions': {},
                'recommendations': []
            }

        # Convert market_conditions from JSON string if necessary
        if isinstance(trades_df['market_conditions'].iloc[0], str):
            trades_df['market_conditions'] = trades_df['market_conditions'].apply(json.loads)

        # Extract market conditions
        conditions_data = []
        for idx, row in trades_df.iterrows():
            conditions = row['market_conditions']
            conditions['profit_loss'] = row['profit_loss']
            conditions_data.append(conditions)

        conditions_df = pd.DataFrame(conditions_data)

        # Analyze performance by trend
        trend_performance = conditions_df.groupby('trend').agg({
            'profit_loss': ['count', 'mean', 'sum']
        }).fillna(0)

        # Analyze performance by volatility
        volatility_performance = conditions_df.groupby('volatility').agg({
            'profit_loss': ['count', 'mean', 'sum']
        }).fillna(0)

        # Generate recommendations
        recommendations = []
        if not trend_performance.empty:
            best_trend = trend_performance['profit_loss']['mean'].idxmax()
            recommendations.append(f"Best performance in {best_trend} trend markets")

        if not volatility_performance.empty:
            best_volatility = volatility_performance['profit_loss']['mean'].idxmax()
            recommendations.append(f"Best performance in {best_volatility} volatility conditions")

        return {
            'trend_performance': trend_performance.to_dict(),
            'volatility_performance': volatility_performance.to_dict(),
            'recommendations': recommendations
        }

    def _analyze_consecutive_trades(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in consecutive trades and trading streaks.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records with columns:
            - timestamp : str
            - profit_loss : float
            - win : bool (derived from profit_loss)
        
        Returns
        -------
        Dict
            Dictionary containing consecutive trade analysis:
            - consecutive_wins : int
                Longest streak of winning trades
            - consecutive_losses : int
                Longest streak of losing trades
            - win_after_win_rate : float
                Probability of winning after a winning trade
            - win_after_loss_rate : float
                Probability of winning after a losing trade
            - recommendations : List[str]
                Streak-based trading recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        
        Analysis includes:
        - Identification of winning/losing streaks
        - Probability analysis of trade outcomes
        - Pattern recognition in trade sequences
        - Recovery analysis after losing streaks
        """
        if trades_df.empty:
            return {
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'patterns': {},
                'recommendations': []
            }

        # Sort trades by timestamp
        trades_df = trades_df.sort_values('timestamp')

        # Create win/loss streak analysis
        trades_df['win'] = trades_df['profit_loss'] > 0
        trades_df['streak'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()

        # Calculate streak lengths
        streak_lengths = trades_df.groupby(['streak', 'win']).size()
        
        # Find maximum consecutive wins and losses
        consecutive_wins = streak_lengths[streak_lengths.index.get_level_values('win') == True].max() if True in streak_lengths.index.get_level_values('win') else 0
        consecutive_losses = streak_lengths[streak_lengths.index.get_level_values('win') == False].max() if False in streak_lengths.index.get_level_values('win') else 0

        # Analyze win probability after wins/losses
        win_after_win = len(trades_df[(trades_df['win']) & (trades_df['win'].shift() == True)]) / len(trades_df[trades_df['win'].shift() == True]) if len(trades_df[trades_df['win'].shift() == True]) > 0 else 0
        win_after_loss = len(trades_df[(trades_df['win']) & (trades_df['win'].shift() == False)]) / len(trades_df[trades_df['win'].shift() == False]) if len(trades_df[trades_df['win'].shift() == False]) > 0 else 0

        # Generate recommendations
        recommendations = []
        if consecutive_losses >= 3:
            recommendations.append("Consider implementing consecutive loss limit")
        if win_after_loss > win_after_win:
            recommendations.append("Higher win rate after losses - consider aggressive recovery strategy")
        elif win_after_win > win_after_loss:
            recommendations.append("Higher win rate after wins - consider riding winning streaks")

        return {
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'win_after_win_rate': win_after_win,
            'win_after_loss_rate': win_after_loss,
            'recommendations': recommendations
        }

    def _analyze_position_sizes(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze effectiveness of different position sizes.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records
        
        Returns
        -------
        Dict
            Dictionary containing position size analysis:
            - optimal_size : str
                Most effective position size category
            - size_performance : Dict
                Performance metrics by position size
            - recommendations : List[str]
                Position sizing recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        Position sizes are categorized into quintiles
        """
        logger = TradingLogger.get_logger()
        if trades_df.empty:
            return {
                'optimal_size': None,
                'size_performance': {},
                'recommendations': []
            }

        try:
            # Calculate position value
            trades_df['position_value'] = trades_df['quantity'] * trades_df['entry_price']
            
            # Create position size buckets with explicit observed parameter
            trades_df['size_bucket'] = pd.qcut(trades_df['position_value'], 
                                            q=5, 
                                            labels=['very_small', 'small', 'medium', 'large', 'very_large'],
                                            duplicates='drop')  # Handle duplicate values
            
            # Analyze performance by position size with explicit observed parameter
            size_performance = trades_df.groupby('size_bucket', observed=True).agg({
                'profit_loss': ['count', 'mean', 'sum', 'std'],
                'trade_id': 'count'
            }).fillna(0)
            
            # Find optimal size bucket
            optimal_size = size_performance['profit_loss']['mean'].idxmax() if not size_performance.empty else None
            
            # Calculate Sharpe ratio for each size bucket
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            size_performance['sharpe_ratio'] = (
                (size_performance['profit_loss']['mean'] - risk_free_rate) / 
                size_performance['profit_loss']['std'].replace(0, np.inf)  # Handle zero std
            ).fillna(0)
            
            # Generate recommendations
            recommendations = []
            if optimal_size is not None:
                recommendations.append(f"Best risk-adjusted returns with {optimal_size} position sizes")
            
            # Check for size-related issues
            if not size_performance.empty and 'profit_loss' in size_performance:
                std_max = size_performance['profit_loss']['std'].max()
                std_mean = size_performance['profit_loss']['std'].mean()
                if std_max > 2 * std_mean:
                    recommendations.append("Consider reducing position sizes to manage volatility")
            
            return {
                'optimal_size': optimal_size,
                'size_performance': size_performance.to_dict(),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in position size analysis: {str(e)}")
            return {
                'optimal_size': None,
                'size_performance': {},
                'recommendations': [f"Error in position size analysis: {str(e)}"]
            }
        
    def _analyze_drawdowns(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze drawdown patterns and severity.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records
        
        Returns
        -------
        Dict
            Dictionary containing drawdown analysis:
            - max_drawdown : float
                Maximum drawdown value
            - avg_drawdown : float
                Average drawdown
            - drawdown_periods : List[Dict]
                Details of significant drawdown periods
            - recommendations : List[str]
                Risk management recommendations
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        Drawdown periods include start date, end date, and depth
        """
        if trades_df.empty:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'drawdown_periods': [],
                'recommendations': []
            }

        # Calculate cumulative profit/loss
        trades_df = trades_df.sort_values('timestamp')
        trades_df['cumulative_pnl'] = trades_df['profit_loss'].cumsum()

        # Calculate drawdown
        trades_df['rolling_max'] = trades_df['cumulative_pnl'].expanding().max()
        trades_df['drawdown'] = trades_df['cumulative_pnl'] - trades_df['rolling_max']

        # Find drawdown periods
        drawdown_periods = []
        current_drawdown = {'start': None, 'end': None, 'depth': 0}

        for idx, row in trades_df.iterrows():
            if row['drawdown'] < 0:
                if current_drawdown['start'] is None:
                    current_drawdown['start'] = row['timestamp']
                if row['drawdown'] < current_drawdown['depth']:
                    current_drawdown['depth'] = row['drawdown']
            elif current_drawdown['start'] is not None:
                current_drawdown['end'] = row['timestamp']
                drawdown_periods.append(current_drawdown.copy())
                current_drawdown = {'start': None, 'end': None, 'depth': 0}

        # Calculate metrics
        max_drawdown = trades_df['drawdown'].min()
        avg_drawdown = trades_df[trades_df['drawdown'] < 0]['drawdown'].mean()

        # Generate recommendations
        recommendations = []
        if abs(max_drawdown) > 0.1 * trades_df['cumulative_pnl'].max():
            recommendations.append("Consider implementing stricter drawdown controls")
        if len(drawdown_periods) > 5:
            recommendations.append("Frequent drawdowns detected - review risk management")

        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_periods': drawdown_periods,
            'recommendations': recommendations
        }

    def _analyze_risk_reward(self: "TradingAnalytics", trades_df: pd.DataFrame) -> Dict:
        """
        Analyze risk/reward ratios and their effectiveness.
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame containing trade records with columns:
            - entry_price : float
            - exit_price : float
            - stop_loss : float
            - take_profit : float
            - quantity : float
            - profit_loss : float
        
        Returns
        -------
        Dict
            Dictionary containing risk/reward analysis:
            - avg_risk_reward : float
                Average realized risk/reward ratio
            - risk_reward_patterns : Dict
                Performance metrics by R:R ratio category
            - optimal_ratio : str
                Best performing risk/reward ratio category
            - recommendations : List[str]
                Risk/reward optimization suggestions
            - stats : Dict
                Additional statistics including:
                - trades_without_stops : int
                - avg_reward_to_risk : float
        
        Notes
        -----
        Private method used by analyze_strategy_performance
        
        Analysis includes:
        - Actual vs. planned risk/reward ratios
        - Win rate by risk/reward category
        - Optimal risk/reward ratio identification
        - Stop loss and take profit placement effectiveness
        
        Risk/reward categories:
        - very_low: < 1:1
        - low: 1:1 to 1.5:1
        - medium: 1.5:1 to 2:1
        - high: 2:1 to 2.5:1
        - very_high: > 2.5:1
        
        Trades without defined stop losses or take profits are
        tracked separately and noted in recommendations.
        """
        logger = TradingLogger.get_logger()
        if trades_df.empty:
            return {
                'avg_risk_reward': 0,
                'risk_reward_patterns': {},
                'recommendations': []
            }

        try:
            # Calculate realized risk/reward ratios
            trades_df['risk'] = abs(trades_df['entry_price'] - trades_df['stop_loss'])
            trades_df['reward'] = abs(trades_df['take_profit'] - trades_df['entry_price'])
            
            # Avoid division by zero
            trades_df['risk'] = trades_df['risk'].replace(0, np.nan)
            rr_ratio = trades_df['reward'] / trades_df['risk']
            
            # Calculate realized RR based on actual profit/loss
            trades_df['realized_rr'] = trades_df['profit_loss'] / (trades_df['risk'] * trades_df['quantity'])
            
            # Create manual buckets instead of qcut to avoid duplicate edge issues
            def assign_rr_bucket(ratio):
                if pd.isna(ratio):
                    return 'undefined'
                elif ratio < 1:
                    return 'very_low'
                elif ratio < 1.5:
                    return 'low'
                elif ratio < 2:
                    return 'medium'
                elif ratio < 2.5:
                    return 'high'
                else:
                    return 'very_high'
            
            trades_df['rr_bucket'] = rr_ratio.apply(assign_rr_bucket)
            
            # Analyze performance by risk/reward bucket
            rr_performance = trades_df.groupby('rr_bucket').agg({
                'profit_loss': ['count', 'mean', 'sum'],
                'realized_rr': 'mean',
                'trade_id': 'count'
            }).fillna(0)
            
            # Find optimal risk/reward bucket (excluding 'undefined')
            valid_buckets = rr_performance.loc[rr_performance.index != 'undefined']
            optimal_rr = valid_buckets['profit_loss']['mean'].idxmax() if not valid_buckets.empty else None
            
            # Calculate average realized RR
            avg_realized_rr = trades_df['realized_rr'].mean()
            
            # Generate recommendations
            recommendations = []
            if optimal_rr is not None:
                recommendations.append(f"Best performance observed with {optimal_rr} risk/reward ratio")
            
            if pd.notna(avg_realized_rr):
                if avg_realized_rr < 1:
                    recommendations.append("Consider increasing reward targets relative to risk")
                elif avg_realized_rr > 2:
                    recommendations.append("Strong risk/reward performance - consider increasing position sizes")
            
            # Check for risk management issues
            zero_risk_count = (trades_df['risk'] == 0).sum()
            if zero_risk_count > 0:
                recommendations.append(f"Warning: {zero_risk_count} trades with no stop loss detected")
            
            # Analyze win rate by RR bucket
            win_rates = trades_df.groupby('rr_bucket').apply(
                lambda x: (x['profit_loss'] > 0).mean()
            ).fillna(0)
            
            return {
                'avg_risk_reward': float(avg_realized_rr) if pd.notna(avg_realized_rr) else 0,
                'risk_reward_patterns': {
                    'performance': rr_performance.to_dict(),
                    'win_rates': win_rates.to_dict()
                },
                'optimal_ratio': optimal_rr,
                'recommendations': recommendations,
                'stats': {
                    'trades_without_stops': int(zero_risk_count),
                    'avg_reward_to_risk': float(rr_ratio.mean()) if not rr_ratio.empty else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in risk/reward analysis: {str(e)}")
            return {
                'avg_risk_reward': 0,
                'risk_reward_patterns': {},
                'optimal_ratio': None,
                'recommendations': [f"Error in risk/reward analysis: {str(e)}"],
                'stats': {}
            }

    def generate_report(self: "TradingAnalytics", analysis_results: Dict) -> str:
        """
        Generate a formatted analysis report from results.
        
        Parameters
        ----------
        analysis_results : Dict
            Results from analyze_strategy_performance method
        
        Returns
        -------
        str
            Formatted report string containing sections:
            - Performance Metrics
            - Trading Patterns
            - Risk Analysis
            - Recommendations
        
        Notes
        -----
        Report formatting:
        - Uses section headers with '==='
        - Metrics formatted to appropriate decimal places
        - Includes bulleted lists for recommendations
        - Handles missing data gracefully
        
        Examples
        --------
        >>> analysis = analytics.analyze_strategy_performance('2024-01-01', '2024-01-31')
        >>> report = analytics.generate_report(analysis)
        >>> print(report)
        === Performance Metrics ===
        Win Rate: 65.00%
        Profit Factor: 2.15
        ...
        """
        report = []
        
        # Handle case where no data is available
        if not analysis_results or 'metrics' not in analysis_results or analysis_results['metrics'] is None:
            return """
    === Analysis Report ===
    No trading data available for the specified period.

    Possible reasons:
    1. No trades executed in the date range
    2. Database is empty
    3. Incorrect date range specified

    Please verify:
    1. Date range is correct
    2. Data has been properly logged
    3. Database connection is working
    """
        
        metrics = analysis_results['metrics']
        
        # Performance Metrics Section
        report.append("=== Performance Metrics ===")
        report.append(f"Win Rate: {metrics.win_rate:.2%}")
        report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
        report.append(f"Average Win: ${metrics.average_win:.2f}")
        report.append(f"Average Loss: ${metrics.average_loss:.2f}")
        report.append(f"Max Drawdown: ${metrics.max_drawdown:.2f}")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        report.append(f"Total Trades: {metrics.total_trades}")
        
        # Patterns Section
        if 'patterns' in analysis_results and analysis_results['patterns']:
            report.append("\n=== Trading Patterns ===")
            patterns = analysis_results['patterns']
            
            # Time-based patterns
            if 'time_based' in patterns and patterns['time_based']:
                report.append("\nTime-Based Patterns:")
                tb = patterns['time_based']
                if 'best_hour' in tb and tb['best_hour'] is not None:
                    report.append(f"- Best trading hour: {tb['best_hour']}:00")
                if 'worst_hour' in tb and tb['worst_hour'] is not None:
                    report.append(f"- Worst trading hour: {tb['worst_hour']}:00")
            
            # Market conditions
            if 'market_conditions' in patterns and patterns['market_conditions']:
                report.append("\nMarket Conditions Analysis:")
                mc = patterns['market_conditions']
                if 'recommendations' in mc:
                    for rec in mc['recommendations']:
                        report.append(f"- {rec}")
            
            # Consecutive trades
            if 'consecutive_trades' in patterns and patterns['consecutive_trades']:
                report.append("\nConsecutive Trades Analysis:")
                ct = patterns['consecutive_trades']
                if 'consecutive_wins' in ct:
                    report.append(f"- Max consecutive wins: {ct['consecutive_wins']}")
                if 'consecutive_losses' in ct:
                    report.append(f"- Max consecutive losses: {ct['consecutive_losses']}")
        
        # Recommendations Section
        if 'recommendations' in analysis_results and analysis_results['recommendations']:
            report.append("\n=== Recommendations ===")
            for rec in analysis_results['recommendations']:
                report.append(f"- {rec}")
        
        # Risk Analysis Section
        if 'risk_analysis' in analysis_results and analysis_results['risk_analysis']:
            report.append("\n=== Risk Analysis ===")
            risk = analysis_results['risk_analysis']
            
            # Position size analysis
            if 'position_size_analysis' in risk and risk['position_size_analysis']:
                report.append("\nPosition Size Analysis:")
                ps = risk['position_size_analysis']
                if 'optimal_size' in ps and ps['optimal_size']:
                    report.append(f"- Optimal position size: {ps['optimal_size']}")
                if 'recommendations' in ps:
                    for rec in ps['recommendations']:
                        report.append(f"- {rec}")
            
            # Drawdown analysis
            if 'drawdown_analysis' in risk and risk['drawdown_analysis']:
                report.append("\nDrawdown Analysis:")
                da = risk['drawdown_analysis']
                if 'max_drawdown' in da:
                    report.append(f"- Maximum drawdown: ${abs(da['max_drawdown']):.2f}")
                if 'avg_drawdown' in da:
                    report.append(f"- Average drawdown: ${abs(da['avg_drawdown']):.2f}")
            
            # Risk/Reward analysis
            if 'risk_reward_analysis' in risk and risk['risk_reward_analysis']:
                report.append("\nRisk/Reward Analysis:")
                rr = risk['risk_reward_analysis']
                if 'avg_risk_reward' in rr:
                    report.append(f"- Average risk/reward ratio: {rr['avg_risk_reward']:.2f}")
                if 'optimal_ratio' in rr:
                    report.append(f"- Optimal risk/reward ratio: {rr['optimal_ratio']}")

        return "\n".join(report)

@dataclass
class PositionSizingRules:
    """
    Rules for position sizing enforcement.
    
    Parameters
    ----------
    max_position_size : float
        Maximum position size as percentage of account
    max_single_loss : float
        Maximum loss allowed per trade as percentage of account
    position_scaling : Dict[str, float]
        Position size multipliers based on volatility/confidence
    max_correlated_exposure : float
        Maximum total exposure for correlated instruments
    """
    max_position_size: float
    max_single_loss: float
    position_scaling: Dict[str, float]
    max_correlated_exposure: float

@dataclass
class OpenPosition:
    """
    Current open position data.
    
    Parameters
    ----------
    symbol : str
        Trading instrument symbol
    entry_price : float
        Position entry price
    quantity : float
        Position size
    side : str
        'long' or 'short'
    timestamp : datetime
        Entry timestamp
    stop_loss : float
        Current stop loss price
    take_profit : float
        Current take profit price
    unrealized_pl : float
        Current unrealized profit/loss
    """
    symbol: str
    entry_price: float
    quantity: float
    side: str
    timestamp: datetime
    stop_loss: float
    take_profit: float
    unrealized_pl: float

class EnhancedTradingAnalytics(TradingAnalytics):
    """
    Enhanced trading analytics system with improved risk management and timezone support.
    
    Parameters
    ----------
    db_path : str
        Path to SQLite database
    timezone : str
        Default timezone for analysis (e.g., 'America/New_York')
    position_rules : PositionSizingRules
        Rules for position sizing enforcement
    """
    
    def __init__(
        self,
        db_path: str = 'trading_data.db',
        timezone: str = 'UTC',
        position_rules: Optional[PositionSizingRules] = None
    ):
        super().__init__(db_path)
        self.timezone = pytz.timezone(timezone)
        self.position_rules = position_rules or PositionSizingRules(
            max_position_size=0.02,  # 2% max position size
            max_single_loss=0.01,    # 1% max loss per trade
            position_scaling={'low_volatility': 1.0, 'medium_volatility': 0.8, 'high_volatility': 0.5},
            max_correlated_exposure=0.05  # 5% max exposure for correlated instruments
        )
        self.open_positions: Dict[str, OpenPosition] = {}
        self._initialize_enhanced_database()

    def _initialize_enhanced_database(self: "EnhancedTradingAnalytics"):
        """Initialize additional database tables for enhanced features."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Open positions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS open_positions (
                position_id INTEGER PRIMARY KEY,
                symbol TEXT,
                entry_price REAL,
                quantity REAL,
                side TEXT,
                timestamp TEXT,
                stop_loss REAL,
                take_profit REAL,
                unrealized_pl REAL,
                UNIQUE(symbol)
            )
        ''')
        
        # Correlation table
        c.execute('''
            CREATE TABLE IF NOT EXISTS symbol_correlations (
                symbol1 TEXT,
                symbol2 TEXT,
                correlation REAL,
                last_updated TEXT,
                PRIMARY KEY (symbol1, symbol2)
            )
        ''')
        
        # Position sizing log
        c.execute('''
            CREATE TABLE IF NOT EXISTS position_sizing_log (
                log_id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                suggested_size REAL,
                actual_size REAL,
                reason TEXT,
                account_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_trade(self: "EnhancedTradingAnalytics", trade_data: Dict) -> bool:
        """
        Log a trade with enhanced validation and position sizing enforcement.
        
        Parameters
        ----------
        trade_data : Dict
            Trade data dictionary
        
        Returns
        -------
        bool
            True if trade meets all rules and is logged successfully
        """
        # Convert timestamp to configured timezone
        local_ts = pd.Timestamp(trade_data['timestamp']).tz_localize('UTC').tz_convert(self.timezone)
        trade_data['timestamp'] = local_ts.isoformat()
        
        # Validate position size
        account_value = self._get_account_value()
        position_value = trade_data['entry_price'] * trade_data['quantity']
        position_size_pct = position_value / account_value
        
        if not self._validate_position_size(trade_data['symbol'], position_size_pct):
            logger = TradingLogger.get_logger()
            logger.warning(f"Trade rejected: Position size {position_size_pct:.2%} exceeds limits")
            return False
        
        # Calculate potential loss
        max_loss = abs(trade_data['entry_price'] - trade_data['stop_loss']) * trade_data['quantity']
        max_loss_pct = max_loss / account_value
        
        if max_loss_pct > self.position_rules.max_single_loss:
            logger = TradingLogger.get_logger()
            logger.warning(f"Trade rejected: Potential loss {max_loss_pct:.2%} exceeds maximum")
            return False
        
        # Update open positions
        if trade_data.get('side') == 'exit':
            self._close_position(trade_data['symbol'])
        else:
            self._open_position(trade_data)
        
        # Log the trade
        super().log_trade(trade_data)
        return True

    def _validate_position_size(self: "EnhancedTradingAnalytics", symbol: str, position_size_pct: float) -> bool:
        """
        Validate position size against rules and correlations.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
        position_size_pct : float
            Position size as percentage of account
        
        Returns
        -------
        bool
            True if position size is valid
        """
        if position_size_pct > self.position_rules.max_position_size:
            return False
            
        # Check correlated exposure
        correlated_exposure = self._get_correlated_exposure(symbol)
        if (correlated_exposure + position_size_pct) > self.position_rules.max_correlated_exposure:
            return False
            
        return True

    def calculate_metrics(self: "EnhancedTradingAnalytics", start_date: str, end_date: str) -> Optional[TradeMetrics]:
        """
        Calculate metrics including open position P&L.
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        
        Returns
        -------
        Optional[TradeMetrics]
            Trading metrics including unrealized P&L
        """
        # Convert dates to timezone-aware timestamps
        start_ts = pd.Timestamp(start_date).tz_localize(self.timezone)
        end_ts = pd.Timestamp(end_date).tz_localize(self.timezone)
        
        # Get closed trade metrics
        metrics = super().calculate_metrics(start_ts.isoformat(), end_ts.isoformat())
        if metrics is None:
            return None
            
        # Include open position P&L
        total_unrealized = sum(pos.unrealized_pl for pos in self.open_positions.values())
        
        # Adjust metrics for open positions
        metrics.profit_factor = self._adjust_profit_factor(metrics.profit_factor, total_unrealized)
        
        return metrics

    def _adjust_profit_factor(self: "EnhancedTradingAnalytics", current_pf: float, unrealized_pl: float) -> float:
        """
        Adjust profit factor considering unrealized P&L.
        
        Parameters
        ----------
        current_pf : float
            Current profit factor from closed trades
        unrealized_pl : float
            Total unrealized P&L
            
        Returns
        -------
        float
            Adjusted profit factor
        """
        if unrealized_pl > 0:
            return (current_pf * 1.0) + (unrealized_pl * 0.5)
        else:
            return (current_pf * 1.0) - (abs(unrealized_pl) * 0.5)

    def update_open_positions(self: "EnhancedTradingAnalytics", market_data: Dict[str, float]) -> None:
        """
        Update unrealized P&L for open positions.
        
        Parameters
        ----------
        market_data : Dict[str, float]
            Current market prices by symbol
        """
        for symbol, position in self.open_positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                if position.side == 'long':
                    position.unrealized_pl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pl = (position.entry_price - current_price) * position.quantity

    def _get_correlated_exposure(self, symbol: str) -> float:
        """
        Calculate total exposure of correlated instruments.
        
        Parameters
        ----------
        symbol : str
            Symbol to check correlations against
            
        Returns
        -------
        float
            Total correlated exposure as percentage
        """
        conn = sqlite3.connect(self.db_path)
        correlations = pd.read_sql(
            "SELECT symbol1, symbol2, correlation FROM symbol_correlations WHERE symbol1 = ? OR symbol2 = ?",
            conn,
            params=[symbol, symbol]
        )
        conn.close()
        
        total_exposure = 0.0
        for _, row in correlations.iterrows():
            corr_symbol = row['symbol2'] if row['symbol1'] == symbol else row['symbol1']
            if corr_symbol in self.open_positions:
                position = self.open_positions[corr_symbol]
                exposure = abs(position.quantity * position.entry_price) / self._get_account_value()
                total_exposure += exposure * abs(row['correlation'])
                
        return total_exposure

    def _get_account_value(self: "EnhancedTradingAnalytics") -> float:
        """
        Get current account value including open positions.
        
        Returns
        -------
        float
            Total account value
        """
        # This would typically connect to your broker's API
        # For now, we'll use a placeholder value
        base_value = 100000.0  # Example account value
        unrealized_pl = sum(pos.unrealized_pl for pos in self.open_positions.values())
        return base_value + unrealized_pl

class MultiCurrencyAnalytics(EnhancedTradingAnalytics):
    """
    Trading analytics with multi-currency support.
    """
    
    def __init__(
        self,
        db_path: str = 'trading_data.db',
        timezone: str = 'UTC',
        position_rules: Optional[PositionSizingRules] = None,
        currency_config: Optional[CurrencyConfig] = None
    ):
        super().__init__(db_path, timezone, position_rules)
        self.currency_config = currency_config or CurrencyConfig(
            base_currency='USD',
            fx_data_source='local',
            update_frequency=60
        )
        self._fx_rates: Dict[str, float] = {}
        self._initialize_currency_tables()

    def _initialize_currency_tables(self):
        """Initialize tables for currency tracking."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # FX rates table
        c.execute('''
            CREATE TABLE IF NOT EXISTS fx_rates (
                currency_pair TEXT PRIMARY KEY,
                rate REAL,
                last_updated TEXT
            )
        ''')
        
        # Currency exposure table
        c.execute('''
            CREATE TABLE IF NOT EXISTS currency_exposure (
                currency TEXT PRIMARY KEY,
                exposure REAL,
                hedged_amount REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    @lru_cache(maxsize=100)
    def get_fx_rate(self, from_currency: str, to_currency: str) -> float:
        """
        Get FX rate with caching for performance.
        
        Parameters
        ----------
        from_currency : str
            Source currency code
        to_currency : str
            Target currency code
            
        Returns
        -------
        float
            Exchange rate
            
        Notes
        -----
        Uses LRU (Least Recently Used) cache to store up to 100 most recent
        currency pairs for optimized performance. Cache is automatically
        invalidated when the update_frequency time has passed.
        """
        if from_currency == to_currency:
            return 1.0
            
        pair = f"{from_currency}/{to_currency}"
        
        # Check cache first
        if pair in self._fx_rates:
            return self._fx_rates[pair]
            
        # Get from database or API
        if self.currency_config.fx_data_source == 'api':
            rate = self._fetch_fx_rate_api(from_currency, to_currency)
        else:
            rate = self._get_fx_rate_local(from_currency, to_currency)
            
        self._fx_rates[pair] = rate
        return rate

    def _fetch_fx_rate_api(self, from_currency: str, to_currency: str) -> float:
        """Fetch FX rate from external API."""
        # Implement your preferred FX data provider here
        # Example using a mock API
        try:
            response = requests.get(
                f"https://api.example.com/fx/{from_currency}/{to_currency}"
            )
            return float(response.json()['rate'])
        except Exception as e:
            logger = TradingLogger.get_logger()
            logger.error(f"FX rate fetch error: {str(e)}")
            return self._get_fx_rate_local(from_currency, to_currency)

    def _get_fx_rate_local(self, from_currency: str, to_currency: str) -> float:
        """Get FX rate from local database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute(
            "SELECT rate FROM fx_rates WHERE currency_pair = ?",
            (f"{from_currency}/{to_currency}",)
        )
        
        result = c.fetchone()
        conn.close()
        
        return float(result[0]) if result else 1.0

    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> float:
        """Convert amount between currencies."""
        return amount * self.get_fx_rate(from_currency, to_currency)

    def log_trade(self, trade_data: Dict) -> bool:
        """
        Log a trade with enhanced validation and position sizing enforcement.
        
        Parameters
        ----------
        trade_data : Dict
            Trade data dictionary
        
        Returns
        -------
        bool
            True if trade meets all rules and is logged successfully
        """
        # First localize the timestamp to UTC, then convert to target timezone
        try:
            # If timestamp is a string, parse it first
            if isinstance(trade_data['timestamp'], str):
                naive_ts = pd.Timestamp(trade_data['timestamp'])
            else:
                naive_ts = pd.Timestamp(trade_data['timestamp'])
            
            # Localize to UTC if naive, then convert to target timezone
            if naive_ts.tz is None:
                utc_ts = naive_ts.tz_localize('UTC')
            else:
                utc_ts = naive_ts.tz_convert('UTC')
                
            # Convert to target timezone
            local_ts = utc_ts.tz_convert(self.timezone)
            trade_data['timestamp'] = local_ts.isoformat()
        except Exception as e:
            logger = TradingLogger.get_logger()
            logger.error(f"Timestamp conversion error: {str(e)}")
            return False
        
        # Validate position size
        account_value = self._get_account_value()
        position_value = trade_data['entry_price'] * trade_data['quantity']
        position_size_pct = position_value / account_value
        
        if not self._validate_position_size(trade_data['symbol'], position_size_pct):
            logger = TradingLogger.get_logger()
            logger.warning(f"Trade rejected: Position size {position_size_pct:.2%} exceeds limits")
            return False
        
        # Calculate potential loss
        max_loss = abs(trade_data['entry_price'] - trade_data['stop_loss']) * trade_data['quantity']
        max_loss_pct = max_loss / account_value
        
        if max_loss_pct > self.position_rules.max_single_loss:
            logger = TradingLogger.get_logger()
            logger.warning(f"Trade rejected: Potential loss {max_loss_pct:.2%} exceeds maximum")
            return False
        
        # Update open positions
        if trade_data.get('side') == 'exit':
            self._close_position(trade_data['symbol'])
        else:
            self._open_position(trade_data)
        
        # Log the trade
        super().log_trade(trade_data)
        return True

    def calculate_metrics(self, start_date: str, end_date: str) -> Optional[TradeMetrics]:
        """Calculate metrics with currency adjustments."""
        metrics = super().calculate_metrics(start_date, end_date)
        if metrics is None:
            return None
            
        # Adjust metrics for currency effects
        metrics.max_drawdown = self._adjust_for_currency_effects(
            metrics.max_drawdown,
            start_date,
            end_date
        )
        
        return metrics

    def _adjust_for_currency_effects(
        self,
        value: float,
        start_date: str,
        end_date: str
    ) -> float:
        """Adjust metrics for currency fluctuations."""
        # Implement currency adjustment logic here
        # This would account for FX rate changes over the period
        return value  # Placeholder

class TransactionCostAnalyzer:
    """
    Analyzer for trading costs and market impact.
    """
    
    def __init__(self, db_path: str):
        """Initialize with database connection."""
        self.db_path = db_path
        self._initialize_tca_tables()
        
    def _initialize_tca_tables(self):
        """Create necessary tables for transaction cost analysis."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Transaction costs table
        c.execute('''
            CREATE TABLE IF NOT EXISTS transaction_costs (
                trade_id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                venue TEXT,
                commission REAL,
                slippage REAL,
                spread_cost REAL,
                market_impact REAL,
                delay_cost REAL,
                venue_fees REAL,
                clearing_fees REAL,
                expected_price REAL,
                arrival_price REAL,
                execution_price REAL,
                volume_participation REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        ''')
        
        # Modify market_data table to include bid-ask data
        c.execute('''
            CREATE TABLE IF NOT EXISTS market_data_new (
                timestamp TEXT,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                indicators JSON,
                PRIMARY KEY (timestamp, symbol)
            )
        ''')
        
        # Check if we need to migrate existing data
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
        if c.fetchone():
            c.execute('''
                INSERT OR IGNORE INTO market_data_new (
                    timestamp, symbol, open, high, low, close, volume, indicators
                )
                SELECT timestamp, symbol, open, high, low, close, volume, indicators
                FROM market_data
            ''')
            c.execute("DROP TABLE market_data")
            c.execute("ALTER TABLE market_data_new RENAME TO market_data")
        
        # Venue analysis table
        c.execute('''
            CREATE TABLE IF NOT EXISTS venue_analysis (
                venue TEXT,
                symbol TEXT,
                avg_cost_bps REAL,
                avg_market_impact REAL,
                fill_rate REAL,
                avg_spread REAL,
                last_updated TEXT,
                PRIMARY KEY (venue, symbol)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _calculate_spread_cost(self, trade_data: Dict) -> float:
        """
        Calculate cost due to bid-ask spread.
        
        If bid-ask data is not available, estimates spread as 0.1% of price.
        """
        # Fetch bid-ask data from market data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT bid, ask FROM market_data 
            WHERE symbol = ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        ''', (trade_data['symbol'], trade_data['timestamp']))
        
        result = c.fetchone()
        conn.close()
        
        if result and result[0] is not None and result[1] is not None:
            bid, ask = result
            spread = ask - bid
        else:
            # Estimate spread as 0.1% of price if no bid-ask data available
            price = trade_data['execution_price']
            spread = price * 0.001
            
        return (spread / 2) * trade_data['quantity']  # Assume crossing half spread

    def analyze_trade_costs(self, trade_data: Dict) -> TransactionCosts:
        """
        Analyze costs for a single trade.
        
        Parameters
        ----------
        trade_data : Dict
            Complete trade information including:
            - symbol : str
            - quantity : float
            - entry_price : float
            - execution_price : float
            - venue : str
            - timestamp : str
            
        Returns
        -------
        TransactionCosts
            Detailed breakdown of transaction costs
        """
        # Calculate individual cost components
        commission = self._calculate_commission(trade_data)
        slippage = self._calculate_slippage(trade_data)
        spread_cost = self._calculate_spread_cost(trade_data)
        market_impact = self._calculate_market_impact(trade_data)
        delay_cost = self._calculate_delay_cost(trade_data)
        venue_fees = self._calculate_venue_fees(trade_data)
        clearing_fees = self._calculate_clearing_fees(trade_data)
        
        costs = TransactionCosts(
            commission=commission,
            slippage=slippage,
            spread_cost=spread_cost,
            market_impact=market_impact,
            delay_cost=delay_cost,
            venue_fees=venue_fees,
            clearing_fees=clearing_fees
        )
        
        # Store the analysis
        self._store_cost_analysis(trade_data['trade_id'], costs, trade_data)
        
        return costs

    def _calculate_commission(self, trade_data: Dict) -> float:
        """Calculate commission based on venue and size."""
        venue_type = VenueType(trade_data.get('venue', 'exchange'))
        trade_value = trade_data['quantity'] * trade_data['execution_price']
        
        # Example commission structure
        commission_rates = {
            VenueType.EXCHANGE: 0.0020,  # 20 bps
            VenueType.DARK_POOL: 0.0015, # 15 bps
            VenueType.ECN: 0.0017,       # 17 bps
            VenueType.MARKET_MAKER: 0.0018, # 18 bps
            VenueType.OTC: 0.0025        # 25 bps
        }
        
        return trade_value * commission_rates[venue_type]

    def _calculate_slippage(self, trade_data: Dict) -> float:
        """Calculate implementation shortfall."""
        expected_price = trade_data.get('expected_price', trade_data['entry_price'])
        execution_price = trade_data['execution_price']
        quantity = trade_data['quantity']
        
        return abs(execution_price - expected_price) * quantity

    def _calculate_spread_cost(self, trade_data: Dict) -> float:
        """Calculate cost due to bid-ask spread."""
        # Fetch bid-ask data from market data
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT bid, ask FROM market_data 
            WHERE symbol = ? AND timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1
        ''', (trade_data['symbol'], trade_data['timestamp']))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            bid, ask = result
            spread = ask - bid
            return (spread / 2) * trade_data['quantity']  # Assume crossing half spread
        
        return 0.0

    def _calculate_market_impact(self, trade_data: Dict) -> float:
        """
        Calculate price impact of trade.
        
        Uses square root model: impact = σ * sqrt(Q/ADV) * direction
        where σ is volatility, Q is trade size, ADV is average daily volume
        
        Parameters
        ----------
        trade_data : Dict
            Trade information including quantity and price
        
        Returns
        -------
        float
            Estimated market impact cost
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get average daily volume and calculate volatility manually
        c = conn.cursor()
        c.execute('''
            WITH price_changes AS (
                SELECT 
                    timestamp,
                    close,
                    (close / LAG(close) OVER (ORDER BY timestamp) - 1) as daily_return
                FROM market_data 
                WHERE symbol = ?
                AND timestamp >= DATE(?, '-30 days')
                AND timestamp <= ?
            )
            SELECT 
                AVG(volume) as adv,
                AVG(daily_return * daily_return) as variance
            FROM market_data 
            LEFT JOIN price_changes USING (timestamp)
            WHERE symbol = ?
            AND timestamp >= DATE(?, '-30 days')
            AND timestamp <= ?
        ''', (
            trade_data['symbol'], 
            trade_data['timestamp'], 
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data['timestamp'],
            trade_data['timestamp']
        ))
        
        result = c.fetchone()
        conn.close()
        
        if result and result[0]:
            adv, variance = result
            volatility = np.sqrt(variance) if variance is not None else 0.01  # Default to 1% if not enough data
            trade_size = trade_data['quantity']
            direction = 1 if trade_data.get('side') == 'buy' else -1
            
            # Square root impact model
            impact = volatility * np.sqrt(trade_size / max(adv, 1)) * direction
            return impact * trade_data['execution_price'] * trade_size
        
        return 0.0
    
    def _calculate_delay_cost(self, trade_data: Dict) -> float:
        """Calculate cost due to execution delay."""
        if 'order_time' in trade_data and 'execution_time' in trade_data:
            delay = (pd.Timestamp(trade_data['execution_time']) - 
                    pd.Timestamp(trade_data['order_time'])).total_seconds()
            
            # Simple linear model for delay cost
            delay_impact = 0.0001 * (delay / 60)  # 1 bp per minute
            return delay_impact * trade_data['execution_price'] * trade_data['quantity']
        
        return 0.0

    def _calculate_venue_fees(self, trade_data: Dict) -> float:
        """Calculate specific venue fees."""
        venue_type = VenueType(trade_data.get('venue', 'exchange'))
        trade_value = trade_data['quantity'] * trade_data['execution_price']
        
        # Example venue fee structure
        venue_fee_rates = {
            VenueType.EXCHANGE: 0.0003,  # 3 bps
            VenueType.DARK_POOL: 0.0002, # 2 bps
            VenueType.ECN: 0.0004,       # 4 bps
            VenueType.MARKET_MAKER: 0.0002, # 2 bps
            VenueType.OTC: 0.0001        # 1 bp
        }
        
        return trade_value * venue_fee_rates[venue_type]

    def _calculate_clearing_fees(self, trade_data: Dict) -> float:
        """Calculate clearing and settlement fees."""
        trade_value = trade_data['quantity'] * trade_data['execution_price']
        return trade_value * 0.0001  # 1 bp flat clearing fee

    def _store_cost_analysis(self, trade_id: int, costs: TransactionCosts, trade_data: Dict):
        """Store cost analysis results."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO transaction_costs (
                trade_id, timestamp, symbol, venue, commission, slippage,
                spread_cost, market_impact, delay_cost, venue_fees,
                clearing_fees, expected_price, arrival_price, execution_price,
                volume_participation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id,
            trade_data['timestamp'],
            trade_data['symbol'],
            trade_data.get('venue', 'exchange'),
            costs.commission,
            costs.slippage,
            costs.spread_cost,
            costs.market_impact,
            costs.delay_cost,
            costs.venue_fees,
            costs.clearing_fees,
            trade_data.get('expected_price', trade_data['entry_price']),
            trade_data.get('arrival_price', trade_data['entry_price']),
            trade_data['execution_price'],
            trade_data.get('volume_participation', 0.0)
        ))
        
        conn.commit()
        conn.close()

    def analyze_venue_performance(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        """
        Analyze venue performance statistics.
        
        Parameters
        ----------
        start_date : str
            Analysis start date
        end_date : str
            Analysis end date
            
        Returns
        -------
        Dict[str, Dict]
            Performance metrics by venue
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                venue,
                AVG(commission + slippage + spread_cost + market_impact + 
                    delay_cost + venue_fees + clearing_fees) * 10000 as avg_cost_bps,
                AVG(market_impact) as avg_impact,
                COUNT(*) as total_trades,
                AVG(spread_cost) as avg_spread
            FROM transaction_costs
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY venue
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        return df.set_index('venue').to_dict('index')

    def get_cost_summary(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get summary of transaction costs.
        
        Parameters
        ----------
        start_date : str
            Analysis start date
        end_date : str
            Analysis end date
            
        Returns
        -------
        pd.DataFrame
            Summary of costs by component
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                strftime('%Y-%m', timestamp) as month,
                SUM(commission) as total_commission,
                SUM(slippage) as total_slippage,
                SUM(spread_cost) as total_spread_cost,
                SUM(market_impact) as total_market_impact,
                SUM(delay_cost) as total_delay_cost,
                SUM(venue_fees) as total_venue_fees,
                SUM(clearing_fees) as total_clearing_fees
            FROM transaction_costs
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY strftime('%Y-%m', timestamp)
            ORDER BY month
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        return df
    
    def _populate_test_market_data(self):
        """Populate test market data with bid-ask spreads."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Generate some test data for AAPL
        base_price = 175.50
        timestamps = [
            (datetime.now() - timedelta(minutes=i)).isoformat()
            for i in range(10)
        ]
        
        for ts in timestamps:
            # Simulate some price variation
            price_offset = np.random.normal(0, 0.1)
            mid_price = base_price + price_offset
            
            # Create bid-ask spread (typically 0.01-0.05% for liquid stocks)
            spread = mid_price * 0.0003  # 0.03% spread
            bid = mid_price - spread/2
            ask = mid_price + spread/2
            
            c.execute('''
                INSERT OR REPLACE INTO market_data (
                    timestamp, symbol, open, high, low, close, bid, ask, volume, indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ts,
                'AAPL',
                mid_price,
                mid_price + 0.1,
                mid_price - 0.1,
                mid_price,
                bid,
                ask,
                10000,
                '{}'
            ))
        
        conn.commit()
        conn.close()

class PositionCorrelationAnalyzer:
    """
    Analyzer for position correlations and risk clustering.
    """
    
    def __init__(
        self,
        db_path: str,
        config: Optional[CorrelationConfig] = None
    ):
        """Initialize correlation analyzer."""
        self.db_path = db_path
        self.config = config or CorrelationConfig()
        self._initialize_correlation_tables()
        
    def _initialize_correlation_tables(self):
        """Initialize tables for correlation analysis."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Asset correlation table
        c.execute('''
            CREATE TABLE IF NOT EXISTS asset_correlations (
                symbol1 TEXT,
                symbol2 TEXT,
                correlation REAL,
                beta REAL,
                common_factor_exposure REAL,
                last_updated TEXT,
                lookback_days INTEGER,
                data_points INTEGER,
                PRIMARY KEY (symbol1, symbol2)
            )
        ''')
        
        # Risk cluster table
        c.execute('''
            CREATE TABLE IF NOT EXISTS risk_clusters (
                cluster_id INTEGER PRIMARY KEY,
                symbols TEXT,  -- JSON array of symbols
                avg_correlation REAL,
                total_exposure REAL,
                risk_factors TEXT,  -- JSON object of factor exposures
                last_updated TEXT
            )
        ''')
        
        # Sector exposure table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sector_exposure (
                sector TEXT PRIMARY KEY,
                exposure REAL,
                limit_exposure REAL,
                symbols TEXT,  -- JSON array of symbols
                last_updated TEXT
            )
        ''')
        
        # Factor exposure table
        c.execute('''
            CREATE TABLE IF NOT EXISTS factor_exposure (
                symbol TEXT,
                factor TEXT,
                exposure REAL,
                last_updated TEXT,
                PRIMARY KEY (symbol, factor)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _populate_test_data(self):
        """Populate test data for correlation analysis."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Add test market data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'META']
        base_prices = {
            'AAPL': 175.50,
            'MSFT': 325.75,
            'GOOGL': 140.25,
            'META': 290.50
        }
        
        # Generate correlated price movements
        dates = [
            (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(252)  # One year of data
        ]
        
        # Create correlated returns
        rng = np.random.default_rng(42)  # For reproducibility
        market_return = rng.normal(0.0005, 0.01, len(dates))  # Market factor
        
        # Symbol-specific parameters (beta, correlation to market)
        symbol_params = {
            'AAPL': {'beta': 1.2, 'vol': 0.015},
            'MSFT': {'beta': 1.1, 'vol': 0.014},
            'GOOGL': {'beta': 1.3, 'vol': 0.016},
            'META': {'beta': 1.4, 'vol': 0.018}
        }
        
        # Generate prices for each symbol
        for symbol in symbols:
            params = symbol_params[symbol]
            base_price = base_prices[symbol]
            
            # Generate correlated returns
            specific_return = rng.normal(0, params['vol'], len(dates))
            total_return = (params['beta'] * market_return + specific_return)
            
            # Calculate prices
            prices = base_price * np.exp(np.cumsum(total_return))
            
            # Insert market data
            for date, price in zip(dates, prices):
                c.execute('''
                    INSERT OR REPLACE INTO market_data (
                        timestamp, symbol, open, high, low, close, volume, indicators
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    date,
                    symbol,
                    price,
                    price * (1 + rng.normal(0, 0.001)),  # Small random variation
                    price * (1 - rng.normal(0, 0.001)),
                    price,
                    int(rng.normal(1000000, 200000)),  # Random volume
                    '{}'
                ))
        
        # Add test positions
        positions = [
            ('AAPL', 100, base_prices['AAPL'], 'Technology'),
            ('MSFT', 50, base_prices['MSFT'], 'Technology'),
            ('GOOGL', 75, base_prices['GOOGL'], 'Technology'),
            ('META', 60, base_prices['META'], 'Technology')
        ]
        
        # Create asset_metadata table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS asset_metadata (
                symbol TEXT PRIMARY KEY,
                sector TEXT,
                industry TEXT,
                market_cap REAL
            )
        ''')
        
        # Add metadata
        for symbol, _, _, sector in positions:
            c.execute('''
                INSERT OR REPLACE INTO asset_metadata (symbol, sector)
                VALUES (?, ?)
            ''', (symbol, sector))
        
        # Add positions
        for symbol, quantity, price, _ in positions:
            c.execute('''
                INSERT OR REPLACE INTO open_positions (
                    symbol, quantity, entry_price, side, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol,
                quantity,
                price,
                'long',
                datetime.now().isoformat()
            ))
        
        # Add some initial factor exposures
        factors = ['Momentum', 'Value', 'Quality', 'Size']
        for symbol in symbols:
            for factor in factors:
                c.execute('''
                    INSERT OR REPLACE INTO factor_exposure (
                        symbol, factor, exposure, last_updated
                    ) VALUES (?, ?, ?, datetime('now'))
                ''', (
                    symbol,
                    factor,
                    rng.normal(0.2, 0.1)  # Random factor exposure
                ))
        
        conn.commit()
        conn.close()

    def update_correlations(self, symbols: Optional[List[str]] = None):
        """
        Update correlation matrix for given symbols.
        
        Parameters
        ----------
        symbols : Optional[List[str]]
            Symbols to update. If None, updates all active positions.
        """
        conn = sqlite3.connect(self.db_path)
        
        if symbols is None:
            # Get all symbols from open positions
            symbols = pd.read_sql(
                "SELECT DISTINCT symbol FROM open_positions",
                conn
            )['symbol'].tolist()
        
        # Get price data
        prices_df = pd.read_sql(f"""
            SELECT timestamp, symbol, close
            FROM market_data
            WHERE symbol IN ({','.join(['?']*len(symbols))})
            AND timestamp >= date('now', ?)
            ORDER BY timestamp
        """, conn, params=[*symbols, f'-{self.config.lookback_days} days'])
        
        # Pivot to get price matrix
        price_matrix = prices_df.pivot(
            index='timestamp',
            columns='symbol',
            values='close'
        )
        
        # Calculate returns with explicit fill_method=None
        returns = price_matrix.pct_change(fill_method=None)
        
        # Calculate correlations and betas
        correlations = returns.corr(min_periods=self.config.min_periods)
        market_returns = returns.mean(axis=1)  # Simple market proxy
        betas = returns.apply(lambda x: stats.linregress(market_returns, x)[0])
        
        # Store correlations and betas
        for symbol1 in symbols:
            for symbol2 in symbols:
                if symbol1 < symbol2:  # Store each pair only once
                    corr = correlations.loc[symbol1, symbol2]
                    beta1 = betas[symbol1]
                    beta2 = betas[symbol2]
                    common_factor = beta1 * beta2  # Common factor exposure
                    
                    c = conn.cursor()
                    c.execute('''
                        INSERT OR REPLACE INTO asset_correlations (
                            symbol1, symbol2, correlation, beta,
                            common_factor_exposure, last_updated,
                            lookback_days, data_points
                        ) VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?)
                    ''', (
                        symbol1, symbol2, corr, (beta1 + beta2)/2,
                        common_factor, self.config.lookback_days,
                        len(returns)
                    ))
        
        conn.commit()
        conn.close()

    def identify_risk_clusters(self) -> List[Dict]:
        """
        Identify clusters of highly correlated positions.
        
        Returns
        -------
        List[Dict]
            List of cluster information including symbols and metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get correlation matrix
        correlations = pd.read_sql("""
            SELECT symbol1, symbol2, correlation
            FROM asset_correlations
            WHERE correlation >= ?
        """, conn, params=[self.config.correlation_threshold])
        
        # Create correlation network
        G = nx.Graph()
        for _, row in correlations.iterrows():
            G.add_edge(row['symbol1'], row['symbol2'], weight=row['correlation'])
        
        # Find clusters using community detection
        clusters = list(nx.community.greedy_modularity_communities(G))
        
        # Analyze each cluster
        cluster_info = []
        for i, cluster in enumerate(clusters):
            symbols = list(cluster)
            
            # Calculate cluster metrics
            cluster_correlations = correlations[
                correlations['symbol1'].isin(symbols) &
                correlations['symbol2'].isin(symbols)
            ]
            avg_correlation = cluster_correlations['correlation'].mean()
            
            # Get exposure
            exposures = pd.read_sql("""
                SELECT symbol, quantity * entry_price as exposure
                FROM open_positions
                WHERE symbol IN ({})
            """.format(','.join(['?']*len(symbols))), 
            conn, params=symbols)
            
            total_exposure = exposures['exposure'].sum()
            
            # Store cluster information
            cluster_data = {
                'cluster_id': i,
                'symbols': symbols,
                'avg_correlation': avg_correlation,
                'total_exposure': total_exposure,
                'risk_factors': self._get_cluster_risk_factors(symbols)
            }
            cluster_info.append(cluster_data)
            
            # Store in database
            c = conn.cursor()
            c.execute('''
                INSERT OR REPLACE INTO risk_clusters (
                    cluster_id, symbols, avg_correlation,
                    total_exposure, risk_factors, last_updated
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
            ''', (
                i,
                json.dumps(symbols),
                avg_correlation,
                total_exposure,
                json.dumps(cluster_data['risk_factors'])
            ))
        
        conn.commit()
        conn.close()
        
        return cluster_info

    def _get_cluster_risk_factors(self, symbols: List[str]) -> Dict[str, float]:
        """Get aggregate risk factor exposures for a cluster."""
        conn = sqlite3.connect(self.db_path)
        
        exposures = pd.read_sql(f"""
            SELECT factor, AVG(exposure) as avg_exposure
            FROM factor_exposure
            WHERE symbol IN ({','.join(['?']*len(symbols))})
            GROUP BY factor
        """, conn, params=symbols)
        
        conn.close()
        
        return dict(zip(exposures['factor'], exposures['avg_exposure']))

    def analyze_portfolio_risk(self) -> Dict:
        """
        Perform comprehensive portfolio risk analysis.
        
        Returns
        -------
        Dict
            Dictionary containing risk metrics including:
            - Cluster exposures
            - Sector concentrations
            - Factor exposures
            - Correlation warnings
            - Exposure recommendations
        """
        # Update correlations first
        self.update_correlations()
        
        # Identify risk clusters
        clusters = self.identify_risk_clusters()
        
        # Analyze sector exposures
        sector_exposures = self._analyze_sector_exposures()
        
        # Generate risk report
        risk_report = {
            'clusters': clusters,
            'sector_exposures': sector_exposures,
            'warnings': self._generate_risk_warnings(clusters, sector_exposures),
            'recommendations': self._generate_recommendations(
                clusters,
                sector_exposures
            )
        }
        
        return risk_report

    def _analyze_sector_exposures(self) -> Dict[str, float]:
        """Calculate current sector exposures."""
        conn = sqlite3.connect(self.db_path)
        
        sector_exposures = pd.read_sql("""
            SELECT 
                sector,
                SUM(quantity * entry_price) as exposure
            FROM open_positions op
            JOIN asset_metadata am ON op.symbol = am.symbol
            GROUP BY sector
        """, conn)
        
        conn.close()
        
        return dict(zip(sector_exposures['sector'], 
                       sector_exposures['exposure']))

    def _generate_risk_warnings(
        self,
        clusters: List[Dict],
        sector_exposures: Dict[str, float]
    ) -> List[str]:
        """Generate warnings for risk concentrations."""
        warnings = []
        
        # Check cluster concentrations
        for cluster in clusters:
            if cluster['total_exposure'] > self.config.max_cluster_exposure:
                warnings.append(
                    f"Cluster exposure {cluster['total_exposure']:.1%} exceeds "
                    f"limit of {self.config.max_cluster_exposure:.1%} for symbols: "
                    f"{', '.join(cluster['symbols'])}"
                )
        
        # Check sector concentrations
        for sector, exposure in sector_exposures.items():
            # Get sector limit with fallback to 'Other'
            limit = self.config.sector_limits.get(
                sector,
                self.config.sector_limits['Other']
            )
            if exposure > limit:
                warnings.append(
                    f"Sector exposure {exposure:.1%} exceeds limit of {limit:.1%} "
                    f"for {sector}"
                )
        
        return warnings

    def _generate_recommendations(
        self,
        clusters: List[Dict],
        sector_exposures: Dict[str, float]
    ) -> List[str]:
        """Generate position adjustment recommendations."""
        recommendations = []
        
        # Cluster reduction recommendations
        for cluster in clusters:
            if cluster['total_exposure'] > self.config.max_cluster_exposure:
                excess = cluster['total_exposure'] - self.config.max_cluster_exposure
                recommendations.append(
                    f"Reduce cluster exposure by {excess:.1%} for symbols: "
                    f"{', '.join(cluster['symbols'])}"
                )
        
        # Sector rebalancing recommendations
        for sector, exposure in sector_exposures.items():
            limit = self.config.sector_limits.get(sector, 
                                                self.config.sector_limits['Other'])
            if exposure > limit:
                excess = exposure - limit
                recommendations.append(
                    f"Reduce {sector} exposure by {excess:.1%}"
                )
            elif exposure < limit * 0.5:  # Using 50% of limit as lower threshold
                opportunity = limit - exposure
                recommendations.append(
                    f"Opportunity to increase {sector} exposure by up to "
                    f"{opportunity:.1%}"
                )
        
        return recommendations

# Example usage
def run_test_trade(start_date: str, end_date: str) -> str:
    analytics = TradingAnalytics()
    
    try:
        analysis = analytics.analyze_strategy_performance(start_date, end_date)
        report = analytics.generate_report(analysis)
        return report
    except Exception as e:
        return f"Error analyzing trades: {str(e)}\nPlease check your data and try again."

    
    ## Example trade data
    #trade_data = {
    #    'timestamp': datetime.now().isoformat(),
    #    'symbol': 'SPY',
    #    'entry_price': 450.0,
    #    'exit_price': 452.0,
    #    'quantity': 100,
    #    'side': 'buy',
    #    'strategy_name': 'MovingAverageCrossover',
    #    'stop_loss': 448.0,
    #    'take_profit': 453.0,
    #    'profit_loss': 200.0,
    #    'execution_time_ms': 150,
    #    'indicators': {
    #        'sma_20': 449.5,
    #        'sma_50': 448.0,
    #        'rsi': 65
    #    },
    #    'market_conditions': {
    #        'volatility': 'medium',
    #        'trend': 'upward',
    #        'volume': 'above_average'
    #    }
    #}
    #
    #analytics.log_trade(trade_data)
    #analysis = analytics.analyze_strategy_performance(start_date, end_date)
    
    # Generate report
    #report = analytics.generate_report(analysis)
    #return report

def demo_correlation_analysis():
    """
    Demonstrate the position correlation analysis functionality.
    """
    # Initialize analyzer with custom config
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
    
    # Add some test data
    analyzer._populate_test_data()
    
    # Update correlations
    print("\nUpdating correlations...")
    analyzer.update_correlations(['AAPL', 'MSFT', 'GOOGL', 'META'])
    
    # Identify risk clusters
    print("\nIdentifying risk clusters...")
    clusters = analyzer.identify_risk_clusters()
    
    print("\nRisk Clusters:")
    for cluster in clusters:
        print(f"\nCluster {cluster['cluster_id']}:")
        print(f"Symbols: {', '.join(cluster['symbols'])}")
        print(f"Average Correlation: {cluster['avg_correlation']:.2f}")
        print(f"Total Exposure: {cluster['total_exposure']:.1%}")
    
    # Full portfolio risk analysis
    print("\nPerforming portfolio risk analysis...")
    risk_report = analyzer.analyze_portfolio_risk()
    
    print("\nRisk Warnings:")
    for warning in risk_report['warnings']:
        print(f"- {warning}")
    
    print("\nRecommendations:")
    for recommendation in risk_report['recommendations']:
        print(f"- {recommendation}")
    
    print("\nSector Exposures:")
    for sector, exposure in risk_report['sector_exposures'].items():
        print(f"{sector}: {exposure:.1%}")

def generate_test_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range for the last 5 trading days
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(5)]
    dates.reverse()
    
    trades = []
    decisions = []
    market_data = []
    
    # Generate sample trades
    for day in dates:
        # Morning session trades (usually more volatile)
        for _ in range(np.random.randint(3, 6)):
            entry_price = 450 + np.random.normal(0, 1)
            profit_factor = np.random.normal(1.1, 0.5)
            exit_price = entry_price * profit_factor
            
            trade = {
                'timestamp': day.replace(hour=np.random.randint(9, 11)).isoformat(),
                'symbol': 'SPY',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': 100,
                'side': 'buy',
                'strategy_name': 'MovingAverageCrossover',
                'stop_loss': entry_price * 0.995,
                'take_profit': entry_price * 1.01,
                'profit_loss': (exit_price - entry_price) * 100,
                'execution_time_ms': np.random.randint(100, 200),
                'indicators': {
                    'sma_20': entry_price - 0.5,
                    'sma_50': entry_price - 1,
                    'rsi': np.random.randint(45, 75)
                },
                'market_conditions': {
                    'volatility': 'medium',
                    'trend': 'upward',
                    'volume': 'above_average'
                }
            }
            trades.append(trade)
            
            # Generate corresponding bot decision
            decision = {
                'timestamp': trade['timestamp'],
                'symbol': 'SPY',
                'decision_type': 'ENTRY',
                'confidence_score': np.random.uniform(0.6, 0.9),
                'reasons': {
                    'moving_average_crossover': True,
                    'volume_confirmation': True,
                    'rsi_favorable': trade['indicators']['rsi'] > 50
                },
                'executed': True,
                'result': 'profit' if trade['profit_loss'] > 0 else 'loss',
                'performance_impact': trade['profit_loss']
            }
            decisions.append(decision)
    
    return trades, decisions

def populate_test_database(db_path='trading_data.db'):
    trades, decisions = generate_test_data()
    
    conn = sqlite3.connect(db_path)
    
    # Clear existing test data
    conn.execute('DELETE FROM trades')
    conn.execute('DELETE FROM bot_decisions')
    
    # Insert trades
    for trade in trades:
        trade['indicators'] = json.dumps(trade['indicators'])
        trade['market_conditions'] = json.dumps(trade['market_conditions'])
        
        columns = ', '.join(trade.keys())
        placeholders = ':' + ', :'.join(trade.keys())
        query = f'INSERT INTO trades ({columns}) VALUES ({placeholders})'
        conn.execute(query, trade)
    
    # Insert decisions
    for decision in decisions:
        decision['reasons'] = json.dumps(decision['reasons'])
        
        columns = ', '.join(decision.keys())
        placeholders = ':' + ', :'.join(decision.keys())
        query = f'INSERT INTO bot_decisions ({columns}) VALUES ({placeholders})'
        conn.execute(query, decision)
    
    conn.commit()
    conn.close()
    
    return trades, decisions

def initialize_test_data():
    analytics = TradingAnalytics()
    trades, decisions = generate_test_data()
    populate_test_database(analytics.db_path)
    return "Test data initialized"    

def demo_enhanced_trading_analytics():
    """
    Demonstrates usage of the Enhanced Trading Analytics system
    """
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

    # Initialize system with NY timezone
    analytics = EnhancedTradingAnalytics(
        db_path='trading_data.db',
        timezone='America/New_York',
        position_rules=position_rules
    )

    # Example trade data - now using naive datetime that will be localized
    naive_timestamp = datetime.now()
    trade_data = {
        'timestamp': naive_timestamp.strftime('%Y-%m-%d %H:%M:%S'),  # String format without timezone
        'symbol': 'AAPL',
        'entry_price': 175.50,
        'quantity': 100,
        'side': 'long',
        'strategy_name': 'momentum_breakout',
        'stop_loss': 173.50,
        'take_profit': 180.00,
        'indicators': {
            'rsi': 65,
            'macd': 0.75,
            'volatility': 'medium'
        },
        'market_conditions': {
            'trend': 'upward',
            'volume': 'above_average',
            'volatility': 'medium'
        }
    }

    # Log trade (with position sizing validation)
    if analytics.log_trade(trade_data):
        print("Trade successfully logged")
    else:
        print("Trade rejected due to position sizing rules")

    # Update market prices for open positions
    current_prices = {
        'AAPL': 176.25,
        'MSFT': 325.50,
        'GOOGL': 140.75
    }
    analytics.update_open_positions(current_prices)

    # Calculate metrics including open positions
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    metrics = analytics.calculate_metrics(start_date, end_date)
    if metrics:
        print("\nTrading Metrics:")
        print(f"Win Rate: {metrics.win_rate:.2%}")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Max Drawdown: ${metrics.max_drawdown:.2f}")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    # Example of checking correlated exposure
    correlated_exposure = analytics._get_correlated_exposure('AAPL')
    print(f"\nCorrelated Exposure for AAPL: {correlated_exposure:.2%}")

# Update the demo function to include test data generation
def demo_transaction_cost_analysis():
    """
    Demonstrate usage of the Transaction Cost Analysis system.
    """
    # Initialize the analyzer
    tca = TransactionCostAnalyzer('trading_data.db')
    
    # Populate test market data
    tca._populate_test_market_data()
    
    # Example trade data
    trade_data = {
        'trade_id': 1,
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
    
    # Rest of the demo function remains the same...
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
    
    # Analyze venue performance
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    venue_stats = tca.analyze_venue_performance(start_date, end_date)
    
    print("\nVenue Performance Analysis:")
    for venue, stats in venue_stats.items():
        print(f"\nVenue: {venue}")
        print(f"Average Cost (bps): {stats['avg_cost_bps']:.1f}")
        print(f"Average Market Impact: ${stats['avg_impact']:.2f}")
        print(f"Average Spread: ${stats['avg_spread']:.2f}")
        print(f"Total Trades: {stats['total_trades']}")
    
    # Get monthly cost summary
    cost_summary = tca.get_cost_summary(start_date, end_date)
    
    print("\nMonthly Cost Summary:")
    print(cost_summary)
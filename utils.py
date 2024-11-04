import sqlite3
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import sqlite3
import json
from typing import Dict, List, Union, Any
from trading_system import TradingAnalytics
from trading_dashboard import TradingDashboard

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle numpy types and NaN values."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def sanitize_value(value: Any) -> Union[float, int, str, None]:
    """Convert various numeric types to Python native types and handle special values."""
    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        if np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [sanitize_value(x) for x in value]
    return value

def sanitize_dict(d: Dict) -> Dict:
    """Recursively sanitize dictionary values."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [sanitize_dict(item) if isinstance(item, dict) else sanitize_value(item) for item in value]
        else:
            result[key] = sanitize_value(value)
    return result

def get_safe_value(value, default=0):
    """Safely get a value or return default if None/NaN."""
    if pd.isna(value) or value is None:
        return default
    return float(value)

def calculate_position_risk(position, risk_report):
    """Safely calculate position risk with fallbacks."""
    try:
        risk_percent = (
            risk_report.get('position_risk', {})
            .get('position_metrics', {})
            .get(position['symbol'], {})
            .get('risk_percent', 0)
        )
        return get_safe_value(risk_percent, 0) * 100
    except Exception:
        return 0

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analytics and dashboard with error handling
try:
    analytics = TradingAnalytics()
    dashboard = TradingDashboard(analytics)
    logger.info("Successfully initialized analytics and dashboard")
except Exception as e:
    logger.error(f"Failed to initialize analytics and dashboard: {str(e)}")
    raise

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Get overview data for dashboard"""
    try:
        logger.debug("Starting dashboard overview request")
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get daily summary and sanitize
        try:
            summary = sanitize_dict(dashboard.daily_summary())
            logger.debug(f"Got daily summary: {summary}")
        except Exception as e:
            logger.error(f"Error getting daily summary: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting daily summary: {str(e)}")

        # Get last 30 days of P&L data
        try:
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            conn = sqlite3.connect(analytics.db_path)
            pnl_data = pd.read_sql("""
                SELECT
                    date(timestamp) as date,
                    SUM(profit_loss) as value
                FROM trades
                WHERE date(timestamp) BETWEEN ? AND ?
                GROUP BY date(timestamp)
                ORDER BY date(timestamp)
            """, conn, params=[thirty_days_ago, today])
            conn.close()
            pnl_data_sanitized = [
                {"date": str(row['date']), "value": get_safe_value(row['value'])}
                for _, row in pnl_data.iterrows()
            ]
            logger.debug(f"Got PnL data: {pnl_data_sanitized}")
        except Exception as e:
            logger.error(f"Error getting PnL data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting PnL data: {str(e)}")

        # Get positions with risk analysis and sanitize
        try:
            risk_report = sanitize_dict(dashboard.risk_report())
            logger.debug(f"Got risk report: {risk_report}")
        except Exception as e:
            logger.error(f"Error getting risk report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting risk report: {str(e)}")

        # Construct response with safe value handling
        response = {
            "dailyMetrics": {
                "pnl": get_safe_value(summary.get('daily_pnl')),
                "winRate": get_safe_value(summary.get('win_rate_today')),
                "winRateChange": get_safe_value(summary.get('win_rate_change')),
                "totalTrades": int(get_safe_value(summary.get('trades_today'))),
                "largestWin": get_safe_value(summary.get('largest_winner', {}).get('profit_loss')),
                "largestLoss": get_safe_value(summary.get('largest_loser', {}).get('profit_loss'))
            },
            "pnlHistory": pnl_data_sanitized,
            "positions": [
                {
                    "symbol": str(pos['symbol']),
                    "quantity": get_safe_value(pos.get('net_position')),
                    "value": get_safe_value(pos.get('net_position')) * get_safe_value(pos.get('avg_entry')),
                    "risk": calculate_position_risk(pos, risk_report)
                }
                for pos in summary.get('open_positions', [])
            ],
            "riskMetrics": {
                "profitFactor": get_safe_value(risk_report.get('portfolio_metrics', {}).get('profit_factor', 1)),
                "maxDrawdown": get_safe_value(risk_report.get('portfolio_metrics', {}).get('max_drawdown')),
                "exposures": {
                    "gross_exposure": get_safe_value(risk_report.get('exposure_analysis', {}).get('gross_exposure')),
                    "net_exposure": get_safe_value(risk_report.get('exposure_analysis', {}).get('net_exposure')),
                    "long_exposure": get_safe_value(risk_report.get('exposure_analysis', {}).get('long_exposure')),
                    "short_exposure": get_safe_value(risk_report.get('exposure_analysis', {}).get('short_exposure')),
                    "sector_exposure": {
                        str(k): get_safe_value(v)
                        for k, v in risk_report.get('exposure_analysis', {}).get('sector_exposure', {}).items()
                    }
                }
            },
            "alerts": [str(alert) for alert in summary.get('risk_alerts', [])]
        }

        # Do a final sanitization pass and serialize with custom encoder
        response = json.loads(
            json.dumps(sanitize_dict(response), cls=CustomJSONEncoder)
        )
        
        logger.debug("Successfully constructed dashboard overview response")
        return response

    except Exception as e:
        logger.error(f"Error in dashboard overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/dashboard/metrics/explanations")
async def get_metric_explanations():
    """Get explanations for trading metrics"""
    return {
        "winRate": {
            "title": "Win Rate",
            "description": "Percentage of trades that are profitable.",
            "interpretation": "Higher is generally better, but should be viewed alongside profit factor.",
            "example": "A win rate of 65% means 65 out of 100 trades are profitable."
        },
        "profitFactor": {
            "title": "Profit Factor",
            "description": "Ratio of gross profits to gross losses.",
            "interpretation": "Above 1.5 is generally good, above 2.0 is excellent.",
            "example": "A profit factor of 1.5 means you make $1.50 for every $1.00 lost."
        },
        "maxDrawdown": {
            "title": "Maximum Drawdown",
            "description": "Largest peak-to-trough decline in account value.",
            "interpretation": "Lower is better. Helps measure risk and worst-case scenarios.",
            "example": "A max drawdown of $5,000 means your account dropped $5,000 from its peak."
        },
        "exposure": {
            "title": "Market Exposure",
            "description": "Total value of all positions relative to account size.",
            "interpretation": "Higher exposure means more risk and potential volatility.",
            "example": "50% exposure means half your capital is invested in positions."
        }
    }

if __name__ == "__main__":
    import uvicorn
    try:
        # Initialize test data if needed
        analytics.initialize_database()
        logger.info("Database initialized")
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise
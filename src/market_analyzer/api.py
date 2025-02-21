from pathlib import Path
import json
from datetime import date
import logging
from typing import Optional

class APIConfigHandler:
    """Handles API configuration and rate limiting"""
    
    def __init__(self, config_dir: str = '.config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / 'api_config.json'
        self.api_keys_file = self.config_dir / 'api_keys.txt'
        self.logger = logging.getLogger(__name__)
        
        # Initialize or load config
        self._load_config()
    
    def _load_config(self) -> None:
        """Load or initialize configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'alpha_vantage': {
                    'daily_calls': 0,
                    'last_reset': str(date.today()),
                    'daily_limit': 25  # Free tier limit
                }
            }
            self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_api_key(self, source: str = 'alpha_vantage') -> Optional[str]:
        """Get API key from file"""
        try:
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r') as f:
                    keys = dict(line.strip().split('=') for line in f if '=' in line)
                    return keys.get(source)
            self.logger.warning(f"No API key found for {source}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading API key: {e}")
            return None
    
    def can_make_api_call(self, source: str = 'alpha_vantage') -> bool:
        """Check if we can make an API call based on rate limits"""
        if source not in self.config:
            return True  # No limits configured
            
        source_config = self.config[source]
        today = str(date.today())
        
        # Reset counter if it's a new day
        if source_config['last_reset'] != today:
            source_config.update({
                'daily_calls': 0,
                'last_reset': today
            })
            self._save_config()
        
        return source_config['daily_calls'] < source_config['daily_limit']
    
    def record_api_call(self, source: str = 'alpha_vantage') -> None:
        """Record an API call"""
        if source not in self.config:
            return
            
        source_config = self.config[source]
        today = str(date.today())
        
        # Reset if it's a new day
        if source_config['last_reset'] != today:
            source_config['daily_calls'] = 0
            source_config['last_reset'] = today
        
        source_config['daily_calls'] += 1
        self._save_config()
    
    def get_remaining_calls(self, source: str = 'alpha_vantage') -> Optional[int]:
        """Get remaining API calls for today"""
        if source not in self.config:
            return None
            
        source_config = self.config[source]
        today = str(date.today())
        
        if source_config['last_reset'] != today:
            return source_config['daily_limit']
            
        return source_config['daily_limit'] - source_config['daily_calls']
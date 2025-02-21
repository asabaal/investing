import pytest
import os
import json
from datetime import date, timedelta
from pathlib import Path
from config_handler import APIConfigHandler

@pytest.fixture
def test_config_dir(tmp_path):
    """Create a temporary directory for test config files"""
    return str(tmp_path / "test_config")

@pytest.fixture
def api_handler(test_config_dir):
    """Create APIConfigHandler instance with test directory"""
    return APIConfigHandler(config_dir=test_config_dir)

@pytest.fixture
def setup_api_keys(test_config_dir):
    """Set up test API keys file"""
    keys_file = Path(test_config_dir) / 'api_keys.txt'
    keys_file.parent.mkdir(exist_ok=True)
    keys_file.write_text('alpha_vantage=test_key\nother_source=other_key\n')
    return keys_file

def test_init_creates_config_dir(test_config_dir):
    """Test that initialization creates config directory"""
    APIConfigHandler(config_dir=test_config_dir)
    assert os.path.exists(test_config_dir)

def test_load_config_creates_default(api_handler):
    """Test that default configuration is created if none exists"""
    config = api_handler.config
    assert 'alpha_vantage' in config
    assert config['alpha_vantage']['daily_calls'] == 0
    assert config['alpha_vantage']['daily_limit'] == 25

def test_get_api_key(api_handler, setup_api_keys):
    """Test getting API key from file"""
    assert api_handler.get_api_key('alpha_vantage') == 'test_key'
    assert api_handler.get_api_key('other_source') == 'other_key'
    assert api_handler.get_api_key('nonexistent') is None

def test_can_make_api_call(api_handler):
    """Test API call availability checking"""
    assert api_handler.can_make_api_call('alpha_vantage') == True
    
    # Simulate reaching the limit
    api_handler.config['alpha_vantage']['daily_calls'] = 25
    assert api_handler.can_make_api_call('alpha_vantage') == False

def test_record_api_call(api_handler):
    """Test recording API calls"""
    initial_calls = api_handler.config['alpha_vantage']['daily_calls']
    api_handler.record_api_call('alpha_vantage')
    assert api_handler.config['alpha_vantage']['daily_calls'] == initial_calls + 1

def test_daily_reset(api_handler):
    """Test daily call counter reset"""
    # Set last reset to yesterday
    yesterday = str(date.today() - timedelta(days=1))
    api_handler.config['alpha_vantage']['last_reset'] = yesterday
    api_handler.config['alpha_vantage']['daily_calls'] = 20
    
    # Check if counter resets on next call
    assert api_handler.can_make_api_call('alpha_vantage') == True
    api_handler.record_api_call('alpha_vantage')
    assert api_handler.config['alpha_vantage']['daily_calls'] == 1
    assert api_handler.config['alpha_vantage']['last_reset'] == str(date.today())

def test_get_remaining_calls(api_handler):
    """Test getting remaining API calls"""
    api_handler.config['alpha_vantage']['daily_calls'] = 15
    assert api_handler.get_remaining_calls('alpha_vantage') == 10  # 25 - 15

def test_config_persistence(test_config_dir):
    """Test that configuration persists between handler instances"""
    handler1 = APIConfigHandler(test_config_dir)
    handler1.record_api_call('alpha_vantage')
    
    handler2 = APIConfigHandler(test_config_dir)
    assert handler2.config['alpha_vantage']['daily_calls'] == 1

def test_invalid_api_keys_file(api_handler):
    """Test handling of invalid API keys file"""
    # Create invalid file format
    keys_file = Path(api_handler.api_keys_file)
    keys_file.parent.mkdir(exist_ok=True)
    keys_file.write_text('invalid format')
    
    assert api_handler.get_api_key('alpha_vantage') is None
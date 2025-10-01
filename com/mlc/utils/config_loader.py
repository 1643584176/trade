"""
配置文件加载器
用于统一加载和管理项目配置
"""

import os


def load_config():
    """
    从account_config.txt文件加载配置
    
    Returns:
        dict: 配置参数字典
    """
    config = {}
    # 获取配置文件路径（在项目根目录）
    config_file_path = os.path.join(os.path.dirname(__file__), '..', 'account_config.txt')
    config_file_path = os.path.abspath(config_file_path)
    
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过空行和注释行
                    if '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
    else:
        print(f"警告: 配置文件不存在: {config_file_path}")
        # 使用默认配置
        config = get_default_config()
    
    return config


def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置参数字典
    """
    return {
        # 测试账户配置 (TEST_ACCOUNT)
        'TEST_ACCOUNT_ENABLED': 'True',
        'TEST_ACCOUNT_NUMBER': '123456',
        'TEST_ACCOUNT_PASSWORD': 'password1',
        'TEST_ACCOUNT_SERVER': 'server1',
        
        # 实盘账户配置 (REAL_ACCOUNT)
        'REAL_ACCOUNT_ENABLED': 'True',
        'REAL_ACCOUNT_NUMBER': '789012',
        'REAL_ACCOUNT_PASSWORD': 'password2',
        'REAL_ACCOUNT_SERVER': 'server2',
        
        # 通用账户配置（用于单账户文件，默认使用测试账户）
        'ACCOUNT_NUMBER': '123456',
        'ACCOUNT_PASSWORD': 'password1',
        'ACCOUNT_SERVER': 'server1',
        
        # 通用交易参数
        'TRADE_SYMBOL': 'XAUUSD',
        'LOT_SIZE': '1.0',
        
        # GBPUSD配置
        'GBPUSD_SYMBOL': 'GBPUSD',
        'GBPUSD_LOT_SIZE': '1.0',
        
        # EURUSD配置
        'EURUSD_SYMBOL': 'EURUSD',
        'EURUSD_LOT_SIZE': '2.0',
        
        # USDJPY配置
        'USDJPY_SYMBOL': 'USDJPY',
        'USDJPY_LOT_SIZE': '1.0',
        
        # AUDUSD配置
        'AUDUSD_SYMBOL': 'AUDUSD',
        'AUDUSD_LOT_SIZE': '1.0'
    }


# 全局配置变量
config = load_config()


def get_config_value(key, default_value=None):
    """
    获取配置值，如果配置文件中没有则返回默认值
    
    Args:
        key (str): 配置键
        default_value (str): 默认值
        
    Returns:
        str: 配置值或默认值
    """
    return config.get(key, default_value)


def get_test_account():
    """
    获取测试账户配置
    
    Returns:
        dict: 测试账户配置信息
    """
    return {
        'enabled': get_config_value('TEST_ACCOUNT_ENABLED', 'True').lower() == 'true',
        'number': get_config_value('TEST_ACCOUNT_NUMBER', '123456'),
        'password': get_config_value('TEST_ACCOUNT_PASSWORD', 'password1'),
        'server': get_config_value('TEST_ACCOUNT_SERVER', 'server1')
    }


def get_real_account():
    """
    获取实盘账户配置
    
    Returns:
        dict: 实盘账户配置信息
    """
    return {
        'enabled': get_config_value('REAL_ACCOUNT_ENABLED', 'True').lower() == 'true',
        'number': get_config_value('REAL_ACCOUNT_NUMBER', '789012'),
        'password': get_config_value('REAL_ACCOUNT_PASSWORD', 'password2'),
        'server': get_config_value('REAL_ACCOUNT_SERVER', 'server2')
    }
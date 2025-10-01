"""
MT5连接器模块
提供统一的MT5连接方法，供各个交易策略模块调用
"""

import MetaTrader5 as mt5
from com.mlc.utils.config_loader import get_config_value, get_test_account


def connect_to_mt5(account_type="real"):
    """
    连接到MT5平台的公共方法
    
    Args:
        account_type (str): 账户类型，"real"表示实盘账户，"demo"表示测试账户
        
    Returns:
        bool: 连接是否成功
    """
    try:
        # 获取账户参数
        if account_type == "demo":
            # 使用测试账户
            account_number = get_config_value('TEST_ACCOUNT_NUMBER', '')
            if account_number:
                # 获取测试账户完整信息
                test_account_info = get_test_account()
                if not mt5.initialize(
                    login=int(account_number),
                    password=test_account_info['password'],
                    server=test_account_info['server']
                ):
                    print(f"MT5初始化失败，测试账户: {account_number}")
                    return False
            else:
                # 如果没有配置测试账户，则使用默认初始化
                if not mt5.initialize():
                    print("MT5初始化失败")
                    return False
        else:
            # 使用实盘账户
            account_number = get_config_value('REAL_ACCOUNT_NUMBER', '')
            if account_number:
                # 获取实盘账户完整信息
                real_account_info = get_test_account()  # 假设这里也有获取实盘账户的方法
                if not mt5.initialize(
                    login=int(account_number),
                    password=real_account_info['password'],
                    server=real_account_info['server']
                ):
                    print(f"MT5初始化失败，实盘账户: {account_number}")
                    return False
            else:
                # 如果没有配置实盘账户，则使用默认初始化
                if not mt5.initialize():
                    print("MT5初始化失败")
                    return False

        # 检查连接状态
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            print("MT5连接失败或未连接")
            return False

        print("MT5连接成功！")
        if account_type == "demo":
            server = get_config_value('TEST_ACCOUNT_SERVER', '')
            account_number = get_config_value('TEST_ACCOUNT_NUMBER', '')
            print(f"当前连接账户: {account_number}@{server}")
        else:
            server = get_config_value('REAL_ACCOUNT_SERVER', '')
            account_number = get_config_value('REAL_ACCOUNT_NUMBER', '')
            print(f"当前连接账户: {account_number}@{server}")
            
        return True
        
    except Exception as e:
        print(f"连接MT5时发生错误: {str(e)}")
        return False


def disconnect_mt5():
    """
    断开MT5连接
    """
    try:
        mt5.shutdown()
        print("MT5连接已断开")
    except Exception as e:
        print(f"断开MT5连接时发生错误: {str(e)}")


def check_symbol_available(symbol):
    """
    检查交易品种是否可用
    
    Args:
        symbol (str): 交易品种名称
        
    Returns:
        bool: 交易品种是否可用
    """
    try:
        # 检查交易品种
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"交易品种 {symbol} 不存在")
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"无法选择交易品种 {symbol}")
                return False
                
        print(f"交易品种{symbol}已就绪")
        return True
        
    except Exception as e:
        print(f"检查交易品种时发生错误: {str(e)}")
        return False
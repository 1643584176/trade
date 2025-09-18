import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import MetaTrader5 as mt5

def initialize_mt5():
    """
    初始化MT5连接
    """
    if not mt5.initialize():
        print(f"MT5初始化失败，错误代码: {mt5.last_error()}")
        return False
    print("MT5初始化成功")
    return True

def get_historical_data(symbol, timeframe, count=100):
    """
    获取历史数据用于指标计算
    :param symbol: 交易品种
    :param timeframe: 时间周期
    :param count: 获取的历史数据条数
    :return: DataFrame格式的历史数据
    """
    # 请求数据
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"获取历史数据失败，错误代码: {mt5.last_error()}")
        return None

    # 转换为DataFrame并处理
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 添加必要列
    df['open'] = df['open']
    df['high'] = df['high']
    df['low'] = df['low']
    df['close'] = df['close']
    df['tick_volume'] = df['tick_volume']
    
    # 计算技术指标
    # 计算ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = pd.Series(true_range).rolling(14).mean()
    
    # 计算RSI指标
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df.dropna().reset_index(drop=True)

def generate_signals(data):
    """
    生成交易信号
    """
    signals = []
    for i in range(len(data)):
        current = data.iloc[i]
        
        # 简单的入场条件
        if current['RSI'] < 30:
            signals.append(1)  # 买入信号
        elif current['RSI'] > 70:
            signals.append(-1)  # 卖出信号
        else:
            signals.append(0)  # 无信号
    return signals

def execute_trade(symbol, direction, lot_size=0.1):
    """
    执行交易
    """
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_BUY if direction > 0 else mt5.ORDER_SELL,
        "price": mt5.symbol_info_tick(symbol).ask if direction > 0 else mt5.symbol_info_tick(symbol).bid,
        "deviation": 20,
        "magic": 123456,
        "comment": "Real-time Strategy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"订单执行失败，错误代码: {result.retcode}")
        return False
    print(f"订单执行成功: {result.order}")
    return True

def run_strategy():
    """
    运行实时交易策略
    """
    # 初始化MT5
    if not initialize_mt5():
        return
    
    # 获取历史数据
    data = get_historical_data("XAUUSD", mt5.TIMEFRAME_H1, 100)
    if data is None or len(data) == 0:
        print("无法获取足够的历史数据")
        return
    
    # 生成信号
    signals = generate_signals(data)
    
    # 检查最新信号
    latest_signal = signals[-1]
    if latest_signal != 0:
        print(f"发现交易信号: {'买入' if latest_signal > 0 else '卖出'}")
        execute_trade("XAUUSD", latest_signal)
    else:
        print("当前无交易信号")
    
    # 断开MT5连接
    mt5.shutdown()

if __name__ == "__main__":
    run_strategy()
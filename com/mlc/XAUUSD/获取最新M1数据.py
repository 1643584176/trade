import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def get_latest_m1_data(symbol="XAUUSD", count=5):
    """
    获取指定品种的最新M1数据
    
    Args:
        symbol (str): 交易品种，默认为"XAUUSD"
        count (int): 获取的数据条数，默认为5
        
    Returns:
        pandas.DataFrame: 包含最新M1数据的DataFrame
    """
    # 检查MT5连接状态
    if not mt5.initialize():
        print("MT5初始化失败")
        return None
    
    # 检查终端连接状态
    terminal_info = mt5.terminal_info()
    if terminal_info is None or not terminal_info.connected:
        print("MT5终端未连接")
        return None

    try:
        # 确保品种在市场观察列表中
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            print(f"无法选择交易品种: {symbol}")
            return None
            
        # 检查品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"无法获取交易品种信息: {symbol}")
            return None

        # 获取最新的M1 K线数据
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        
        if rates is None or len(rates) == 0:
            print("未获取到历史数据")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 重命名列
        df.rename(columns={
            'time': 'MT5时间',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价',
            'close': '收盘价',
            'tick_volume': '成交量'
        }, inplace=True)
        
        # 选择需要的列
        result = df[['MT5时间', '开盘价', '最高价', '最低价', '收盘价', '成交量']]
        
        return result

    except Exception as e:
        print(f"获取数据时发生错误: {str(e)}")
        return None
    finally:
        # 注意：不关闭MT5连接，因为可能其他地方还需要使用
        pass

def main():
    """主函数"""
    print("正在查询XAUUSD最新5条M1数据...")
    
    # 获取数据
    data = get_latest_m1_data("XAUUSD", 5)
    
    if data is not None and not data.empty:
        print(f"\nXAUUSD 最新5条M1数据:")
        print("=" * 100)
        # 显示表头
        print(f"{'MT5时间':<20} {'北京时间':<20} {'开盘价':>8} {'最高价':>8} {'最低价':>8} {'收盘价':>8} {'成交量':>8}")
        print("-" * 100)
        
        # 格式化输出，保留小数点后2位
        for index, row in data.iterrows():
            # 计算北京时间（MT5时间+5小时）
            beijing_time = row['MT5时间'] + timedelta(hours=5)
            print(f"{row['MT5时间'].strftime('%Y-%m-%d %H:%M:%S'):<20} "
                  f"{beijing_time.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                  f"{row['开盘价']:>8.2f} {row['最高价']:>8.2f} {row['最低价']:>8.2f} {row['收盘价']:>8.2f} {row['成交量']:>8}")
        
        print("\n时区说明：MT5时间与北京时间相差5小时")
    else:
        print("未能获取到XAUUSD M1数据")
        
    # 等待用户查看结果
    input("\n按回车键退出...")

# 执行程序
if __name__ == "__main__":
    main()
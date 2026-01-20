import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import os


def 查询最新m1时间():
    """
    查询XAUUSD M1（1分钟）最新K线的时间
    """
    # 1. 初始化并连接MT5客户端
    if not mt5.initialize():
        print(f"MT5初始化失败，错误代码：{mt5.last_error()}")
        mt5.shutdown()
        return None

    # 2. 检查XAUUSD品种是否在市场报价中
    symbol = "XAUUSD"
    if not mt5.symbol_select(symbol, True):  # True表示如果未选中则自动选中
        print(f"无法找到品种 {symbol}，错误代码：{mt5.last_error()}")
        mt5.shutdown()
        return None

    # 3. 获取M1周期的最新1条K线数据
    timeframe = mt5.TIMEFRAME_M1  # M1时间周期
    n = 1  # 获取最新1条数据
    rates_kline = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)

    # 4. 关闭MT5连接
    mt5.shutdown()

    # 5. 检查数据是否获取成功
    if rates_kline is None or len(rates_kline) == 0:
        print("未获取到XAUUSD M1数据")
        return None

    # 6. 提取时间并返回（使用原始时间，不做转换）
    timestamp = rates_kline[0][0]  # 时间戳在第一个元素
    dt = datetime.utcfromtimestamp(timestamp)  # 使用UTC时间

    return dt


def 获取最近60天m1数据(save_csv=False, output_dir="m1_trend_analysis_results"):
    """
    获取XAUUSD最近60天的M1（1分钟）历史数据，并转换为UTC+2时区
    :param save_csv: 是否保存为CSV文件，默认False
    :param output_dir: CSV输出目录，默认为m1_trend_analysis_results
    :return: 带UTC+2时间索引的DataFrame，None表示失败
    """
    # 初始化MT5连接
    if not mt5.initialize():
        print(f"MT5初始化失败，错误代码：{mt5.last_error()}")
        return None

    # 定义时区：UTC+2
    utc2_tz = pytz.FixedOffset(120)  # UTC+2 = 120分钟偏移

    # 计算时间范围（基于UTC+2时区的"过去60天"）
    now_utc2 = datetime.now(utc2_tz)
    end_time_utc2 = now_utc2.replace(second=0, microsecond=0)  # 取整到分钟
    start_time_utc2 = end_time_utc2 - timedelta(days=60)

    # 转换为UTC时间（MT5接口要求UTC时间戳）
    start_time_utc = start_time_utc2.astimezone(pytz.UTC)
    end_time_utc = end_time_utc2.astimezone(pytz.UTC)

    # 转为MT5支持的时间戳（秒级）
    start_ts = int(start_time_utc.timestamp())
    end_ts = int(end_time_utc.timestamp())

    # 获取M1（1分钟）数据
    rates = mt5.copy_rates_range(
        "XAUUSD",
        mt5.TIMEFRAME_M1,
        start_ts,
        end_ts
    )

    # 关闭MT5连接
    mt5.shutdown()

    # 数据校验
    if rates is None or len(rates) == 0:
        print(f"获取XAUUSD M1数据失败，错误代码：{mt5.last_error()}")
        return None

    # 数据格式化与时区转换
    df = pd.DataFrame(rates)
    # 1. 先将MT5的UTC时间戳转为UTC时区的datetime
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # 2. 转换为UTC+2时区
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    # 保持UTC+2时区的时间，但不带时区信息
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=2)

    # 3. 保留核心字段
    df = df[["timestamp", "open", "high", "low", "close", "tick_volume", "spread"]]

    # 添加星期信息
    df['weekday'] = df['timestamp'].dt.weekday
    # 将英文星期名称转换为中文
    weekday_map = {
        'Monday': '星期一',
        'Tuesday': '星期二', 
        'Wednesday': '星期三',
        'Thursday': '星期四',
        'Friday': '星期五',
        'Saturday': '星期六',
        'Sunday': '星期日'
    }
    df['week_day_name'] = df['timestamp'].dt.day_name().map(weekday_map)
    
    # 添加价格变化指标
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
    df['range_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # 添加移动平均线 (用于趋势判断)
    df['ema_fast'] = df['close'].ewm(span=5).mean()  # 5周期快速EMA
    df['ema_slow'] = df['close'].ewm(span=20).mean()  # 20周期慢速EMA
    
    # 添加波动率指标
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # 添加相对强弱指标 (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 根据项目规范过滤周末数据，仅保留周一至周五
    df = df[df['weekday'] < 5]  # 0-4 代表周一到周五
    
    # 将数值列四舍五入到合适的小数位数
    numeric_columns = ['open', 'high', 'low', 'close', 'price_change', 'price_change_pct', 'range_size', 
                     'upper_shadow', 'lower_shadow', 'ema_fast', 'ema_slow', 'volatility', 'rsi']
    for col in numeric_columns:
        if col in df.columns:
            if col in ['rsi']:
                df[col] = df[col].round(2)  # RSI保留2位小数
            elif col in ['price_change_pct']:
                df[col] = df[col].round(4)  # 百分比保留4位小数
            else:
                df[col] = df[col].round(2)  # 其他价格相关数据保留2位小数
    
    # 交易量通常是整数
    if 'tick_volume' in df.columns:
        df['tick_volume'] = df['tick_volume'].astype(int)
    
    # 根据参数决定是否保存为CSV
    if save_csv:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/m1_raw_data_{timestamp}.csv"
        
        try:
            # 保存数据到CSV（包含星期和交易时段信息）
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ M1原始数据已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存CSV失败: {str(e)}")
    
    return df


# 测试函数
if __name__ == "__main__":
    # latest_time = 查询最新m1时间()
    # if latest_time is not None:
    #     print(f"最新M1 K线时间（UTC）：{latest_time}")

    # 测试获取最近60天的数据
    print(f"\n" + "=" * 50)
    print("测试获取最近60天数据：")
    data_60d = 获取最近60天m1数据(save_csv=False)  # 测试保存功能
    if data_60d is not None:
        print(f"数据形状：{data_60d.shape}")
        print("后5行数据：")
        print(data_60d.tail())
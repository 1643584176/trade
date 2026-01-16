import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz


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


def 获取最近60天m1数据():
    """
    获取XAUUSD最近60天的M1（1分钟）历史数据，并转换为UTC+2时区
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
    # 3. 设置UTC+2时间为索引，保留核心字段
    df = df.set_index("timestamp")
    df = df[["open", "high", "low", "close", "tick_volume"]]

    return df


# 测试函数
if __name__ == "__main__":
    latest_time = 查询最新m1时间()
    if latest_time is not None:
        print(f"最新M1 K线时间（UTC）：{latest_time}")

    # 测试获取最近60天的数据
    print(f"\n" + "=" * 50)
    print("测试获取最近60天数据：")
    data_60d = 获取最近60天m1数据()
    if data_60d is not None:
        print("后5行数据：")
        print(data_60d.tail())
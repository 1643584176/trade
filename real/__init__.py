import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def fetch_m1_data_for_period( days_back=60):
    """获取过去指定天数的M1数据"""

    # 初始化MT5连接
    if not mt5.initialize():
        print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
        return None

    # 检查交易品种
    symbol = "XAUUSD"

    # 计算时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    # 获取M1数据
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)

    if rates is None or len(rates) == 0:
        print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
        print(f"💡 提示: 请检查MT5终端是否开启，以及XAUUSD品种是否可用")
        mt5.shutdown()
        return None

    # 转换为DataFrame
    df = pd.DataFrame(rates)

    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')

    # 转换为UTC+2时区（您指定的时区）
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=2)

    # 重命名列
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'spread': 'spread'
    })

    # 选择需要的列
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]

    print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

    # 检查数据的连续性
    time_diffs = df['timestamp'].diff().dropna()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]  # 超过5分钟的间隙
    if len(gaps) > 0:
        # print(f"⚠️  检测到 {len(gaps)} 个数据间隙，可能影响趋势分析")
        # 显示最大的几个间隙
        largest_gaps = gaps.nlargest(min(3, len(gaps)))  # 确保不超过间隙总数
        # for idx, gap in largest_gaps.items():
        #     if idx < len(df):  # 确保索引有效
        #         print(f"   间隙: {gap} 在 {df['timestamp'].iloc[idx]} 附近")

    # 不再导出原始数据到CSV文件
    # print(f"✅ 原始数据处理完成")

    # 断开MT5连接
    mt5.shutdown()

    return df

def get_xauusd_m1_last_2h_utc2():
    """
    获取XAUUSD最近2小时的M1数据，时间转换为UTC+2并格式化展示
    :return: DataFrame（包含时间(UTC+2)、开盘、最高、最低、收盘、成交量）
    """
    # 1. 初始化MT5连接
    if not mt5.initialize():
        print(f"❌ MT5初始化失败: {mt5.last_error()}")
        return None

    # 2. 验证XAUUSD品种
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not symbol_info.visible:
        print(f"⚠️ 品种{symbol}不可用或未激活，请添加到市场观察列表")
        mt5.shutdown()
        return None

    # 3. 计算数据范围：最近2小时 = 120根M1 K线（额外多取5根，避免数据缺失）
    num_candles = 120 + 5
    # 获取M1数据（MT5返回的时间是UTC时间）
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, num_candles)


    if rates is None or len(rates) == 0:
        print(f"❌ 错误: 未获取到过去 {1} 天的M1数据")
        print(f"💡 提示: 请检查MT5终端是否开启，以及XAUUSD品种是否可用")
        mt5.shutdown()
        return None


    # 4. 转换为DataFrame并处理时间（UTC → UTC+2）
    df = pd.DataFrame(rates)
    # MT5的time列是时间戳，先转换为UTC datetime
    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')

    # 转换为UTC+2时区（您指定的时区）
    df['timestamp'] = df['timestamp']


    # 5. 整理数据列（保留核心字段，适配XAUUSD价格精度）
    result_df = df[['timestamp', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
    # 重命名列，更易读
    result_df.rename(
        columns={
            'timestamp': '时间',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价',
            'close': '收盘价',
            'tick_volume': '成交量'
        },
        inplace=True
    )
    # 保留价格小数点后3位（XAUUSD的标准精度）
    price_cols = ['开盘价', '最高价', '最低价', '收盘价']
    result_df[price_cols] = result_df[price_cols].round(3)

    # 6. 只保留最近2小时（120根）数据，剔除多余的5根

    # 7. 断开MT5连接
    mt5.shutdown()

    return result_df

def get_short_term_trend( num_candles=5):
        """
        基于N根M1均线判断短期趋势，优化版：适合检测短期回调和趋势变化
        :param num_candles: 均线周期（默认5，即5M均线）
        :return: 1 上涨趋势，0 下跌趋势，-1 横盘
        """
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return -1  # 返回横盘状态

        # 检查交易品种
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            print(f"⚠️ 品种{symbol}不可用或未激活")
            mt5.shutdown()
            return -1

        # 获取适量的数据，重点关注近期变化
        need_rates = max(num_candles * 2 + 10, 20)  # 获取2个周期+额外数据，共20根
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, need_rates)

        if rates is None or len(rates) < need_rates:
            print(f"⚠️  未获取到足够的M1数据({need_rates}根)，实际获取{len(rates) if rates is not None else 0}根")
            mt5.shutdown()
            return -1

        # 提取收盘价数组
        closes = np.array([rate['close'] for rate in rates])

        # 使用更短周期对比，更快响应短期变化
        recent_3 = closes[:3]  # 最近3根K线
        previous_3 = closes[3:6]  # 之前3根K线

        recent_ma = np.mean(recent_3)
        previous_ma = np.mean(previous_3)

        # 计算短期价格变化率
        change_rate = (recent_ma - previous_ma) / previous_ma

        # 设置较小的阈值以检测短期变化
        threshold = 0.0002  # 0.02%的阈值

        # 主要判断逻辑：短期均值变化
        if change_rate > threshold:
            trend = 1  # 上涨趋势
        elif change_rate < -threshold:
            trend = 0  # 下跌趋势
        else:
            # 使用价格动量判断短期趋势
            # 比较最近价格与稍早价格
            if len(closes) >= 6:
                # 检查最近1根 vs 3根前的价格
                latest_price = closes[0]
                price_3_bars_ago = closes[3]

                mom_change = (latest_price - price_3_bars_ago) / price_3_bars_ago

                if mom_change > threshold:
                    trend = 1
                elif mom_change < -threshold:
                    trend = 0
                else:
                    # 再次检查最近几根K线的高低点变化
                    if len(closes) >= 5:
                        highest_recent = max(closes[:3])
                        lowest_recent = min(closes[:3])
                        highest_prev = max(closes[3:6])
                        lowest_prev = min(closes[3:6])

                        # 如果最近高点比之前高点高，且最近低点比之前低点高，则为上涨
                        if highest_recent > highest_prev and lowest_recent >= lowest_prev:
                            trend = 1
                        # 如果最近低点比之前低点低，且最近高点比之前高点低，则为下跌
                        elif lowest_recent < lowest_prev and highest_recent <= highest_prev:
                            trend = 0
                        else:
                            trend = -1  # 横盘
                    else:
                        trend = -1  # 横盘

        # 断开MT5连接
        mt5.shutdown()

        return trend
# ------------------- 测试调用 -------------------
if __name__ == "__main__":
    # 获取数据
    xauusd_m1_data = get_short_term_trend()
    print(xauusd_m1_data)
    #
    # if xauusd_m1_data is not None:
    #     print("✅ XAUUSD最近2小时M1数据（UTC+2）：")
    #     print(xauusd_m1_data)
    #
    #     # 可选：保存为CSV文件，方便后续分析
    #     xauusd_m1_data.to_csv('XAUUSD_M1_Last2H_UTC2.csv', index=False, encoding='utf-8')
    #     print("\n📁 数据已保存为 XAUUSD_M1_Last2H_UTC2.csv")
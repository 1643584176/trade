import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

# 根据数据生个交易策略,需要复合ftmo规则,每天至少交易三次,起手为3手,需要分析历史数据来做策略,注意昨日和今日的对比,注意星期的对比,注意周五的特殊性,注意周一周二关联性,注意周三的反转,注意周四周五的连续,注意大趋势,注意开盘价和当前价格的关系,注意你在当前日期时间点看不到后面的数据,,注意可以可以根据最高价最低价预测进场点位,注意止盈止损

#根据数据生个交易策略,需要复合ftmo规则,每天至少交易三次,起手为3手,需要分析历史数据来做策略,注意昨日和今日的对比,注意星期的对比,注意周五的特殊性,注意周一周二关联性,注意周三的反转,注意周四周五的连续,注意大趋势,注意开盘价和当前价格的关系,注意你在当前日期时间点看不到后面的数据,,注意可以可以根据最高价最低价预测进场点位,注意止盈止损,不要根据平均值来计算涨跌,这种概率都是50%,你参考其他的比如开盘价和当前价位关系,或者其他的都可以,买涨跌都可以,不是非要一个方向

def get_timeframe(timeframe_str):
    """根据字符串返回对应的MT5时间周期"""
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    return timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M30)


def parse_user_input(user_input):
    """解析用户输入，格式如: XAUUSD-M30"""
    try:
        parts = user_input.split('-')
        if len(parts) != 2:
            raise ValueError("输入格式错误")

        symbol = parts[0].upper()
        timeframe_str = parts[1].upper()
        timeframe = get_timeframe(timeframe_str)

        return symbol, timeframe, timeframe_str
    except Exception as e:
        print(f"解析输入时出错: {e}")
        return "XAUUSD", mt5.TIMEFRAME_M30, "M30"


def get_historical_data(symbol, timeframe, timeframe_str):
    """获取历史数据"""
    # 初始化MT5连接
    if not mt5.initialize():
        print("MT5初始化失败")
        return None

    try:
        # 确保品种在市场观察列表中
        selected = mt5.symbol_select(symbol, True)
        if not selected:
            print(f"无法选择交易品种: {symbol}")
            return None

        # 设置时间范围 - 从2024年1月1日到当前时间
        from_date = datetime(2024, 1, 1)
        to_date = datetime.now()

        # 获取历史数据
        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

        if rates is None or len(rates) == 0:
            print("未获取到历史数据")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # 按日期分组计算每日数据
        df['date'] = df['time'].dt.date
        daily_data = df.groupby('date').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min'
        }).reset_index()

        # 计算每日的增减和波动幅度
        daily_data['daily_change'] = daily_data['close'] - daily_data['open']  # 当日增减（可正可负）
        daily_data['daily_range'] = daily_data['high'] - daily_data['low']  # 当日波动幅度（始终为正）

        # 添加昨日数据（向前移动一行）
        daily_data['prev_daily_change'] = daily_data['daily_change'].shift(1)  # 昨天增减（前一日的增减，可正可负）
        daily_data['prev_daily_range'] = daily_data['daily_range'].shift(1)  # 昨日波动幅度（前一日的波动幅度，始终为正）

        # 将昨日数据合并到分钟数据中
        df = df.merge(daily_data[['date', 'prev_daily_change', 'prev_daily_range']],
                      on='date', how='left')

        # 添加星期几
        weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        df['星期几'] = df['time'].dt.weekday.apply(lambda x: weekday_names[x])

        # 计算当前K线的波动幅度
        df['current_range'] = df['high'] - df['low']  # 当前波动幅度（始终为正）

        # 计算当前K线的增减（收盘价-开盘价，可正可负）
        df['current_change'] = df['close'] - df['open']  # 当前增减（可正可负）

        # 重命名列以符合要求
        df.rename(columns={
            'time': '时间点',
            'open': '当前开盘价',
            'close': '当前收盘价',
            'high': '当前最高价',
            'low': '当前最低价',
            'prev_daily_change': '昨天增减',
            'prev_daily_range': '昨日波动幅度',
            'current_change': '当前增减',
            'current_range': '当前波动幅度'
        }, inplace=True)

        # 选择需要的列并格式化数值（保留两位小数）
        result = df[[
            '时间点', '星期几', '昨天增减', '昨日波动幅度',
            '当前开盘价', '当前收盘价', '当前最高价', '当前最低价', '当前增减', '当前波动幅度'
        ]].round(6)
        
        # 过滤掉2024年1月2日的数据
        result = result[result['时间点'].dt.date != datetime(2024, 1, 2).date()]

        return result

    except Exception as e:
        print(f"获取数据时发生错误: {str(e)}")
        return None
    finally:
        # 关闭MT5连接
        mt5.shutdown()


def main():
    # 获取用户输入
    user_input = input("请输入查询参数 (格式: 品种-周期，例如: XAUUSD-M30): ")

    if not user_input:
        user_input = "XAUUSD-M30"  # 默认值

    # 解析用户输入
    symbol, timeframe, timeframe_str = parse_user_input(user_input)

    print(f"正在查询 {symbol} {timeframe_str} 历史数据...")

    # 获取数据
    data = get_historical_data(symbol, timeframe, timeframe_str)

    if data is not None and not data.empty:
        print(f"成功获取 {len(data)} 条记录")

        # 显示前20行数据
        print(f"\n{symbol} {timeframe_str} 历史数据 (前20行):")
        print("=" * 130)
        print(data.head(20).to_string(index=False))

        # 显示后20行数据
        print(f"\n{symbol} {timeframe_str} 历史数据 (后20行):")
        print("=" * 130)
        print(data.tail(20).to_string(index=False))

        # 查看一些有完整数据的示例（跳过前几行NaN值）
        valid_data = data.dropna()
        if len(valid_data) > 0:
            print(f"\n{symbol} {timeframe_str} 有完整数据的示例 (前10行):")
            print("=" * 130)
            print(valid_data.head(10).to_string(index=False))

            # 统计信息
            print("\n=== 数据统计 ===")
            print(f"完整数据记录数: {len(valid_data)}")
            print(f"昨天增减平均值: {valid_data['昨天增减'].mean():.2f}")
            print(f"昨天增减标准差: {valid_data['昨天增减'].std():.2f}")
            print(f"昨日波动幅度平均值: {valid_data['昨日波动幅度'].mean():.2f}")
            print(f"当前增减平均值: {valid_data['当前增减'].mean():.2f}")
            print(f"当前波动幅度平均值: {valid_data['当前波动幅度'].mean():.2f}")

        # 保存到CSV文件
        filename = f'{symbol}_{timeframe_str}_历史数据.csv'
        data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到 {filename}")
    else:
        print(f"未能获取到 {symbol} {timeframe_str} 历史数据")


# 执行程序
if __name__ == "__main__":
    main()
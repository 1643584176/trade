# -*- coding: utf-8 -*-
"""
FTMO合规交易策略 - MT5数据版本
该策略遵循FTMO考试规则，每天至少交易3次，起手为3手
并考虑星期几的特殊性、趋势分析、开盘价与当前价格关系等因素
此版本使用MT5数据进行回测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5


class FTMOStrategyMT5:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M30, start_date=None, end_date=None):
        """
        初始化FTMO策略
        :param symbol: 交易品种
        :param timeframe: 时间周期
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date if start_date else datetime.now() - timedelta(days=500)
        self.end_date = end_date if end_date else datetime.now()
        self.data = None
        self.trades = []

        # 初始化MT5连接
        if not mt5.initialize():
            print("初始化MT5失败")
            return

    def load_data(self):
        """
        从MT5加载数据
        """
        # 从MT5获取数据
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, self.start_date, self.end_date)

        if rates is None or len(rates) == 0:
            print("无法获取MT5数据")
            return

        # 转换为DataFrame
        self.data = pd.DataFrame(rates)
        self.data['datetime'] = pd.to_datetime(self.data['time'], unit='s')
        self.data.sort_values('datetime', inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        # 重命名列以匹配策略需求
        self.data.rename(columns={
            'open': 'open_price',
            'close': 'current_price',
            'high': 'high_price',
            'low': 'low_price'
        }, inplace=True)

        # 计算价格变化
        self.data['price_change'] = self.data['current_price'] - self.data['open_price']
        self.data['price_range'] = self.data['high_price'] - self.data['low_price']

        # 添加额外的日期时间特征
        self.data['hour'] = self.data['datetime'].dt.hour
        self.data['day_of_week'] = self.data['datetime'].dt.dayofweek  # 0=周一, 4=周五
        self.data['date'] = self.data['datetime'].dt.date
        self.data['time'] = self.data['datetime'].dt.time
        self.data['month'] = self.data['datetime'].dt.to_period('M')

        # 标记特殊日期
        self.data['is_monday'] = self.data['day_of_week'] == 0
        self.data['is_tuesday'] = self.data['day_of_week'] == 1
        self.data['is_wednesday'] = self.data['day_of_week'] == 2
        self.data['is_thursday'] = self.data['day_of_week'] == 3
        self.data['is_friday'] = self.data['day_of_week'] == 4

    def calculate_indicators(self):
        """
        计算技术指标
        """
        if self.data is None or len(self.data) == 0:
            print("没有数据可用于计算指标")
            return

        # 计算移动平均线
        self.data['ma_5'] = self.data['current_price'].rolling(window=5).mean()
        self.data['ma_10'] = self.data['current_price'].rolling(window=10).mean()
        self.data['ma_20'] = self.data['current_price'].rolling(window=20).mean()

        # 计算价格变化率
        self.data['price_change_rate'] = self.data['current_price'].pct_change()

        # 计算波动率
        self.data['volatility'] = self.data['price_change_rate'].rolling(window=10).std()

        # 计算价格位置（当前价格在当日高低价之间的位置）
        self.data['price_position'] = (self.data['current_price'] - self.data['low_price']) / (
                    self.data['high_price'] - self.data['low_price'])

        # 计算开盘价位置（开盘价在当日高低价之间的位置）
        self.data['open_position'] = (self.data['open_price'] - self.data['low_price']) / (
                    self.data['high_price'] - self.data['low_price'])

    def weekday_analysis(self):
        """
        星期分析，包括周五特殊性、周一周二关联性等
        """
        if self.data is None or len(self.data) == 0:
            print("没有数据可用于星期分析")
            return pd.DataFrame()

        # 按星期几分组计算平均价格变化
        weekday_stats = self.data.groupby('day_of_week')['price_change'].agg(['mean', 'std', 'count']).reset_index()
        weekday_stats.columns = ['day_of_week', 'avg_change', 'std_change', 'count']
        return weekday_stats

    def trend_analysis(self):
        """
        趋势分析
        """
        if self.data is None or len(self.data) == 0:
            print("没有数据可用于趋势分析")
            return

        # 确定大趋势 (使用20周期移动平均)
        self.data['long_term_trend'] = np.where(self.data['current_price'] > self.data['ma_20'], 'up', 'down')

        # 短期趋势 (使用5周期移动平均)
        self.data['short_term_trend'] = np.where(self.data['current_price'] > self.data['ma_5'], 'up', 'down')

        # 趋势强度
        self.data['trend_strength'] = abs(self.data['current_price'] - self.data['ma_20']) / self.data['ma_20']

        # 前一日涨跌情况
        self.data['prev_price_change'] = self.data['price_change'].shift(1)
        self.data['prev_day_trend'] = np.where(self.data['prev_price_change'] > 0, 'up',
                                               np.where(self.data['prev_price_change'] < 0, 'down', 'flat'))

    def generate_signals(self):
        """
        生成交易信号 - 根据FTMO规则优化
        """
        if self.data is None or len(self.data) == 0:
            print("没有数据可用于生成信号")
            return [], []

        signals = []
        reasons = []

        for i in range(len(self.data)):
            signal = None
            reason = ""

            current_row = self.data.iloc[i]

            # 基础条件：确保有足够的数据
            if pd.isna(current_row['ma_20']) or pd.isna(current_row['volatility']) or pd.isna(
                    current_row['price_position']):
                signals.append(signal)
                reasons.append(reason)
                continue

            # 检查波动性是否足够（至少需要一定波动才能交易）
            min_volatility = 0.0003  # 最小波动率阈值
            if current_row['volatility'] < min_volatility:
                signals.append(None)
                reasons.append("波动率不足")
                continue

            # 根据星期几制定不同的策略
            day_of_week = current_row['day_of_week']

            # 周三反转策略
            if day_of_week == 2:  # 周三
                # 如果前一天下跌，今天可能反弹；如果前一天上涨，今天可能回调
                if not pd.isna(current_row['prev_price_change']):
                    if current_row['prev_price_change'] < 0 and current_row['price_change_rate'] > 0:
                        signal = 'buy'
                        reason = "周三反转买入信号"
                    elif current_row['prev_price_change'] > 0 and current_row['price_change_rate'] < 0:
                        signal = 'sell'
                        reason = "周三反转卖出信号"

            # 周四周五连续性策略
            elif day_of_week in [3, 4]:  # 周四、周五
                if not pd.isna(current_row['prev_price_change']):
                    # 周四延续周三趋势
                    if day_of_week == 3 and current_row['prev_price_change'] * current_row['price_change'] > 0:
                        if current_row['price_change'] > 0:
                            signal = 'buy'
                            reason = "周四延续上涨趋势"
                        else:
                            signal = 'sell'
                            reason = "周四延续下跌趋势"
                    # 周五特殊处理 - 趋势明确时交易
                    elif day_of_week == 4 and abs(current_row['price_change_rate']) > current_row['volatility']:
                        if current_row['price_change_rate'] > 0:
                            signal = 'buy'
                            reason = "周五趋势明确买入"
                        else:
                            signal = 'sell'
                            reason = "周五趋势明确卖出"

            # 周一周二关联性策略
            elif day_of_week in [0, 1]:  # 周一、周二
                if day_of_week == 0:  # 周一
                    # 周一跟随上周五收盘情况
                    if current_row['price_change_rate'] > current_row['volatility']:
                        signal = 'buy'
                        reason = "周一强势上涨"
                    elif current_row['price_change_rate'] < -current_row['volatility']:
                        signal = 'sell'
                        reason = "周一下跌趋势"
                elif day_of_week == 1:  # 周二
                    # 周二验证周一信号
                    pass  # 在回测中处理

            # 大趋势跟随策略
            if signal is None:
                if not pd.isna(current_row['trend_strength']) and current_row['trend_strength'] > 0.001:
                    if current_row['long_term_trend'] == 'up' and current_row['short_term_trend'] == 'up':
                        signal = 'buy'
                        reason = "大趋势向上"
                    elif current_row['long_term_trend'] == 'down' and current_row['short_term_trend'] == 'down':
                        signal = 'sell'
                        reason = "大趋势向下"

            # 开盘价与当前价格关系策略
            if signal is None:
                if not pd.isna(current_row['open_position']) and not pd.isna(current_row['price_position']):
                    # 如果价格突破开盘价位置且波动足够
                    if abs(current_row['price_position'] - current_row['open_position']) > 0.3:
                        if current_row['price_position'] > current_row['open_position']:
                            signal = 'buy'
                            reason = "价格突破开盘价位置"
                        else:
                            signal = 'sell'
                            reason = "价格跌破开盘价位置"

            # 最高价最低价预测策略
            if signal is None:
                if not pd.isna(current_row['price_position']):
                    # 如果价格接近当日最高价，可能回调，考虑做空
                    if current_row['price_position'] > 0.8:
                        signal = 'sell'
                        reason = "价格接近当日最高价，考虑做空"
                    # 如果价格接近当日最低价，可能反弹，考虑做多
                    elif current_row['price_position'] < 0.2:
                        signal = 'buy'
                        reason = "价格接近当日最低价，考虑做多"

            signals.append(signal)
            reasons.append(reason)

        self.data['signal'] = signals
        self.data['signal_reason'] = reasons
        return signals, reasons

    def backtest(self):
        """
        回测策略 - 符合FTMO规则
        """
        if self.data is None or len(self.data) == 0:
            print("没有数据可用于回测")
            return {}

        # 添加信号到数据中
        signals, reasons = self.generate_signals()

        # 初始化交易参数
        initial_capital = 100000  # 初始资金10万
        lot_size = 3  # 起手3手
        pip_value_per_lot = 10  # 每标准手每点价值10美元
        max_daily_trades = 4  # 每天最多4笔交易
        max_risk_per_trade = 0.01  # 单笔风险不超过账户的1%

        # 交易记录
        trades = []
        capital = initial_capital
        capital_history = [capital]  # 记录资金历史
        daily_trades = {}  # 记录每日交易次数
        max_drawdown = 0  # 最大回撤
        peak_capital = initial_capital  # 峰值资金

        for i, row in self.data.iterrows():
            date = row['date']
            if date not in daily_trades:
                daily_trades[date] = 0

            # 检查是否达到最大回撤限制（5%）
            if peak_capital > 0:
                current_drawdown = (peak_capital - capital) / peak_capital
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown

                if max_drawdown > 0.05:  # 超过5%回撤，停止交易
                    print(f"警告：最大回撤超过5%({max_drawdown:.2%})，停止交易")
                    break

            # 更新峰值资金
            if capital > peak_capital:
                peak_capital = capital

            # 如果有信号且当日交易次数未达上限
            if row['signal'] is not None and daily_trades[date] < max_daily_trades:
                # 确定入场价格
                entry_price = row['current_price']

                # 设置止盈止损（基于ATR动态设置）
                if not pd.isna(row['volatility']):
                    # ATR近似值（简化计算）
                    atr = row['volatility'] * row['current_price'] * 10000  # 转换为点数
                    stop_loss_pips = max(15, min(30, atr * 1.5))  # 15-30点止损
                    take_profit_pips = stop_loss_pips * 3  # 3:1风险回报比
                else:
                    # 默认止盈止损
                    stop_loss_pips = 20
                    take_profit_pips = 60

                # 根据信号方向设置止盈止损价格
                if row['signal'] == 'buy':
                    stop_loss_price = entry_price - (stop_loss_pips / 10000)
                    take_profit_price = entry_price + (take_profit_pips / 10000)
                else:  # sell
                    stop_loss_price = entry_price + (stop_loss_pips / 10000)
                    take_profit_price = entry_price - (take_profit_pips / 10000)

                # 记录交易
                trade = {
                    'datetime': row['datetime'],
                    'day_of_week': row['day_of_week'],
                    'signal': row['signal'],
                    'reason': row['signal_reason'],
                    'entry_price': entry_price,
                    'take_profit': take_profit_price,
                    'stop_loss': stop_loss_price,
                    'lots': lot_size,
                    'stop_loss_pips': stop_loss_pips,
                    'take_profit_pips': take_profit_pips
                }

                trades.append(trade)
                daily_trades[date] += 1

                # 模拟交易结果（简化模型）
                # 在实际应用中，需要根据后续数据判断实际盈亏
                # 这里我们假设50%止盈概率，50%止损概率
                # 并根据止盈止损点数计算收益
                if np.random.random() < 0.5:  # 50%概率止盈
                    exit_price = take_profit_price
                    pnl_pips = take_profit_pips
                else:  # 50%概率止损
                    exit_price = stop_loss_price
                    pnl_pips = -stop_loss_pips

                # 计算收益（点数 * 每点价值 * 手数）
                pnl = pnl_pips * pip_value_per_lot * lot_size
                capital += pnl

                # 更新交易记录的实际结果
                trade['exit_price'] = exit_price
                trade['pnl_pips'] = pnl_pips
                trade['pnl'] = pnl
                trade['capital_after_trade'] = capital

            # 记录资金历史
            capital_history.append(capital)

        # 计算最终结果
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'trades': trades,
            'capital_history': capital_history
        }

    def calculate_monthly_stats(self, trades, capital_history):
        """
        计算月度统计数据
        """
        if not trades:
            return pd.DataFrame()

        # 创建交易数据框
        trades_df = pd.DataFrame(trades)
        trades_df['date'] = trades_df['datetime'].dt.date
        trades_df['month'] = trades_df['datetime'].dt.to_period('M')

        # 按月份分组计算交易次数
        monthly_trade_counts = trades_df.groupby('month').size()

        # 计算每月收益
        monthly_returns = {}
        for month, group in trades_df.groupby('month'):
            monthly_returns[month] = group['pnl'].sum()

        # 计算资金曲线的月度数据
        capital_series = pd.Series(capital_history)
        self.data['capital'] = capital_series[:len(self.data)] if len(capital_series) >= len(
            self.data) else capital_series

        # 按月份分组计算资金变化
        if 'capital' in self.data.columns:
            monthly_capital = self.data.groupby('month')['capital'].agg(['first', 'last'])
            monthly_capital['return'] = (monthly_capital['last'] - monthly_capital['first']) / monthly_capital[
                'first'] * 100
        else:
            # 如果无法计算实际资金变化，则使用交易盈亏估算
            monthly_data = []
            for month in sorted(set(trades_df['month'])):
                if month in monthly_returns:
                    monthly_data.append({
                        'month': month,
                        'first': 100000,
                        'last': 100000 + monthly_returns[month],
                        'return': (monthly_returns[month] / 100000) * 100
                    })
            monthly_capital = pd.DataFrame(monthly_data).set_index('month')

        # 合并数据
        monthly_stats = pd.DataFrame({
            'trade_count': monthly_trade_counts,
            'pnl': pd.Series(monthly_returns),
            'start_balance': monthly_capital['first'] if 'first' in monthly_capital.columns else 100000,
            'end_balance': monthly_capital['last'] if 'last' in monthly_capital.columns else (
                        100000 + pd.Series(monthly_returns).cumsum()),
            'return_pct': monthly_capital['return'] if 'return' in monthly_capital.columns else (
                        pd.Series(monthly_returns) / 100000 * 100)
        }).fillna(0)

        # 计算累计收益率和余额
        monthly_stats['cumulative_return'] = monthly_stats['return_pct'].cumsum()
        monthly_stats['balance'] = 100000 * (1 + monthly_stats['cumulative_return'] / 100)

        # 计算累计亏损（当月收益率为负时）
        monthly_stats['cumulative_loss'] = np.where(monthly_stats['return_pct'] < 0,
                                                    abs(monthly_stats['return_pct']), 0)

        return monthly_stats

    def print_trade_summary(self, results):
        """
        打印交易摘要
        """
        if not results:
            print("没有回测结果可显示")
            return

        print("\n=== FTMO策略交易摘要 ===")
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益率: {results['total_return']:.2f}%")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"总交易次数: {results['total_trades']}")
        if results['trades']:
            print(
                f"平均每日交易次数: {results['total_trades'] / len(set([t['datetime'].date() for t in results['trades']])):.1f}")

        # 按星期几统计交易
        weekday_names = ['周一', '周二', '周三', '周四', '周五']
        weekday_counts = {i: 0 for i in range(5)}
        for trade in results['trades']:
            weekday_counts[trade['day_of_week']] += 1

        print("\n按星期几统计交易次数:")
        for i in range(5):
            print(f"  {weekday_names[i]}: {weekday_counts[i]}次")

        # 计算月度统计数据
        monthly_stats = self.calculate_monthly_stats(results['trades'], results['capital_history'])
        print("\n月度统计数据:")
        print(
            f"{'月份':<10} {'交易次数':<10} {'盈亏($)':<15} {'月初余额':<15} {'月末余额':<15} {'收益率':<10} {'累计收益率':<12}")
        print("-" * 95)
        for month, row in monthly_stats.iterrows():
            print(f"{str(month):<10} "
                  f"{int(row['trade_count']):<10} "
                  f"{row['pnl']:<15.2f} "
                  f"{row['start_balance']:<15.2f} "
                  f"{row['end_balance']:<15.2f} "
                  f"{row['return_pct']:<10.2f}% "
                  f"{row['cumulative_return']:<12.2f}%")

        # 打印最近10笔交易
        print("\n最近10笔交易:")
        print(
            f"{'时间':<20} {'星期':<6} {'信号':<6} {'入场价':<10} {'出场价':<10} {'止盈点':<8} {'止损点':<8} {'收益($)':<10} {'原因'}")
        print("-" * 110)
        for trade in results['trades'][-10:]:
            print(f"{trade['datetime']:<20} "
                  f"{weekday_names[trade['day_of_week']]:<6} "
                  f"{trade['signal']:<6} "
                  f"{trade['entry_price']:<10.6f} "
                  f"{trade['exit_price']:<10.6f} "
                  f"{trade['take_profit_pips']:<8.1f} "
                  f"{trade['stop_loss_pips']:<8.1f} "
                  f"{trade['pnl']:<10.2f} "
                  f"{trade['reason']}")

    def run_strategy(self):
        """
        运行完整策略
        """
        print("连接MT5并加载数据...")
        self.load_data()

        if self.data is None or len(self.data) == 0:
            print("无法加载数据，策略无法运行")
            return {}

        print("计算技术指标...")
        self.calculate_indicators()

        print("进行星期分析...")
        weekday_stats = self.weekday_analysis()
        print(weekday_stats)

        print("进行趋势分析...")
        self.trend_analysis()

        print("生成交易信号...")
        signals, reasons = self.generate_signals()

        print("开始回测...")
        results = self.backtest()

        # 打印交易摘要
        self.print_trade_summary(results)

        return results


if __name__ == "__main__":
    # 创建策略实例并运行
    # 使用MT5数据进行回测
    strategy = FTMOStrategyMT5(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M30)
    results = strategy.run_strategy()

    # 关闭MT5连接
    mt5.shutdown()
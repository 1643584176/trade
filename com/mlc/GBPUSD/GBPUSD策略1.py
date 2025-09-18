import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import datetime
import pytz


class FTMOStrategy:
    def __init__(self, symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, lot_size=3):
        """初始化策略参数和连接MT5平台"""
        # 初始化MT5连接
        if not mt5.initialize():
            print("MT5连接初始化失败")
            mt5.shutdown()
            return

        # 验证交易品种
        self.symbol = symbol
        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            print(f"{self.symbol}品种信息获取失败")
            mt5.shutdown()
            return

        # 检查交易品种是否可用
        if not self.symbol_info.visible:
            print(f"{self.symbol}品种不可用")
            mt5.shutdown()
            return

        # 策略参数
        self.timeframe = timeframe
        self.lot_size = lot_size  # 固定3手
        self.stop_loss = 15  # 点数止损
        self.take_profit = 30  # 点数止盈
        self.max_daily_loss = 1200  # 每日最大亏损
        self.total_max_loss = 1800  # 总最大亏损
        self.consecutive_losses = 0  # 连续亏损计数
        self.max_consecutive_losses = 2  # 最大连续亏损次数
        self.daily_trades = 0  # 当日交易计数
        self.account_balance = 0  # 账户余额
        self.min_trades_per_day = 3  # 每日最低交易次数
        self.max_trades_per_day = 5  # 每日最高交易次数
        self.monthly_stats = {}  # 月度统计
        self.timezone = pytz.timezone("Etc/UTC")  # 设置时区
        self.trades = []  # 存储交易记录

        # 获取历史数据
        self._get_historical_data()

    def _get_historical_data(self):
        """从MT5获取历史数据"""
        # 获取2024年1月1日至今的历史数据
        utc_from = datetime.datetime(2024, 1, 1, tzinfo=self.timezone)
        utc_to = datetime.datetime.now(self.timezone) + datetime.timedelta(days=1)

        # 获取历史数据
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            print("历史数据获取失败")
            mt5.shutdown()
            return None

        # 转换为DataFrame
        self.data = pd.DataFrame(rates)
        self.data['时间点'] = pd.to_datetime(self.data['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(
            'Asia/Shanghai')
        self.data = self.data.sort_values('时间点')

        # 数据预处理
        self._preprocess_data()

        # 关闭MT5连接
        mt5.shutdown()

    def _preprocess_data(self):
        """数据预处理和特征工程"""
        # 计算技术指标
        self.data['MA10'] = self.data['close'].rolling(10).mean()
        self.data['MA20'] = self.data['close'].rolling(20).mean()

        # RSI计算
        delta = self.data['close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # 布林带计算
        self.data['BB_mid'] = self.data['close'].rolling(20).mean()
        std = self.data['close'].rolling(20).std()
        self.data['BB_upper'] = self.data['BB_mid'] + 2 * std
        self.data['BB_lower'] = self.data['BB_mid'] - 2 * std

        # 添加星期特征
        self.data['星期'] = self.data['时间点'].dt.weekday  # 0=星期一, 4=星期五
        self.data['hour'] = self.data['时间点'].dt.hour  # 小时特征

        # ATR计算（14周期）
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift())
        tr3 = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(14).mean()

    def _check_trend(self, row):
        """趋势识别模块：结合均线和价格动量"""
        ma10 = row['MA10']
        ma20 = row['MA20']

        if pd.isna(ma10) or pd.isna(ma20):
            return None

        # 使用diff计算均线变化方向
        ma10_diff = self.data['MA10'].diff().iloc[row.name]

        if ma10 > ma20 and ma10_diff > 0:  # 上升趋势
            return 'long'
        elif ma10 < ma20 and ma10_diff < 0:  # 下降趋势
            return 'short'
        return None

    def _check_momentum(self, row):
        """动量确认模块：使用RSI指标"""
        if row['RSI'] < 30:  # 超买区域，看涨动量
            return 'long'
        elif row['RSI'] > 70:  # 超卖区域，看跌动量
            return 'short'
        return None

    def _check_price_action(self, row):
        """价格行为模块：识别反转形态"""
        # 计算K线形态
        body = abs(row['close'] - row['open'])
        upper_shadow = row['high'] - max(row['open'], row['close'])
        lower_shadow = min(row['open'], row['close']) - row['low']

        # 识别锤子线（看涨反转）
        if lower_shadow > 2 * body and upper_shadow < body:
            return 'long'
        # 识别倒锤子线（看跌反转）
        elif upper_shadow > 2 * body and lower_shadow < body:
            return 'short'
        return None

    def _check_time_factors(self, row):
        """时间因素模块：根据星期几和小时调整策略"""
        weekday = row['时间点'].weekday()
        hour = row['hour']

        # 周五尾盘交易限制
        if weekday == 4 and hour >= 20:  # 周五20点后保守交易
            return 'conservative'

        # 周一周二趋势延续性
        if weekday in [0, 1]:  # 周一、周二
            return True

        # 周三反转策略
        if weekday == 2:  # 周三
            return 'reversal'

        # 周四趋势延续
        if weekday == 3:  # 周四
            return True

        # 周五常规交易时段
        if weekday == 4 and hour < 20:  # 周五早盘
            return 'conservative'

        return True

    def _check_trading_conditions(self, row):
        """综合检查交易条件"""
        # 检查每日交易次数
        if self.daily_trades >= self.max_trades_per_day:
            return False

        # 检查每日最大亏损
        if self.account_balance < -self.max_daily_loss:
            return False

        # 检查总最大亏损
        if self.account_balance < -self.total_max_loss:
            return False

        # 检查连续亏损
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False

        return True

    def _calculate_position_size(self, risk_per_trade):
        """计算仓位大小，根据账户余额和风险参数"""
        # 根据风险调整仓位大小（简化为固定3手）
        return self.lot_size

    def _calculate_risk_reward(self, row, signal):
        """基于ATR动态计算止损止盈点位"""
        # 获取当前ATR值
        atr = row['ATR']

        # 根据时段调整风险参数
        hour = row['hour']
        if 7 <= hour < 15:  # 亚洲时段
            sl = atr * 0.8
            tp = atr * 1.6
        elif 15 <= hour < 22:  # 欧洲时段
            sl = atr * 1.0
            tp = atr * 2.0
        else:  # 美洲时段
            sl = atr * 1.2
            tp = atr * 2.4

        # 获取当前价格
        current_price = row['close']

        # 计算点数
        pip_size = 0.0001  # 假设1点=0.0001
        sl_points = round(sl / pip_size)
        tp_points = round(tp / pip_size)

        # 计算点位
        if signal == 'long':
            stop_loss = current_price - sl
            take_profit = current_price + tp
        else:
            stop_loss = current_price + sl
            take_profit = current_price - tp

        return stop_loss, take_profit, sl_points, tp_points

    def _generate_signal(self, row):
        """生成交易信号：结合多因子验证"""
        # 检查基本交易条件
        if not self._check_trading_conditions(row):
            return None

        # 获取各模块信号
        trend_signal = self._check_trend(row)
        momentum_signal = self._check_momentum(row)
        price_action = self._check_price_action(row)
        time_factor = self._check_time_factors(row)

        # 综合信号生成
        signals = []
        if trend_signal:
            signals.append(trend_signal)
        if momentum_signal:
            signals.append(momentum_signal)
        if price_action:
            signals.append(price_action)

        # 严格信号验证：需要至少两个模块一致
        # 周三反转策略
        if time_factor == 'reversal':
            if len(signals) >= 2:
                # 反转主要信号方向
                if signals.count('long') > signals.count('short'):
                    return 'short'
                elif signals.count('short') > signals.count('long'):
                    return 'long'

            # 周三强制交易机制：在最后交易时段强制反向交易
            if row['hour'] >= 20:  # 周三20点后强制反转
                if signals:
                    if signals.count('long') > signals.count('short'):
                        return 'short'
                    elif signals.count('short') > signals.count('long'):
                        return 'long'

                # 如果没有信号，根据布林带强制交易
                if row['close'] < row['BB_lower']:
                    return 'long'
                elif row['close'] > row['BB_upper']:
                    return 'short'

        # 周五保守策略
        elif time_factor == 'conservative':
            if len(signals) >= 2 and signals.count('long') > signals.count('short'):
                return 'long'
            elif len(signals) >= 2 and signals.count('short') > signals.count('long'):
                return 'short'

        # 常规交易日
        else:
            if signals.count('long') >= 2:
                return 'long'
            elif signals.count('short') >= 2:
                return 'short'

        return None

    def _execute_trade(self, signal, row):
        """执行交易：基于当前K线收盘价交易"""
        # 获取当前价格
        entry_price = row['close']
        timestamp = row['时间点']

        # 计算动态止损止盈
        stop_loss, take_profit, sl_points, tp_points = self._calculate_risk_reward(row, signal)

        # 计算盈亏（基于ATR的动态风险）
        if signal == 'long':
            price_move = row['high'] - entry_price  # 当前K线最高点
            if price_move >= (take_profit - entry_price):  # 触发止盈
                profit_loss = self.lot_size * tp_points
            elif entry_price - row['low'] >= (entry_price - stop_loss):  # 触发止损
                profit_loss = -self.lot_size * sl_points
            else:  # 未触发，按收盘价计算
                profit_loss = self.lot_size * (row['close'] - entry_price) / 0.0001
        else:
            price_move = entry_price - row['low']  # 当前K线最低点
            if price_move >= (entry_price - take_profit):  # 触发止盈
                profit_loss = self.lot_size * tp_points
            elif row['high'] - entry_price >= (stop_loss - entry_price):  # 触发止损
                profit_loss = -self.lot_size * sl_points
            else:  # 未触发，按收盘价计算
                profit_loss = self.lot_size * (entry_price - row['close']) / 0.0001

        # 记录交易
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': self.lot_size,
            'profit_loss': profit_loss,
            'status': 'closed'
        }

        # 更新账户余额
        self.account_balance += profit_loss

        # 记录月度统计
        month_key = f"{timestamp.year}-{timestamp.month:02d}"
        if month_key not in self.monthly_stats:
            self.monthly_stats[month_key] = {'total_trades': 0, 'total_loss': 0, 'total_profit': 0, 'win_trades': 0,
                                             'loss_trades': 0}

        # 更新月度统计
        self.monthly_stats[month_key]['total_trades'] += 1
        if profit_loss > 0:
            self.monthly_stats[month_key]['win_trades'] += 1
            self.monthly_stats[month_key]['total_profit'] += profit_loss
        else:
            self.monthly_stats[month_key]['loss_trades'] += 1
            self.monthly_stats[month_key]['total_loss'] += abs(profit_loss)

        # 更新交易计数
        self.daily_trades += 1
        if profit_loss > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        # 存储交易记录
        self.trades.append(trade)
        return trade

    def backtest(self):
        """回测函数：遍历数据执行策略"""
        current_day = None

        for _, row in self.data.iterrows():
            # 每日重置交易计数
            if current_day != row['时间点'].date():
                current_day = row['时间点'].date()
                self.daily_trades = 0

            # 生成交易信号
            signal = self._generate_signal(row)

            # 执行交易
            if signal:
                self._execute_trade(signal, row)

        return self.trades

    def print_monthly_stats(self):
        """输出月度统计信息"""
        print("\n月度统计报告:")
        print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            '月份', '交易次数', '盈利次数', '亏损次数', '月度盈亏', '账户余额'))

        total_months = 0
        profitable_months = 0
        losing_months = 0
        total_trades = 0
        total_profit = 0
        total_loss = 0

        # 按月份排序输出
        sorted_months = sorted(self.monthly_stats.items())

        for month, stats in sorted_months:
            total_months += 1
            total_trades += stats['total_trades']
            total_profit += stats['total_profit']
            total_loss += stats['total_loss']

            # 计算月度盈亏
            monthly_pl = stats['total_profit'] - stats['total_loss']
            if monthly_pl > 0:
                profitable_months += 1
            elif monthly_pl < 0:
                losing_months += 1

            # 获取该月最后一天的账户余额
            month_trades = [t for t in self.trades if t['timestamp'].strftime('%Y-%m') == month]
            if month_trades:
                final_balance = f"{month_trades[-1]['timestamp'].strftime('%Y-%m-%d')} ${self.account_balance:.2f}"
            else:
                final_balance = f"${self.account_balance:.2f}"

            print("{:<10} {:<10} {:<10} {:<10} ${:<9.2f} {}".format(
                month,
                stats['total_trades'],
                stats['win_trades'],
                stats['loss_trades'],
                monthly_pl,
                final_balance
            ))

        # 输出总体统计
        print("\n总体统计:")
        print(f"交易月份总数: {total_months}")
        print(f"盈利月份: {profitable_months} ({profitable_months / total_months * 100:.1f}%)")
        print(f"亏损月份: {losing_months} ({losing_months / total_months * 100:.1f}%)")
        print(f"总交易次数: {total_trades}")
        print(f"总盈利: ${total_profit:.2f}")
        print(f"总亏损: ${total_loss:.2f}")
        print(f"净收益: ${total_profit - total_loss:.2f}")
        print(f"最终账户余额: ${self.account_balance:.2f}")


# 使用示例
if __name__ == '__main__':
    # 初始化策略并执行回测
    strategy = FTMOStrategy()
    trades = strategy.backtest()

    # 输出交易统计
    print(f'生成交易信号数量: {len(trades)}')

    # 输出月度统计
    strategy.print_monthly_stats()
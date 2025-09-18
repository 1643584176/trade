import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import datetime
import pytz

class FTMOStrategy:
    def __init__(self, symbol, timeframe, initial_balance=10000,
                 risk_per_trade=0.01, max_daily_trades=3,
                 stop_loss_pips=20, take_profit_pips=40,
                 data_file=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_trades = max_daily_trades
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.data = None
        self.trades = []
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.account_balance = initial_balance
        self.monthly_stats = {}
        self.timezone = pytz.timezone('Asia/Shanghai')

        # 设置默认的交易限制
        self.max_trades_per_day = max_daily_trades
        self.max_consecutive_losses = 5
        self.total_max_loss = initial_balance * 0.1  # 总最大亏损为初始资金的10%
        self.max_daily_loss = initial_balance * 0.02  # 每日最大亏损为初始资金的2%
        self.lot_size = 3  # 默认手数

        if data_file:
            self._load_csv_data(data_file)
        else:
            self._load_data()

    def _load_csv_data(self, data_file):
        self.data = pd.read_csv(data_file)
        column_mapping = {
            '当前开盘价': 'open',
            '当前收盘价': 'close',
            '当前最高价': 'high',
            '当前最低价': 'low',
            '时间点': 'time'
        }
        self.data = self.data.rename(columns=column_mapping)
        # 确保时间列转换为datetime格式
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data = self.data.sort_values('time')
        self._preprocess_data()

    def _load_data(self):
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
        self.data['时间点'] = pd.to_datetime(self.data['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
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
        
        # 添加历史K线数据（预处理）
        self.data['prev_close'] = self.data['close'].shift(1)
        self.data['prev_open'] = self.data['open'].shift(1)
        self.data['prev_body'] = abs(self.data['prev_close'] - self.data['prev_open'])
        
        # 添加星期特征
        self.data['星期'] = self.data['时间点'].dt.weekday  # 0=星期一, 4=星期五
        self.data['hour'] = self.data['时间点'].dt.hour  # 小时特征
        
        # ATR计算（14周期）
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift())
        tr3 = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(14).mean()
        
        # 填充NaN值
        self.data = self.data.ffill()  # 前向填充
        
        # 确保所有价格数据保留6位小数（符合数据精度处理规范）
        price_columns = ['open', 'high', 'low', 'close', 'prev_close', 'prev_open', 'BB_upper', 'BB_lower', 'BB_mid']
        for col in price_columns:
            self.data[col] = self.data[col].round(6)
        
    def _check_trend(self, row):
        """趋势识别模块：结合均线和价格动量"""
        # 获取当前行的MA值
        ma10 = row['MA10']
        ma20 = row['MA20']
        
        if pd.isna(ma10) or pd.isna(ma20):
            return None
            
        # 使用diff计算均线变化方向
        ma10_diff = self.data['MA10'].diff().iloc[row.name]
        
        # 基于开盘价的趋势判断
        daily_open = row['open']  # 当前K线开盘价作为当日基准
        
        # 增加趋势确认条件：结合价格行为
        if row['close'] > daily_open and row['close'] > row['BB_mid']:  # 价格在布林带中轨之上
            return 'long'
        elif row['close'] < daily_open and row['close'] < row['BB_mid']:  # 价格在布林带中轨之下
            return 'short'
        return None
        
    def _check_time_factors(self):
        """时间因素模块：根据小时调整策略"""
        # 移除时段限制，始终返回活跃状态
        return 'active'
        
    def _generate_signal(self, row):
        """生成交易信号：结合多因子验证"""
        # 检查基本交易条件
        if not self._check_trading_conditions(row):
            return None
            
        # 时间过滤机制 - 直接使用新的时间因素模块
        if self._check_time_factors() == 'inactive':
            return None
            
        # 获取当前价格行为
        curr_close = row['close']
        curr_open = row['open']
        curr_high = row['high']
        curr_low = row['low']
        
        # 计算K线形态
        body = abs(curr_close - curr_open)
        upper_shadow = curr_high - max(curr_open, curr_close)
        lower_shadow = min(curr_open, curr_close) - curr_low
        
        # 获取前一根K线数据（使用预处理数据）
        prev_close = row['prev_close']
        prev_open = row['prev_open']
        prev_body = row['prev_body']
        
        # 基于开盘价的趋势判断
        if curr_close > curr_open:  # 阳线
            return 'long'
        elif curr_close < curr_open:  # 阴线
            return 'short'
            
        # 基于布林带突破
        if curr_close < row['BB_lower']:
            return 'long'
        elif curr_close > row['BB_upper']:
            return 'short'
            
        # 基于锤子线形态
        if lower_shadow > body * 2 and upper_shadow < body:
            return 'long'
        elif upper_shadow > body * 2 and lower_shadow < body:
            return 'short'
            
        # 基于吞没形态
        if not pd.isna(prev_close) and not pd.isna(prev_open):
            # 多头吞没
            if (prev_open > prev_close and  # 前一根阴线
                curr_close > prev_open and  # 当前收盘价超过前一根开盘价
                body > prev_body):  # 当前实体更大
                return 'long'
            # 空头吞没
            elif (prev_close > prev_open and  # 前一根阳线
                  curr_close < prev_open and  # 当前收盘价低于前一根开盘价
                  body > prev_body):  # 当前实体更大
                return 'short'
                
        return None
        
    def _calculate_risk_reward(self, row, signal):
        """基于ATR动态计算止损止盈点位"""
        # 获取当前ATR值（确保不是NaN）
        atr = row['ATR'] if not pd.isna(row['ATR']) else 0.0001  # 默认最小值
        
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
            
        # 确保风险参数合理
        sl = max(sl, 0.0001)  # 最小风险值
        tp = max(tp, 0.0001)
        
        # 获取当前价格
        current_price = row['close']
        
        # 计算点位
        if signal == 'long':
            stop_loss = current_price - sl
            take_profit = current_price + tp
        else:
            stop_loss = current_price + sl
            take_profit = current_price - tp
            
        # 计算点数（处理NaN情况）
        pip_size = 0.0001
        sl_points = round(sl / pip_size) if sl >= 0.0001 else 1  # 最小1点
        tp_points = round(tp / pip_size) if tp >= 0.0001 else 1  # 最小1点
        
        return stop_loss, take_profit, sl_points, tp_points
        
    def _create_order(self, row):
        """创建交易订单"""
        return {
            'entry_price': row['open'],
            'stop_loss': row['open'] - self.stop_loss_pips * 0.0001,
            'take_profit': row['open'] + self.take_profit_pips * 0.0001,
            'timestamp': row['time'],
            'status': 'open'
        }
        
    def _execute_trade(self, signal, row):
        """执行交易：基于当前K线收盘价交易"""
        # 获取当前价格
        entry_price = row['close']
        timestamp = row['时间点']
        
        # 计算动态止损止盈
        stop_loss, take_profit, sl_points, tp_points = self._calculate_risk_reward(row, signal)
        
        # 计算盈亏（基于ATR的动态风险）
        if signal == 'long':
            # 实际价格变动（考虑点差）
            price_move = row['high'] - entry_price - 2 * 0.0001  # 点差
            # 检查是否触发止盈
            if price_move >= (take_profit - entry_price):
                profit_loss = self.lot_size * tp_points
            # 检查是否触发止损
            elif entry_price - row['low'] >= (entry_price - stop_loss):
                profit_loss = -self.lot_size * sl_points
            # 未触发，按收盘价计算
            else:
                profit_loss = self.lot_size * price_move / 0.0001
        else:
            # 实际价格变动（考虑点差）
            price_move = entry_price - row['low'] - 2 * 0.0001  # 点差
            # 检查是否触发止盈
            if price_move >= (entry_price - take_profit):
                profit_loss = self.lot_size * tp_points
            # 检查是否触发止损
            elif row['high'] - entry_price >= (stop_loss - entry_price):
                profit_loss = -self.lot_size * sl_points
            # 未触发，按收盘价计算
            else:
                profit_loss = self.lot_size * price_move / 0.0001
                
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'signal': signal,
            'entry_price': round(entry_price, 6),
            'stop_loss': round(stop_loss, 6),
            'take_profit': round(take_profit, 6),
            'position_size': self.lot_size,
            'profit_loss': profit_loss,
            'status': 'closed'
        }
        
        # 更新账户余额
        self.account_balance += profit_loss
        
        # 记录月度统计
        month_key = f"{timestamp.year}-{timestamp.month:02d}"
        if month_key not in self.monthly_stats:
            self.monthly_stats[month_key] = {'total_trades': 0, 'total_loss': 0, 'total_profit': 0, 'win_trades': 0, 'loss_trades': 0}
            
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
        
    def _check_exit_conditions(self, row, order):
        """检查是否触发止损或止盈"""
        if order['status'] == 'closed':
            return

        # 检查止损
        if row['low'] <= order['stop_loss']:
            order['exit_price'] = order['stop_loss']
            order['status'] = 'closed'
            order['pnl'] = -self.stop_loss_pips
            self.trades.append(order.copy())
            return

        # 检查止盈
        if row['high'] >= order['take_profit']:
            order['exit_price'] = order['take_profit']
            order['status'] = 'closed'
            order['pnl'] = self.take_profit_pips
            self.trades.append(order.copy())

    def backtest(self):
        """执行策略回测"""
        if self.data is None:
            print("数据未加载")
            return []

        current_day = None
        for index, row in self.data.iterrows():
            # 修正列名引用：使用'time'代替'时间点'
            if current_day != row['time'].date():
                current_day = row['time'].date()
                self.daily_trades = 0
                self.consecutive_losses = 0

            # 检查是否满足交易条件 - 修复方法名
            if self._check_trading_conditions(row):
                # 创建订单
                order = self._create_order(row)
                # 检查是否触发止损/止盈
                self._check_exit_conditions(row, order)

        return self.trades
        
    def print_monthly_stats(self):
        if not self.trades:
            print("没有交易记录")
            return
        
        # 创建交易DataFrame
        trades_df = pd.DataFrame(self.trades)
        # 确保timestamp列是datetime类型
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['month'] = trades_df['timestamp'].dt.to_period('M')
        
        # 按月分组统计
        monthly_stats = trades_df.groupby('month').agg(
            交易次数=('pnl', 'count'),
            盈利次数=('pnl', lambda x: (x > 0).sum()),
            亏损次数=('pnl', lambda x: (x <= 0).sum()),
            月度盈亏=('pnl', 'sum')
        ).reset_index()
        
        total_months = len(monthly_stats)
        if total_months == 0:
            print("没有月度交易数据")
            return
            
        profitable_months = len(monthly_stats[monthly_stats['月度盈亏'] > 0])
        losing_months = total_months - profitable_months
        avg_monthly_pnl = monthly_stats['月度盈亏'].mean()
        max_monthly_pnl = monthly_stats['月度盈亏'].max()
        min_monthly_pnl = monthly_stats['月度盈亏'].min()
        
        # 计算账户余额变化
        monthly_stats['账户余额'] = self.initial_balance + monthly_stats['月度盈亏'].cumsum()
        
        print("月度统计报告:")
        print(monthly_stats.to_string(index=False))
        print("\n总体统计:")
        print(f"交易月份总数: {total_months}")
        print(f"盈利月份: {profitable_months} ({profitable_months/total_months*100:.1f}%)")
        print(f"亏损月份: {losing_months} ({losing_months/total_months*100:.1f}%)")
        print(f"平均月盈利: ${avg_monthly_pnl:.2f}")
        print(f"最大月盈利: ${max_monthly_pnl:.2f}")
        print(f"最大月亏损: ${min_monthly_pnl:.2f}")

# 使用示例
if __name__ == '__main__':
    # 创建策略实例并从CSV加载数据
    strategy = FTMOStrategy(
        symbol='GBPUSD',
        timeframe='H1',
        stop_loss_pips=20,
        take_profit_pips=40,
        data_file=r'C:\Users\孙明辉\PycharmProjects\PythonProject\.venv\Include\AI策略\GBPUSD\GBPUSD_H1_历史数据.csv'
    )
    trades = strategy.backtest()
    print(f"交易信号数量: {len(trades)}")
    strategy.print_monthly_stats()
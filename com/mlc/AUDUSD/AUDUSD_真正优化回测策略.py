"""
AUDUSD真正优化回测策略
修复了基于历史交易结果调整开仓方向的逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import MetaTrader5 as mt5

# ======================== 1. 策略参数 ========================
# 交易参数
INITIAL_BALANCE = 100000.0  # 初始资金
CONTRACT_SIZE = 100000     # 合约规模
MIN_LOT = 2.0              # 最小手数
MAX_LOT = 3.0              # 最大手数

# 风险控制参数
DAILY_LOSS_LIMIT = INITIAL_BALANCE * 0.05  # 每日最大亏损5%
TOTAL_LOSS_LIMIT = INITIAL_BALANCE * 0.10  # 总最大亏损10%
MIN_EQUITY_LIMIT = INITIAL_BALANCE * 0.90  # 最低净值限制90%

# 点差参数（模拟真实交易环境）
SPREAD_PIPS = 1.5  # 点差为1.5点
POINT_SIZE = 0.0001  # 点值大小

# ======================== 2. 工具函数 ========================
def load_historical_data_from_mt5(symbol="AUDUSD", timeframe=mt5.TIMEFRAME_H1, days=730):
    """
    直接从MT5加载历史数据
    
    Args:
        symbol (str): 交易品种
        timeframe: 时间周期
        days (int): 获取天数
        
    Returns:
        DataFrame: 历史数据
    """
    try:
        # 初始化MT5连接
        if not mt5.initialize():
            print("MT5初始化失败")
            return None

        # 计算日期范围
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        print(f"正在从MT5获取 {symbol} 从 {from_date.strftime('%Y-%m-%d')} 到 {to_date.strftime('%Y-%m-%d')} 的H1历史数据...")

        # 获取历史数据
        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
        
        if rates is None or len(rates) == 0:
            print("未能获取到历史数据")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 重命名列以匹配回测策略
        df.rename(columns={
            'time': '时间点',
            'open': '当前开盘价',
            'close': '当前收盘价',
            'high': '当前最高价',
            'low': '当前最低价',
            'tick_volume': '成交量'
        }, inplace=True)
        
        # 重新排序列
        df = df[['时间点', '当前开盘价', '当前最高价', '当前最低价', '当前收盘价', '成交量']]
        
        # 添加技术指标
        df = add_technical_indicators(df)
        
        print(f"成功获取 {len(df)} 条数据")
        return df
        
    except Exception as e:
        print(f"从MT5获取历史数据时发生错误: {str(e)}")
        return None

def add_technical_indicators(df):
    """
    添加技术指标到数据中
    
    Args:
        df (DataFrame): 原始数据
        
    Returns:
        DataFrame: 添加技术指标后的数据
    """
    # 计算当前增减（收盘价-开盘价）
    df['当前增减'] = df['当前收盘价'] - df['当前开盘价']
    
    # 计算当前波动幅度（最高价-最低价）
    df['当前波动幅度'] = df['当前最高价'] - df['当前最低价']
    
    # 计算昨天增减（前一天的增减）
    df['昨天增减'] = df['当前增减'].shift(1)
    
    # 计算昨日波动幅度（前一天的波动幅度）
    df['昨日波动幅度'] = df['当前波动幅度'].shift(1)
    
    # 计算技术指标（用于入场信号，最小周期确保有值）
    df['MA5'] = df['当前收盘价'].rolling(window=5, min_periods=5).mean()
    df['MA10'] = df['当前收盘价'].rolling(window=10, min_periods=10).mean()

    # 计算过去7天（168小时）平均波动幅度（判断市场稳定性）
    df['past_week_vol'] = df['当前波动幅度'].rolling(window=168, min_periods=168).mean()
    
    # 计算ATR指标（平均真实波幅）
    df['ATR'] = calculate_atr(df, 14)
    
    return df

def calculate_atr(df, period=14):
    """
    计算ATR指标
    
    Args:
        df (DataFrame): 数据
        period (int): 周期
        
    Returns:
        Series: ATR值
    """
    # 计算真实波幅(TR)
    df['high_low'] = df['当前最高价'] - df['当前最低价']
    df['high_close'] = abs(df['当前最高价'] - df['当前收盘价'].shift())
    df['low_close'] = abs(df['当前最低价'] - df['当前收盘价'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    
    # 计算ATR
    atr = df['tr'].rolling(window=period).mean()
    
    # 清理临时列
    df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1, inplace=True)
    
    return atr

def adjust_lot_size(current_row, past_week_vol):
    """
    动态调整仓位（2-3手，基于趋势强度）
    
    Args:
        current_row (Series): 当前K线数据
        past_week_vol (float): 过去一周波动幅度均值
        
    Returns:
        float: 手数
    """
    if pd.isna(past_week_vol) or current_row['当前波动幅度'] == 0:
        return MIN_LOT

    # 趋势强度=当前价格变动/当前波动幅度（越大趋势越明确）
    trend_strength = abs(current_row['当前增减']) / current_row['当前波动幅度']

    # 趋势明确且波动较小时用3手，否则用2手
    return MAX_LOT if (trend_strength > 0.6 and current_row['当前波动幅度'] < past_week_vol) else MIN_LOT

def calculate_sl_tp(current_row, direction, atr=None):
    """
    计算止损止盈（考虑点差的真实价格，使用ATR优化）
    
    Args:
        current_row (Series): 当前K线数据
        direction (str): 交易方向 "long" 或 "short"
        atr (float): ATR值，用于动态止损止盈
        
    Returns:
        tuple: (止损价, 止盈价)
    """
    # 使用ATR或默认波动幅度计算止损止盈
    if atr is not None and not pd.isna(atr):
        volatility = atr
    else:
        volatility = current_row['当前波动幅度']
    
    # 根据时间调整止损止盈倍数
    hour = current_row['时间点'].hour
    
    # 流动性低时段（凌晨0-5点）：收紧止损止盈
    if 0 <= hour <= 5:
        sl_multiplier = 1.0
        tp_multiplier = 2.0
    else:
        sl_multiplier = 1.5
        tp_multiplier = 3.0

    # 按多空方向计算点位（考虑点差）
    spread_price = SPREAD_PIPS * POINT_SIZE
    if direction == "long":
        entry_price = current_row['当前开盘价'] + spread_price  # 买入价（考虑点差）
        sl = entry_price - sl_multiplier * volatility
        tp = entry_price + tp_multiplier * volatility
    else:
        entry_price = current_row['当前开盘价'] - spread_price  # 卖出价（考虑点差）
        sl = entry_price + sl_multiplier * volatility
        tp = entry_price - tp_multiplier * volatility

    return round(sl, 5), round(tp, 5)

def get_weekday_name(weekday_num):
    """
    获取星期名称
    
    Args:
        weekday_num (int): 星期数字 (0-6)
        
    Returns:
        str: 星期名称
    """
    weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    return weekday_names[weekday_num] if 0 <= weekday_num <= 6 else "未知"

# ======================== 3. 策略逻辑 ========================
class AUDUSDStrategy:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.equity = INITIAL_BALANCE
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.last_trading_day = None
        self.positions = []  # 确保任何时候最多只持有一笔交易
        self.trade_log = []
        self.last_trade_result = None  # 记录上一笔交易结果（止盈/止损）
        self.consecutive_losses = 0  # 连续亏损次数
        
    def check_daily_reset(self, current_date):
        """
        检查是否需要重置每日统计
        
        Args:
            current_date (date): 当前日期
            
        Returns:
            bool: 是否需要重置
        """
        if self.last_trading_day is None or current_date > self.last_trading_day:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trading_day = current_date
            return True
        return False
    
    def can_open_position(self):
        """
        检查是否可以开仓（满足风控条件）
        
        Returns:
            bool: 是否可以开仓
        """
        # 检查各种风控条件
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            return False  # 达到每日最大亏损限制
            
        if self.balance - INITIAL_BALANCE <= -TOTAL_LOSS_LIMIT:
            return False  # 达到总最大亏损限制
            
        if self.equity < MIN_EQUITY_LIMIT:
            return False  # 净值低于最低限制
            
        # 确保任何时候最多只持有一笔交易
        if len(self.positions) > 0:
            return False  # 已有持仓，不能再开新仓
            
        # 如果连续亏损超过3次，暂停交易
        if self.consecutive_losses >= 3:
            print("[AUDUSD真正优化] 连续亏损超过3次，暂停交易")
            return False
            
        return True
    
    def generate_signal(self, df, i):
        """
        生成交易信号（真正优化版）
        
        Args:
            df (DataFrame): 历史数据
            i (int): 当前索引
            
        Returns:
            tuple: (信号方向, 信号类型) 或 (None, None)
        """
        if i < 2:
            return None, None
            
        latest_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        has_enough_history = i >= 3
        
        # 检查必要的数据是否存在且不为空
        if (pd.isna(latest_data['MA5']) or pd.isna(latest_data['MA10']) or 
            pd.isna(prev_data['MA5']) or pd.isna(prev_data['MA10'])):
            return None, None
        
        # 获取星期几
        current_weekday = latest_data['时间点'].weekday()
        current_weekday_chinese = get_weekday_name(current_weekday)
        
        prev_weekday = prev_data['时间点'].weekday()
        prev_weekday_chinese = get_weekday_name(prev_weekday)
        
        print(f"[AUDUSD真正优化] 分析信号: 昨天{prev_weekday_chinese}, 今天{current_weekday_chinese}")
        
        # 基于上一笔交易结果调整开仓策略
        should_reverse = False
        if self.last_trade_result == "止盈":
            should_reverse = True
            print("[AUDUSD真正优化] 上一笔交易止盈，考虑反向开仓")
        elif self.last_trade_result == "止损":
            print("[AUDUSD真正优化] 上一笔交易止损，考虑顺势开仓")
        
        # 1. 周一→周二趋势延续
        if prev_weekday == 0 and current_weekday == 1:  # 周一→周二
            # 周一上涨→周二延续：做多
            if (prev_data['昨天增减'] > 0 and
                    latest_data['当前开盘价'] > prev_data['当前收盘价'] and
                    latest_data['当前增减'] > 0):
                direction = "long"
                signal_type = "周一上涨周二延续"
                if should_reverse:
                    direction = "short"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type

            # 周一下跌→周二延续：做空
            elif (prev_data['昨天增减'] < 0 and
                  latest_data['当前开盘价'] < prev_data['当前收盘价'] and
                  latest_data['当前增减'] < 0):
                direction = "short"
                signal_type = "周一下跌周二延续"
                if should_reverse:
                    direction = "long"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type

        # 2. 周三反转信号
        elif current_weekday == 2 and has_enough_history:  # 周三
            # 前两日下跌→周三反转：做多
            if (df.iloc[i-2]['当前增减'] < 0 and
                    prev_data['当前增减'] < 0 and
                    latest_data['当前开盘价'] > prev_data['当前收盘价'] and
                    latest_data['当前增减'] > 0):
                direction = "long"
                signal_type = "周三反转做多"
                if should_reverse:
                    direction = "short"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type

            # 前两日上涨→周三反转：做空
            elif (df.iloc[i-2]['当前增减'] > 0 and
                  prev_data['当前增减'] > 0 and
                  latest_data['当前开盘价'] < prev_data['当前收盘价'] and
                  latest_data['当前增减'] < 0):
                direction = "short"
                signal_type = "周三反转做空"
                if should_reverse:
                    direction = "long"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type

        # 3. MA金叉/死叉（技术信号）
        elif i >= 1:
            prev_ma_data = df.iloc[i-1]
            # MA金叉
            if (latest_data['MA5'] > latest_data['MA10'] and 
                prev_ma_data['MA5'] <= prev_ma_data['MA10']):
                direction = "long"
                signal_type = "MA金叉"
                if should_reverse:
                    direction = "short"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type
            
            # MA死叉
            elif (latest_data['MA5'] < latest_data['MA10'] and 
                  prev_ma_data['MA5'] >= prev_ma_data['MA10']):
                direction = "short"
                signal_type = "MA死叉"
                if should_reverse:
                    direction = "long"
                    signal_type += "(反向)"
                print(f"[AUDUSD真正优化] 满足做{direction}条件: {signal_type}")
                return direction, signal_type
        
        return None, None
    
    def execute_trade(self, signal_type, signal_direction, data_row, lot_size):
        """
        执行交易
        
        Args:
            signal_type (str): 信号类型
            signal_direction (str): 信号方向
            data_row (Series): 数据行
            lot_size (float): 手数
        """
        # 计算止损止盈（考虑点差，使用ATR优化）
        sl, tp = calculate_sl_tp(data_row, signal_direction, data_row['ATR'])
        
        # 使用下一根K线的开盘价作为入场价（模拟市价单，考虑点差）
        if signal_direction == "long":
            entry_price = data_row['当前开盘价'] + SPREAD_PIPS * POINT_SIZE  # 买入价
        else:
            entry_price = data_row['当前开盘价'] - SPREAD_PIPS * POINT_SIZE  # 卖出价
        
        # 创建交易记录
        trade = {
            'timestamp': data_row['时间点'],
            'direction': signal_direction,
            'lot_size': lot_size,
            'entry_price': entry_price,
            'sl': sl,
            'tp': tp,
            'signal_type': signal_type,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0.0,
            'status': 'open'
        }
        
        self.positions.append(trade)
        self.daily_trades += 1
        
        print(f"[AUDUSD真正优化] 开仓: {signal_direction}, 信号: {signal_type}, 入场价: {entry_price}, 止损: {sl}, 止盈: {tp}, 手数: {lot_size}")
    
    def check_exit_conditions(self, current_data, next_data):
        """
        检查平仓条件
        
        Args:
            current_data (Series): 当前K线数据
            next_data (Series): 下一根K线数据
        """
        if not self.positions:
            return
            
        # 使用下一根K线的开盘价作为平仓价（模拟市价单，考虑点差）
        if next_data is not None:
            # 平仓时使用相应的买入/卖出价
            if self.positions[0]['direction'] == 'long':
                exit_price = next_data['当前开盘价'] - SPREAD_PIPS * POINT_SIZE  # 卖出价
            else:
                exit_price = next_data['当前开盘价'] + SPREAD_PIPS * POINT_SIZE  # 买入价
            exit_time = next_data['时间点']
        else:
            # 如果没有下一根K线，使用当前K线收盘价平仓
            exit_price = current_data['当前收盘价']
            exit_time = current_data['时间点']
        
        for position in self.positions[:]:  # 使用切片复制避免在迭代时修改列表
            if position['status'] == 'closed':
                continue
                
            should_close = False
            close_reason = ""
            
            # 止损
            if position['direction'] == 'long' and exit_price <= position['sl']:
                should_close = True
                close_reason = "止损"
            elif position['direction'] == 'long' and exit_price >= position['tp']:
                should_close = True
                close_reason = "止盈"
            elif position['direction'] == 'short' and exit_price >= position['sl']:
                should_close = True
                close_reason = "止损"
            elif position['direction'] == 'short' and exit_price <= position['tp']:
                should_close = True
                close_reason = "止盈"
            
            # 时间止损（最大持仓2根K线，即2小时）
            entry_time = position['timestamp']
            time_diff = (exit_time - entry_time).total_seconds()
            if time_diff > 7200:  # 超过2小时
                should_close = True
                close_reason = "时间止损"
            
            if should_close:
                # 计算盈亏
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['lot_size'] * CONTRACT_SIZE
                else:
                    pnl = (position['entry_price'] - exit_price) * position['lot_size'] * CONTRACT_SIZE
                
                # 更新交易记录
                position['exit_price'] = exit_price
                position['exit_time'] = exit_time
                position['pnl'] = pnl
                position['status'] = 'closed'
                
                # 记录上一笔交易结果
                self.last_trade_result = close_reason
                
                # 更新连续亏损计数
                if pnl > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                
                # 更新账户状态
                self.balance += pnl
                self.equity = self.balance
                self.daily_pnl += pnl
                self.total_pnl += pnl
                
                # 记录交易日志
                self.trade_log.append({
                    'entry_time': position['timestamp'],
                    'exit_time': exit_time,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'lot_size': position['lot_size'],
                    'pnl': pnl,
                    'signal_type': position['signal_type'],
                    'close_reason': close_reason
                })
                
                # 从持仓中移除
                self.positions.remove(position)
                
                print(f"[AUDUSD真正优化] 平仓: {close_reason}, 盈亏: {pnl:.2f}, 余额: {self.balance:.2f}")
    
    def run_backtest(self, df):
        """
        运行回测
        
        Args:
            df (DataFrame): 历史数据
        """
        print("开始AUDUSD真正优化策略回测...")
        print(f"数据范围: {df['时间点'].iloc[0]} 至 {df['时间点'].iloc[-1]}")
        print(f"数据点数量: {len(df)}")
        print("-" * 50)
        
        for i in range(len(df) - 1):  # 留最后一根K线用于平仓
            current_data = df.iloc[i]
            next_data = df.iloc[i + 1] if i + 1 < len(df) else None
            
            # 检查日期重置
            current_date = current_data['时间点'].date()
            self.check_daily_reset(current_date)
            
            # 检查平仓条件
            self.check_exit_conditions(current_data, next_data)
            
            # 检查是否可以开仓
            if not self.can_open_position():
                continue
                
            # 生成交易信号
            signal_direction, signal_type = self.generate_signal(df, i)
            
            if signal_direction and signal_type:
                # 计算手数
                lot_size = adjust_lot_size(current_data, current_data['past_week_vol'])
                lot_size = max(MIN_LOT, min(lot_size, MAX_LOT))
                
                # 执行交易
                self.execute_trade(signal_type, signal_direction, current_data, lot_size)
        
        # 强制平仓剩余持仓
        if self.positions:
            last_data = df.iloc[-1]
            self.check_exit_conditions(last_data, None)
        
        # 输出回测结果
        self.print_results()
    
    def print_monthly_statistics(self):
        """
        打印月度统计
        """
        if not self.trade_log:
            print("没有交易记录，无法生成月度统计")
            return
        
        # 转换交易日志为DataFrame
        df_trades = pd.DataFrame(self.trade_log)
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        
        # 添加月份列
        df_trades['year_month'] = df_trades['exit_time'].dt.to_period('M')
        
        # 按月份分组统计
        monthly_stats = df_trades.groupby('year_month').agg({
            'pnl': ['count', 'sum', 'mean'],
            'direction': 'count'
        }).round(2)
        
        # 修复列名重命名的问题
        # 先展平多级列索引
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        
        # 重命名列
        monthly_stats.rename(columns={
            'pnl_count': '交易次数', 
            'pnl_sum': '月度盈亏', 
            'pnl_mean': '平均盈亏',
            'direction_count': '方向计数'
        }, inplace=True)
        
        # 计算累积收益
        monthly_stats['累积收益'] = monthly_stats['月度盈亏'].cumsum()
        
        # 计算月度收益率
        monthly_stats['月度收益率(%)'] = (monthly_stats['月度盈亏'] / INITIAL_BALANCE * 100).round(2)
        
        print("\n" + "=" * 70)
        print("月度统计")
        print("=" * 70)
        print(f"{'月份':<10} {'交易次数':<10} {'月度盈亏':<12} {'平均盈亏':<12} {'累积收益':<12} {'收益率(%)':<10}")
        print("-" * 70)
        
        cumulative_pnl = 0
        for index, row in monthly_stats.iterrows():
            # 修复：显示实际盈亏值，包括负值
            monthly_pnl = row['月度盈亏']
            cumulative_pnl += monthly_pnl
            print(f"{str(index):<10} {row['交易次数']:<10} "
                  f"{monthly_pnl:<12.2f} {row['平均盈亏']:<12.2f} "
                  f"{cumulative_pnl:<12.2f} {row['月度收益率(%)']:<10.2f}")
        
        print("-" * 70)
    
    def print_results(self):
        """
        打印回测结果
        """
        print("\n" + "=" * 50)
        print("回测结果")
        print("=" * 50)
        print(f"初始资金: ${INITIAL_BALANCE:,.2f}")
        print(f"最终资金: ${self.balance:,.2f}")
        print(f"总盈亏: ${self.total_pnl:,.2f}")
        print(f"总收益率: {(self.total_pnl / INITIAL_BALANCE) * 100:.2f}%")
        print(f"交易次数: {len(self.trade_log)}")
        
        if self.trade_log:
            win_trades = [t for t in self.trade_log if t['pnl'] > 0]
            loss_trades = [t for t in self.trade_log if t['pnl'] <= 0]
            
            print(f"盈利次数: {len(win_trades)}")
            print(f"亏损次数: {len(loss_trades)}")
            
            if len(win_trades) > 0:
                print(f"平均盈利: ${np.mean([t['pnl'] for t in win_trades]):.2f}")
                
            if len(loss_trades) > 0:
                print(f"平均亏损: ${np.mean([t['pnl'] for t in loss_trades]):.2f}")
                
            if len(win_trades) > 0 and len(loss_trades) > 0:
                win_rate = len(win_trades) / len(self.trade_log) * 100
                print(f"胜率: {win_rate:.2f}%")
                
                avg_win = np.mean([t['pnl'] for t in win_trades])
                avg_loss = abs(np.mean([t['pnl'] for t in loss_trades]))
                profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
                print(f"盈亏比: {profit_factor:.2f}")
        
        print("-" * 50)
        
        # 打印月度统计
        self.print_monthly_statistics()
        
        # 打印最近20条交易明细
        if self.trade_log:
            print("\n最近20条交易记录:")
            print("-" * 120)
            print(f"{'入场时间':<20} {'出场时间':<20} {'方向':<6} {'入场价':<8} {'出场价':<8} {'手数':<6} {'盈亏':<10} {'信号类型':<12} {'平仓原因':<8}")
            print("-" * 120)
            
            # 只显示最近20条记录
            recent_trades = self.trade_log[-20:] if len(self.trade_log) > 20 else self.trade_log
            for trade in recent_trades:
                print(f"{str(trade['entry_time']):<20} {str(trade['exit_time']):<20} "
                      f"{trade['direction']:<6} {trade['entry_price']:<8.5f} {trade['exit_price']:<8.5f} "
                      f"{trade['lot_size']:<6.1f} {trade['pnl']:<10.2f} {trade['signal_type']:<12} {trade['close_reason']:<8}")

# ======================== 4. 主函数 ========================
def main():
    # 直接从MT5获取数据进行回测
    print("正在从MT5获取实时历史数据进行真正优化回测...")
    df = load_historical_data_from_mt5("AUDUSD", mt5.TIMEFRAME_H1, 730)
    
    if df is None or len(df) == 0:
        print("错误: 无法从MT5获取历史数据")
        return
    
    print(f"成功加载 {len(df)} 条历史数据")
    
    # 创建策略实例并运行回测
    strategy = AUDUSDStrategy()
    strategy.run_backtest(df)

if __name__ == "__main__":
    main()
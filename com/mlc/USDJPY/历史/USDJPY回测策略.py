"""
USDJPY专用回测策略
基于USDJPY历史数据特征开发的专用交易策略
注意：USDJPY的计价货币是日元，需要将盈亏转换为美元计价
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from collections import defaultdict

# 添加utils目录到Python路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
utils_path = os.path.abspath(utils_path)
sys.path.insert(0, utils_path)

# 导入共享状态（用于FTMO规则）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from shared_state import shared_state
except ImportError:
    # 如果无法导入共享状态，则创建一个简单的替代方案
    class SharedState:
        def __init__(self):
            pass
    
    shared_state = SharedState()

# ======================== 1. 初始化参数 ========================
# 交易参数（USDJPY标准合约）
contract_size = 100000  # 1标准手=10万日元
min_lot = 1.0  # 最小仓位
max_lot = 3.0  # 最大仓位调整为3手
symbol = "USDJPY"  # 交易品种

# USD到JPY的转换参数（用于将盈亏转换为美元计价）
# 这里使用一个近似的汇率，实际应用中应该动态获取
usd_jpy_rate = 145.0  # 1美元约等于145日元

# FTMO风险管理参数
initial_balance = 100000.0  # 初始资金
daily_loss_limit = 0.05 * initial_balance  # 每日最大亏损5%
total_loss_limit = 0.1 * initial_balance   # 总最大亏损10%
min_equity_limit = 0.9 * initial_balance   # 最低净值90%

# 立即刷新输出
sys.stdout.reconfigure(encoding='utf-8')
print("开始执行USDJPY回测策略...")
sys.stdout.flush()

# ======================== 2. 订单类 ========================
class TradeOrder:
    def __init__(self, order_id, timestamp, direction, lot_size, entry_price, sl, tp):
        self.order_id = order_id  # 订单ID
        self.timestamp = timestamp  # 入场时间
        self.direction = direction  # 方向：long/short
        self.lot_size = lot_size  # 手数
        self.entry_price = entry_price  # 入场价
        self.sl = sl  # 止损价
        self.tp = tp  # 止盈价
        self.exit_price = None  # 出场价
        self.exit_timestamp = None  # 出场时间
        self.pnl = 0.0  # 盈亏（美元）
        self.status = "open"  # 状态：open/closed

    def close(self, exit_price, exit_timestamp, usd_jpy_rate=145.0):
        """平仓并计算实际盈亏（转换为美元计价）"""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.status = "closed"

        # 外汇盈亏公式：(平仓价-入场价)×手数×合约单位（做多）；(入场价-平仓价)×手数×合约单位（做空）
        if self.direction == "long":
            jpy_pnl = (exit_price - self.entry_price) * self.lot_size * contract_size
        else:
            jpy_pnl = (self.entry_price - exit_price) * self.lot_size * contract_size

        # 将日元盈亏转换为美元盈亏
        self.pnl = jpy_pnl / usd_jpy_rate

        return self.pnl

# ======================== 3. 工具函数 ========================
def adjust_lot_size(current_row, past_week_vol, win_rate=0.5):
    """动态调整仓位（1-3手，基于信号强度）"""
    # 基于胜率和波动率调整仓位
    base_lot = min_lot
    
    # 根据胜率调整仓位
    if win_rate > 0.6:  # 胜率较高时增加仓位
        base_lot += 0.5
    if win_rate > 0.7:  # 胜率很高时进一步增加仓位
        base_lot += 0.5
        
    # 根据波动率调整仓位
    if not pd.isna(past_week_vol) and past_week_vol > 0:
        vol_ratio = current_row['当前波动幅度'] / past_week_vol if past_week_vol > 0 else 1
        if vol_ratio > 1.5:  # 波动很大时减仓
            base_lot *= 0.7
        elif vol_ratio < 0.5:  # 波动很小时可以适当加仓
            base_lot *= 1.2

    return max(min_lot, min(base_lot, max_lot))

def calculate_sl_tp(current_row, direction, atr_value):
    """计算止损止盈（基于ATR动态调整，考虑点差）"""
    # 处理时间格式：确保是字符串格式
    time_val = current_row['时间点']
    if isinstance(time_val, pd.Timestamp):
        hour = time_val.hour
    else:
        time_str = str(time_val)
        hour = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").hour

    # 根据USDJPY的特性调整止损止盈倍数
    # 美元日元通常波动较小，需要更精细的止损止盈设置
    if 0 <= hour <= 6 or 22 <= hour <= 23:  # 亚洲时段，波动较小
        sl_multiplier = 1.2
        tp_multiplier = 2.0
    elif 7 <= hour <= 10 or 15 <= hour <= 18:  # 欧美重叠时段，波动较大
        sl_multiplier = 1.8
        tp_multiplier = 2.7
    else:  # 其他时段
        sl_multiplier = 1.5
        tp_multiplier = 2.3

    # 使用ATR计算止损止盈
    sl_distance = sl_multiplier * atr_value
    tp_distance = tp_multiplier * atr_value
    
    # USDJPY典型点差约为0.3-0.5点
    spread = 0.4  # 点差
    
    # 按多空方向计算点位（考虑点差）
    if direction == "long":
        sl = current_row['当前开盘价'] - sl_distance - spread  # 止损考虑点差
        tp = current_row['当前开盘价'] + tp_distance  # 止盈不考虑点差（实际到账）
    else:
        sl = current_row['当前开盘价'] + sl_distance + spread  # 止损考虑点差
        tp = current_row['当前开盘价'] - tp_distance  # 止盈不考虑点差（实际到账）

    # 四舍五入到合适的小数位数（USDJPY通常为3位小数）
    sl = round(sl, 3)
    tp = round(tp, 3)
    
    return sl, tp

def check_risk_limits(daily_pnl, total_pnl, equity):
    """检查是否超出风险限制"""
    # 检查各项风险指标
    if daily_pnl <= -daily_loss_limit:
        return False, "日亏损超限"
    if total_pnl <= -total_loss_limit:
        return False, "总亏损超限"
    if equity <= min_equity_limit:
        return False, "净值低于限制"
    return True, "正常"

# ======================== 4. 核心交易逻辑 ========================
def run_backtest(data_file):
    """运行回测"""
    # 读取历史数据
    print("正在读取USDJPY历史数据...")
    df = pd.read_csv(data_file, encoding='utf-8')
    
    # 转换时间列为datetime类型
    df['时间点'] = pd.to_datetime(df['时间点'])
    
    # 重命名列以匹配策略逻辑
    df.rename(columns={
        '昨天增减': '昨天增减',
        '昨日波动幅度': '昨日波动幅度',
        '当前开盘价': '当前开盘价',
        '当前收盘价': '当前收盘价',
        '当前最高价': '当前最高价',
        '当前最低价': '当前最低价',
        '当前增减': '当前增减',
        '当前波动幅度': '当前波动幅度'
    }, inplace=True)
    
    # 添加技术指标
    df['MA5'] = df['当前收盘价'].rolling(window=5, min_periods=5).mean()
    df['MA10'] = df['当前收盘价'].rolling(window=10, min_periods=10).mean()
    df['MA20'] = df['当前收盘价'].rolling(window=20, min_periods=20).mean()
    df['MA50'] = df['当前收盘价'].rolling(window=50, min_periods=50).mean()
    
    # 计算ATR（平均真实波幅）
    df['TR'] = df.apply(lambda x: max(
        x['当前最高价'] - x['当前最低价'], 
        abs(x['当前最高价'] - x['当前收盘价']), 
        abs(x['当前最低价'] - x['当前收盘价'])
    ), axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=14).mean()
    
    df['past_week_vol'] = df['当前波动幅度'].rolling(window=168, min_periods=168).mean()  # 7天波动率
    df['ATR_MA50'] = df['ATR'].rolling(window=50, min_periods=50).mean()  # ATR的50周期均线用于波动率过滤
    
    # USDJPY特定指标
    # 价格变化率
    df['Price_Change_Rate'] = df['当前收盘价'].pct_change(1)
    # 波动率指标
    df['Volatility_Signal'] = df['ATR'] / df['当前收盘价']  # ATR与价格的比率
    
    # 只保留有完整指标的数据
    df = df.dropna().reset_index(drop=True)
    
    print(f"成功读取 {len(df)} 条数据")
    
    # 初始化账户和统计数据
    balance = initial_balance
    equity = initial_balance
    daily_pnl = 0.0
    total_pnl = 0.0
    orders = []
    current_position = None
    order_id_counter = 1
    last_trading_day = None
    daily_trades = 0
    max_daily_trades = 3  # 每日最大交易次数
    
    # 记录最近一次平仓时间，用于确保等待下一个完整的H1 K线
    last_exit_time = None
    
    # 月度统计
    monthly_stats = defaultdict(lambda: {
        'pnl': 0.0,
        'trades': 0,
        'winning_trades': 0,
        'losing_trades': 0
    })
    
    # 统计变量
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    max_drawdown = 0.0
    peak_balance = initial_balance
    
    print("开始回测...")
    
    # 遍历数据进行回测
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 获取当前时间
        current_time = current_row['时间点']
        current_date = current_time.date()
        current_month = current_time.strftime("%Y-%m")
        
        # 每日初始化
        if last_trading_day != current_date:
            last_trading_day = current_date
            daily_trades = 0
            daily_pnl = 0.0
            print(f"\n{current_date} 开始交易")
        
        # 检查风险限制
        risk_ok, risk_reason = check_risk_limits(daily_pnl, total_pnl, equity)
        if not risk_ok:
            if current_position:
                # 强制平仓
                exit_price = current_row['当前开盘价']
                current_position.close(exit_price, current_time, usd_jpy_rate)
                orders.append(current_position)
                balance += current_position.pnl
                equity = balance
                daily_pnl += current_position.pnl
                total_pnl += current_position.pnl
                total_trades += 1
                
                # 更新月度统计
                monthly_stats[current_month]['pnl'] += current_position.pnl
                monthly_stats[current_month]['trades'] += 1
                if current_position.pnl > 0:
                    winning_trades += 1
                    monthly_stats[current_month]['winning_trades'] += 1
                else:
                    losing_trades += 1
                    monthly_stats[current_month]['losing_trades'] += 1
                
                print(f"[{current_time}] 因{risk_reason}强制平仓: 盈亏 {current_position.pnl:.2f} USD")
                # 记录平仓时间，启动冷却期
                last_exit_time = current_time
                current_position = None
            continue  # 超出风险限制，跳过本次交易
        
        # 如果没有持仓且未达到每日交易限制，则检查入场条件
        # 注意：如果刚刚平仓，则需要等待下一个完整的H1 K线才能重新开仓
        if current_position is None and daily_trades < max_daily_trades:
            # 检查是否处于平仓后的冷却期（必须等待到下一个完整的H1 K线）
            in_cooling_period = False
            if last_exit_time is not None:
                # 计算从上次平仓到当前时间的间隔
                time_since_exit = current_time - last_exit_time
                # 必须至少经过1小时才能重新开仓
                if time_since_exit < timedelta(hours=1):
                    # 处于冷却期，跳过本次交易信号检查
                    in_cooling_period = True
                else:
                    # 冷却期已过，可以重新评估交易信号
                    last_exit_time = None  # 重置冷却期标记
           
            # 如果不在冷却期，则检查入场条件
            if not in_cooling_period and last_exit_time is None:
                signal_type = None
                direction = None
                
                # 计算当前胜率
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
                
                # USDJPY特定过滤条件
                # 1. 波动率过滤：避免在波动率极低或极高时交易（适度放宽）
                volatility_filter = (current_row['Volatility_Signal'] > df['Volatility_Signal'].quantile(0.1)) and \
                                   (current_row['Volatility_Signal'] < df['Volatility_Signal'].quantile(0.9))
                
                # 2. 趋势过滤：只在趋势明确时交易（适度放宽）
                trend_filter = abs(current_row['当前增减']) > current_row['ATR'] * 0.3
                
                # 3. 时间过滤：避免在特定时段交易
                current_hour = current_time.hour
                time_filter = not (7 <= current_hour <= 8 or 15 <= current_hour <= 16)  # 避免重要数据发布时间
                
                # USDJPY策略信号
                # 1. 均线突破信号
                if (current_row['当前收盘价'] > current_row['MA20'] and 
                    prev_row['当前收盘价'] <= prev_row['MA20'] and
                    current_row['MA5'] > current_row['MA20'] and
                    volatility_filter and trend_filter and time_filter):
                    signal_type = "多头突破"
                    direction = "long"
                    
                elif (current_row['当前收盘价'] < current_row['MA20'] and 
                      prev_row['当前收盘价'] >= prev_row['MA20'] and
                      current_row['MA5'] < current_row['MA20'] and
                      volatility_filter and trend_filter and time_filter):
                    signal_type = "空头突破"
                    direction = "short"
                
                # 2. 均线金叉死叉信号
                elif (current_row['MA5'] > current_row['MA10'] and 
                      prev_row['MA5'] <= prev_row['MA10'] and
                      current_row['当前收盘价'] > current_row['MA50'] and
                      volatility_filter and trend_filter and time_filter):
                    signal_type = "MA金叉"
                    direction = "long"
                    
                elif (current_row['MA5'] < current_row['MA10'] and 
                      prev_row['MA5'] >= prev_row['MA10'] and
                      current_row['当前收盘价'] < current_row['MA50'] and
                      volatility_filter and trend_filter and time_filter):
                    signal_type = "MA死叉"
                    direction = "short"
                
                # 3. 强趋势信号
                elif (abs(current_row['当前增减']) > current_row['ATR'] * 1.2 and
                      current_row['Price_Change_Rate'] * current_row['昨天增减'] > 0 and
                      volatility_filter and time_filter):
                    if current_row['当前增减'] > 0:
                        signal_type = "强势上涨"
                        direction = "long"
                    else:
                        signal_type = "强势下跌"
                        direction = "short"
                
                # 如果有信号，则开仓
                if signal_type and direction:
                    # 计算手数
                    current_lot = adjust_lot_size(current_row, current_row['past_week_vol'], win_rate)
                    current_lot = max(min_lot, min(current_lot, max_lot))
                    
                    # 计算止损止盈（使用ATR，考虑点差）
                    sl, tp = calculate_sl_tp(current_row, direction, current_row['ATR'])
                    
                    # 获取入场价格（考虑点差）
                    entry_price = current_row['当前开盘价']
                    
                    # 开仓
                    current_position = TradeOrder(
                        order_id=order_id_counter,
                        timestamp=current_time,
                        direction=direction,
                        lot_size=current_lot,
                        entry_price=entry_price,
                        sl=sl,
                        tp=tp
                    )
                    order_id_counter += 1
                    daily_trades += 1
                    print(f"[{current_time}] 开仓: {direction}, 信号: {signal_type}, 入场价: {entry_price}, 止损: {sl}, 止盈: {tp}, 手数: {current_lot:.1f}")
        
        # 如果有持仓，检查平仓条件
        elif current_position is not None:
            current_price = current_row['当前收盘价']
            exit_condition = False
            exit_reason = ""
            exit_price = current_price
            
            # 检查止损
            if current_position.direction == "long" and current_price <= current_position.sl:
                exit_condition = True
                exit_reason = "止损"
                exit_price = current_position.sl
            elif current_position.direction == "short" and current_price >= current_position.sl:
                exit_condition = True
                exit_reason = "止损"
                exit_price = current_position.sl
                
            # 检查止盈
            elif current_position.direction == "long" and current_price >= current_position.tp:
                exit_condition = True
                exit_reason = "止盈"
                exit_price = current_position.tp
            elif current_position.direction == "short" and current_price <= current_position.tp:
                exit_condition = True
                exit_reason = "止盈"
                exit_price = current_position.tp
                
            # 检查时间止损（持仓超过12小时）
            elif (current_time - current_position.timestamp).total_seconds() > 12 * 3600:
                exit_condition = True
                exit_reason = "时间止损"
                exit_price = current_price
                
            # 平仓
            if exit_condition:
                current_position.close(exit_price, current_time, usd_jpy_rate)
                orders.append(current_position)
                balance += current_position.pnl
                equity = balance
                daily_pnl += current_position.pnl
                total_pnl += current_position.pnl
                total_trades += 1
                
                # 更新月度统计
                monthly_stats[current_month]['pnl'] += current_position.pnl
                monthly_stats[current_month]['trades'] += 1
                if current_position.pnl > 0:
                    winning_trades += 1
                    monthly_stats[current_month]['winning_trades'] += 1
                else:
                    losing_trades += 1
                    monthly_stats[current_month]['losing_trades'] += 1
                
                print(f"[{current_time}] 平仓: {exit_reason}, 盈亏 {current_position.pnl:.2f} USD")
                
                # 记录平仓时间，启动冷却期
                last_exit_time = current_time
                current_position = None
                
                # 更新峰值和最大回撤
                peak_balance = max(peak_balance, balance)
                drawdown = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
    
    # 如果回测结束时还有持仓，强制平仓
    if current_position is not None:
        last_row = df.iloc[-1]
        exit_price = last_row['当前收盘价']
        current_position.close(exit_price, last_row['时间点'], usd_jpy_rate)
        orders.append(current_position)
        balance += current_position.pnl
        equity = balance
        daily_pnl += current_position.pnl
        total_pnl += current_position.pnl
        total_trades += 1
        
        # 更新月度统计
        current_month = last_row['时间点'].strftime("%Y-%m")
        monthly_stats[current_month]['pnl'] += current_position.pnl
        monthly_stats[current_month]['trades'] += 1
        if current_position.pnl > 0:
            winning_trades += 1
            monthly_stats[current_month]['winning_trades'] += 1
        else:
            losing_trades += 1
            monthly_stats[current_month]['losing_trades'] += 1
            
        print(f"[{last_row['时间点']}] 强制平仓: 盈亏 {current_position.pnl:.2f} USD")
        
        # 记录平仓时间，启动冷却期
        last_exit_time = last_row['时间点']
    
    # 输出回测结果
    print("\n" + "="*50)
    print("回测结果")
    print("="*50)
    print(f"初始资金: ${initial_balance:,.2f} USD")
    print(f"最终资金: ${balance:,.2f} USD")
    print(f"总盈亏: ${total_pnl:,.2f} USD")
    print(f"总收益率: {total_pnl / initial_balance * 100:.2f}%")
    print(f"总交易次数: {total_trades}")
    print(f"盈利次数: {winning_trades}")
    print(f"亏损次数: {losing_trades}")
    print(f"胜率: {winning_trades / total_trades * 100:.2f}%") if total_trades > 0 else print("胜率: 0%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    
    # 输出月度统计
    print("\n" + "="*50)
    print("月度统计")
    print("="*50)
    for month, stats in sorted(monthly_stats.items()):
        win_rate = stats['winning_trades'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
        print(f"{month}: 交易 {stats['trades']} 次, 盈亏 ${stats['pnl']:,.2f} USD, 胜率 {win_rate:.2f}%")
    
    # 输出最近20条订单记录
    print("\n" + "="*50)
    print("最近20条订单记录")
    print("="*50)
    for order in orders[-20:]:
        if order.status == "closed":
            print(f"[{order.exit_timestamp}] 平仓: {order.direction}, 盈亏 {order.pnl:.2f} USD")
    
    # 强制刷新输出
    sys.stdout.flush()
    
    # 检查是否满足FTMO挑战目标
    profit_target = initial_balance * 0.1  # 10%盈利目标
    if total_pnl >= profit_target:
        print("\n✅ 达成FTMO挑战目标（10%盈利）")
    else:
        print(f"\n❌ 未达成FTMO挑战目标，还需盈利: ${profit_target - total_pnl:,.2f} USD")
    
    # 检查风险控制
    if total_pnl <= -total_loss_limit:
        print("❌ 触发总亏损限制（10%）")
    elif daily_pnl <= -daily_loss_limit:
        print("❌ 触发单日亏损限制（5%）")
    elif equity <= min_equity_limit:
        print("❌ 触发净值限制（90%）")
    else:
        print("✅ 通过所有风险控制检查")
    
    return {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_pnl': total_pnl,
        'total_return': total_pnl / initial_balance * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'max_drawdown': max_drawdown,
        'orders': orders,
        'monthly_stats': monthly_stats
    }

# ======================== 5. 主函数 ========================
if __name__ == "__main__":
    # 数据文件路径
    data_file = os.path.join(os.path.dirname(__file__), "USDJPY_H1_历史数据.csv")
    
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 {data_file}")
        sys.exit(1)
    
    try:
        results = run_backtest(data_file)
    except Exception as e:
        print(f"回测过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

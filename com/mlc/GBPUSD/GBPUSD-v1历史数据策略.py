import pandas as pd
from datetime import datetime

# ======================== 1. 初始化参数（真实交易模拟） ========================
# 账户信息（FTMO 10万账户）
initial_balance = 100000.0  # 初始资金
current_balance = initial_balance
equity = initial_balance  # 净值 = 余额 + 未平仓盈亏

# 交易参数（GBPUSD标准合约）
contract_size = 100000  # 1标准手=10万英镑
min_lot = 2.0  # 最小仓位
max_lot = 3.0  # 最大仓位

# 风险控制参数（FTMO规则）
daily_loss_limit = 0.05 * initial_balance  # 每日最大亏损5%
total_loss_limit = 0.1 * initial_balance  # 总最大亏损10%
min_equity_limit = 0.9 * initial_balance  # 最低净值90%

# 订单记录
orders = []  # 所有订单历史
current_position = None  # 当前持仓（None为空仓）
daily_trades = 0  # 当日交易次数
daily_pnl = 0.0  # 当日盈亏
total_pnl = 0.0  # 总盈亏
last_trading_day = None  # 上一交易日


# ======================== 2. 订单类（模拟真实订单） ========================
class TradeOrder:
    def __init__(self, order_id, timestamp, direction, lot_size, entry_price, sl, tp):
        self.order_id = order_id  # 订单ID
        self.timestamp = timestamp  # 入场时间（str格式）
        self.direction = direction  # 方向：long/short
        self.lot_size = lot_size  # 手数
        self.entry_price = entry_price  # 入场价
        self.sl = sl  # 止损价
        self.tp = tp  # 止盈价
        self.exit_price = None  # 出场价
        self.exit_timestamp = None  # 出场时间
        self.pnl = 0.0  # 盈亏
        self.status = "open"  # 状态：open/closed

    def close(self, exit_price, exit_timestamp):
        """平仓并计算实际盈亏（基于真实点数）"""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.status = "closed"

        # 外汇盈亏公式：(平仓价-入场价)×手数×合约单位（做多）；(入场价-平仓价)×手数×合约单位（做空）
        if self.direction == "long":
            self.pnl = (exit_price - self.entry_price) * self.lot_size * contract_size
        else:
            self.pnl = (self.entry_price - exit_price) * self.lot_size * contract_size

        return self.pnl


# ======================== 3. 工具函数（提前测试无错误） ========================
def adjust_lot_size(current_row, past_week_vol):
    """动态调整仓位（1-3手，基于趋势强度）"""
    if pd.isna(past_week_vol) or current_row['当前波动幅度'] == 0:
        return min_lot

    # 趋势强度=当前价格变动/当前波动幅度（越大趋势越明确）
    trend_strength = abs(current_row['当前增减']) / current_row['当前波动幅度']

    # 趋势明确且波动较小时用3手，否则1手
    return max_lot if (trend_strength > 0.6 and current_row['当前波动幅度'] < past_week_vol) else min_lot


def calculate_sl_tp(current_row, direction):
    """计算止损止盈（真实点位，按时间段调整流动性）"""
    # 处理时间格式：Timestamp→str→提取小时（避免strptime错误）
    time_str = current_row['时间点'].strftime("%Y-%m-%d %H:%M:%S")
    hour = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").hour

    # 流动性低时段（凌晨0-5点）：收紧止损止盈
    if 0 <= hour <= 5:
        sl_multiplier = 0.3
        tp_multiplier = 0.6
    else:
        sl_multiplier = 0.5
        tp_multiplier = 1.5

    # 按多空方向计算点位
    if direction == "long":
        sl = current_row['当前开盘价'] - sl_multiplier * current_row['当前波动幅度']
        tp = current_row['当前开盘价'] + tp_multiplier * current_row['当前波动幅度']
    else:
        sl = current_row['当前开盘价'] + sl_multiplier * current_row['当前波动幅度']
        tp = current_row['当前开盘价'] - tp_multiplier * current_row['当前波动幅度']

    return round(sl, 5), round(tp, 5)  # 保留5位小数（外汇标准精度）


def calculate_pnl(entry_price, exit_price, direction, lot_size):
    """根据止盈止损价格计算实际盈亏"""
    if direction == "long":
        return (exit_price - entry_price) * lot_size * contract_size
    else:
        return (entry_price - exit_price) * lot_size * contract_size


# ======================== 4. 数据预处理（确保格式正确） ========================
# 读取数据（请替换为你的实际文件路径！！！）
# 示例路径：C:/Users/HS/Desktop/GBPUSD_H1_历史数据.csv（桌面路径）
df = pd.read_csv('GBPUSD_H1_历史数据.csv')

# 强制时间列为datetime格式（避免混合格式错误）
df['时间点'] = pd.to_datetime(df['时间点'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

# 删除时间格式错误的行（确保数据有效性）
df = df.dropna(subset=['时间点'])

# 计算技术指标（用于入场信号，最小周期确保有值）
df['MA5'] = df['当前收盘价'].rolling(window=5, min_periods=5).mean()
df['MA10'] = df['当前收盘价'].rolling(window=10, min_periods=10).mean()

# 计算过去7天（168小时）平均波动幅度（判断市场稳定性）
df['past_week_vol'] = df['当前波动幅度'].rolling(window=168, min_periods=168).mean()

# 只保留有完整指标的数据（避免空值错误）
df = df.dropna(subset=['MA5', 'MA10', 'past_week_vol'])

# ======================== 5. 核心交易逻辑（逐小时模拟，本地测试通过） ========================
order_id_counter = 1  # 订单ID自增

for idx, current in df.iterrows():
    # 提取当前/前一小时数据
    if idx == 0:
        last_trading_day = current['时间点'].date()  # 初始化上一交易日
        continue  # 跳过第一行（无历史数据）
    
    # 安全地获取前一行数据
    try:
        prev = df.iloc[idx - 1]
    except IndexError:
        # 如果索引越界，则跳过当前循环
        continue
    
    # 检查是否有足够的历史数据用于周三信号判断
    has_enough_history = idx >= 2

    # 时间处理（统一为str格式，避免类型错误）
    current_time = current['时间点']
    current_date = current_time.date()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # 每日初始化（新交易日重置统计）
    if last_trading_day is None:
        last_trading_day = current_date
    elif current_date != last_trading_day:
        daily_trades = 0
        daily_pnl = 0.0
        last_trading_day = current_date

    # -------------------- 风险控制：禁止违规开仓 --------------------
    # 当日亏损≥5% 或 总亏损≥10% 或 净值<90%：禁止开仓
    if (daily_pnl <= -daily_loss_limit) or \
            (current_balance - initial_balance <= -total_loss_limit) or \
            (equity < min_equity_limit):
        current_lot = 0.0
    else:
        current_lot = adjust_lot_size(current, current['past_week_vol'])
        
    # 确保仓位在合理范围内
    current_lot = max(min_lot, min(current_lot, max_lot))

    # -------------------- 空仓：检查入场条件（多信号触发） --------------------
    if current_position is None and current_lot > 0:
        signal_type = None
        direction = None
        
        # 1. 周一→周二趋势延续
        if prev['星期几'] == '星期一' and current['星期几'] == '星期二':
            # 周一上涨→周二延续：做多
            if (prev['昨天增减'] > 0 and
                    current['当前开盘价'] > prev['当前收盘价'] and
                    current['当前增减'] > 0):
                signal_type = "周一上涨周二延续"
                direction = "long"

            # 周一下跌→周二延续：做空
            elif (prev['昨天增减'] < 0 and
                  current['当前开盘价'] < prev['当前收盘价'] and
                  current['当前增减'] < 0):
                signal_type = "周一下跌周二延续"
                direction = "short"

        # 2. 周三反转信号
        elif current['星期几'] == '星期三' and has_enough_history:
            # 前两日下跌→周三反转：做多
            if (df.iloc[idx - 2]['当前增减'] < 0 and
                    prev['当前增减'] < 0 and
                    current['当前开盘价'] > prev['当前收盘价'] and
                    current['当前增减'] > 0):
                signal_type = "周三反转做多"
                direction = "long"

            # 前两日上涨→周三反转：做空
            elif (df.iloc[idx - 2]['当前增减'] > 0 and
                  prev['当前增减'] > 0 and
                  current['当前开盘价'] < prev['当前收盘价'] and
                  current['当前增减'] < 0):
                signal_type = "周三反转做空"
                direction = "short"

        # 3. MA金叉/死叉（技术信号）
        elif current['MA5'] > current['MA10'] and prev['MA5'] <= prev['MA10']:
            signal_type = "MA金叉"
            direction = "long"
            
        elif current['MA5'] < current['MA10'] and prev['MA5'] >= prev['MA10']:
            signal_type = "MA死叉"
            direction = "short"
            
        # 如果有信号，则开仓
        if signal_type and direction:
            sl, tp = calculate_sl_tp(current, direction)
            current_position = TradeOrder(
                order_id=order_id_counter,
                timestamp=current_time_str,
                direction=direction,
                lot_size=current_lot,
                entry_price=current['当前开盘价'],
                sl=sl,
                tp=tp
            )
            orders.append(current_position)
            order_id_counter += 1
            print(f"[{current_time_str}] 开{'' if direction=='long' else '空'}单{current_position.order_id}：{signal_type}，价{current['当前开盘价']}，"
                  f"止损{sl}，止盈{tp}，手数{current_lot}")

    # -------------------- 有持仓：检查平仓条件（止盈/止损） --------------------
    if current_position is not None:
        current_close = current['当前收盘价']

        # 检查止盈止损
        exit_price = None
        exit_reason = ""
        
        # 止盈平仓
        if (current_position.direction == "long" and current_close >= current_position.tp) or \
                (current_position.direction == "short" and current_close <= current_position.tp):
            exit_price = current_position.tp
            exit_reason = "止盈"

        # 止损平仓
        elif (current_position.direction == "long" and current_close <= current_position.sl) or \
                (current_position.direction == "short" and current_close >= current_position.sl):
            exit_price = current_position.sl
            exit_reason = "止损"
            
        # 如果需要平仓
        if exit_price is not None:
            pnl = current_position.close(exit_price, current_time_str)
            daily_pnl += pnl
            total_pnl += pnl
            current_balance += pnl
            daily_trades += 1
            print(
                f"[{current_time_str}] 平单{current_position.order_id}：{exit_reason}，盈亏{pnl:.2f}元，余额{current_balance:.2f}元")
            current_position = None
            
        # 添加时间止损机制（最大持仓24小时）
        from datetime import datetime
        if current_position is not None:
            entry_time = datetime.strptime(current_position.timestamp, "%Y-%m-%d %H:%M:%S")
            current_time_obj = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S")
            time_diff = current_time_obj - entry_time
            if time_diff.total_seconds() > 24 * 3600:  # 超过24小时
                # 使用当前收盘价平仓
                pnl = current_position.close(current_close, current_time_str)
                daily_pnl += pnl
                total_pnl += pnl
                current_balance += pnl
                daily_trades += 1
                print(
                    f"[{current_time_str}] 平单{current_position.order_id}：时间止损，盈亏{pnl:.2f}元，余额{current_balance:.2f}元")
                current_position = None

    # 更新净值（含未平仓盈亏）
    if current_position is not None:
        if current_position.direction == "long":
            unrealized_pnl = (current_close - current_position.entry_price) * current_position.lot_size * contract_size
        else:
            unrealized_pnl = (current_position.entry_price - current_close) * current_position.lot_size * contract_size
        equity = current_balance + unrealized_pnl
    else:
        equity = current_balance
        
    # 确保只持有一个仓位的逻辑正确
    if current_position is not None and current_lot > 0:
        # 如果已经有仓位，则不允许再开新仓，将当前仓位设为0以防止新仓 opening
        current_lot = 0.0

# ======================== 6. 最终结果统计（清晰展示） ========================
print("\n" + "=" * 80)
print("                    GBPUSD策略回测结果（FTMO 10万账户）")
print("=" * 80)
print(f"初始资金：{initial_balance:,.2f} 元")
print(f"最终资金：{current_balance:,.2f} 元")
print(f"总盈亏：{total_pnl:,.2f} 元")
print(f"总交易次数：{len(orders)} 次")

# 盈利/亏损订单统计
winning_orders = [o for o in orders if o.pnl > 0]
losing_orders = [o for o in orders if o.pnl <= 0]
print(f"盈利订单：{len(winning_orders)} 次，平均盈利：{sum(o.pnl for o in winning_orders) / len(winning_orders):,.2f} 元"
      if winning_orders else "盈利订单：0 次")
print(f"亏损订单：{len(losing_orders)} 次，平均亏损：{sum(o.pnl for o in losing_orders) / len(losing_orders):,.2f} 元"
      if losing_orders else "亏损订单：0 次")

# 按月统计盈亏
print("\n" + "-" * 50)
print("                   每月收益统计")
print("-" * 50)

# 创建一个字典来存储每月的盈亏
monthly_pnl_dict = {}

# 遍历所有订单，按月份统计盈亏
for order in orders:
    if order.exit_timestamp:
        # 从订单平仓时间中提取年月
        timestamp = datetime.strptime(order.exit_timestamp, "%Y-%m-%d %H:%M:%S")
        month_key = timestamp.strftime("%Y-%m")
        
        # 累计每月盈亏
        if month_key in monthly_pnl_dict:
            monthly_pnl_dict[month_key] += order.pnl
        else:
            monthly_pnl_dict[month_key] = order.pnl

# 按时间顺序排序并打印每月盈亏
sorted_months = sorted(monthly_pnl_dict.keys())
for month in sorted_months:
    pnl = monthly_pnl_dict[month]
    print(f"{month}: {pnl:,.2f} 元 {'📈' if pnl > 0 else '📉'}")

# 打印每月平均收益
if monthly_pnl_dict:
    avg_monthly_pnl = sum(monthly_pnl_dict.values()) / len(monthly_pnl_dict)
    print(f"\n月平均收益: {avg_monthly_pnl:,.2f} 元")

# 回测周期与月收益（按实际天数计算）
first_date = df['时间点'].iloc[0].date()
last_date = df['时间点'].iloc[-1].date()
total_days = (last_date - first_date).days
if total_days > 0:
    daily_avg_pnl = total_pnl / total_days
    monthly_pnl = daily_avg_pnl * 30  # 月均按30天
    print(f"\n回测周期：{total_days} 天")
    print(f"日均盈亏：{daily_avg_pnl:,.2f} 元")
    print(f"月均收益：{monthly_pnl:,.2f} 元")
    print(f"月均收益率：{(monthly_pnl / initial_balance) * 100:.2f}%")
    print(f"是否满足FTMO目标（月收益≥10%）：{'✅ 是' if monthly_pnl >= 0.1 * initial_balance else '❌ 否'}")
print("=" * 80)
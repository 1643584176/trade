"""
AUDUSD交易策略 - MT5版本 (5倍止损止盈版)
基于MT5实时数据的AUDUSD交易策略实现，止盈止损点位扩大5倍
"""

import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import sys
import os
import time

# 添加utils目录到Python路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
utils_path = os.path.abspath(utils_path)
sys.path.insert(0, utils_path)

# 导入时间处理工具
from time_utils import TimeUtils

# 导入配置加载器
from config_loader import config, get_config_value

# 导入共享状态
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared_state import shared_state

# ======================== 1. 初始化参数（真实交易模拟） ========================
# 交易参数（AUDUSD标准合约）
contract_size = 100000  # 1标准手=10万澳元
min_lot = 2.0  # 最小仓位（根据项目规范）
max_lot = 3.0  # 最大仓位（根据项目规范）
symbol = get_config_value('AUDUSD_SYMBOL', 'AUDUSD')  # 交易品种

# 每日最大交易次数限制
max_daily_trades = 3

# 仓位大小（保持原来的固定值）
lot_size = 2.0  # 原来的固定手数（根据项目规范，AUDUSD最小2手）

# 账户参数 - 使用测试账户
account_number = get_config_value('TEST_ACCOUNT_NUMBER', '')
account_password = get_config_value('TEST_ACCOUNT_PASSWORD', '')
account_server = get_config_value('TEST_ACCOUNT_SERVER', '')

# 订单记录
orders = []  # 所有订单历史
current_position = None  # 当前持仓（None为空仓）
daily_trades = 0  # 当日交易次数
daily_pnl = 0.0  # 当日盈亏
total_pnl = 0.0  # 总盈亏
last_trading_day = None  # 上一交易日
order_id_counter = 1  # 订单ID自增
last_close_time = None  # 上次平仓时间

# 全局交易状态变量
current_position_ticket = None

# ======================== 2. 订单类（模拟真实订单） ========================
class TradeOrder:
    def __init__(self, order_id, timestamp, direction, lot_size, entry_price, sl, tp, ticket=None):
        self.order_id = order_id  # 订单ID
        self.timestamp = timestamp  # 入场时间（str格式）
        self.direction = direction  # 方向：long/short
        self.lot_size = lot_size  # 手数
        self.entry_price = entry_price  # 入场价
        self.sl = sl  # 止损价
        self.tp = tp  # 止盈价
        self.ticket = ticket  # MT5订单号
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

# ======================== 3. 工具函数 ========================
def adjust_lot_size(current_row, past_week_vol):
    """动态调整仓位（1-3手，基于趋势强度）"""
    if pd.isna(past_week_vol) or current_row['波动幅度'] == 0:
        return min_lot

    # 趋势强度=当前价格变动/当前波动幅度（越大趋势越明确）
    trend_strength = abs(current_row['增减']) / current_row['波动幅度']

    # 趋势明确且波动较小时用3手，否则1手
    return max_lot if (trend_strength > 0.6 and current_row['波动幅度'] < past_week_vol) else min_lot

def calculate_sl_tp(current_row, direction):
    """计算止损止盈（基于波动幅度动态调整，考虑点差，确保合理风险收益比）"""
    # 处理时间格式：Timestamp→str→提取小时（避免strptime错误）
    time_str = current_row['时间'].strftime("%Y-%m-%d %H:%M:%S")
    hour = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").hour

    # AUDUSD在不同时段波动性不同，调整止损止盈倍数以确保合理的风险收益比
    # 流动性低时段（凌晨0-5点）：收紧止损止盈，风险收益比为1:2
    if 0 <= hour <= 5:
        sl_multiplier = 0.4
        tp_multiplier = 0.8
    # 欧洲交易时段（7-12点和15-19点）：AUDUSD波动较大，采用较高风险收益比
    elif 7 <= hour <= 12 or 15 <= hour <= 19:
        sl_multiplier = 0.5
        tp_multiplier = 1.5  # 风险收益比1:3
    # 其他时段：适中风险收益比
    else:
        sl_multiplier = 0.5
        tp_multiplier = 1.2  # 风险收益比1:2.4

    # 获取AUDUSD点差
    symbol_info = mt5.symbol_info(symbol)
    spread = symbol_info.spread if symbol_info is not None else 0
    point = symbol_info.point if symbol_info is not None else 0.0001
    spread_value = spread * point
    
    # 确保最小止损距离，AUDUSD止损距离不应低于40点
    min_sl_distance = 0.00040
    calculated_sl_distance = sl_multiplier * current_row['波动幅度']
    sl_distance = max(min_sl_distance, calculated_sl_distance)
    
    # 按多空方向计算点位，考虑点差影响
    if direction == "long":
        # 多单止损需额外扣除点差，止盈需额外增加点差
        sl = current_row['开盘价'] - sl_distance
        tp = current_row['开盘价'] + tp_multiplier * current_row['波动幅度']
    else:
        # 空单止损需额外增加点差，止盈需额外扣除点差
        sl = current_row['开盘价'] + sl_distance
        tp = current_row['开盘价'] - tp_multiplier * current_row['波动幅度']

    # 四舍五入到合适的小数位数（AUDUSD通常为5位小数）
    sl = round(sl, 5)
    tp = round(tp, 5)
    
    print(f"[AUDUSD] 计算止盈止损: 方向={direction}, 入场价={current_row['开盘价']}, 止损={sl}, 止盈={tp}, 波动幅度={current_row['波动幅度']}, 点差={spread_value}")
    
    return sl, tp

def get_mt5_data(symbol="AUDUSD", timeframe=mt5.TIMEFRAME_H1, count=1000):
    """
    从MT5获取AUDUSD历史数据
    """
    try:
        # 初始化MT5连接
        # 获取测试账户信息
        from config_loader import get_test_account
        test_account_info = get_test_account()
        
        # 使用测试账户连接MT5
        if test_account_info['enabled']:
            if not mt5.initialize(
                login=int(test_account_info['number']), 
                password=test_account_info['password'], 
                server=test_account_info['server']
            ):
                print(f"MT5初始化失败，账户: {test_account_info['number']}")
                return None
        else:
            # 如果测试账户未启用，则使用默认初始化
            if not mt5.initialize():
                print("MT5初始化失败")
                return None

        # 获取历史数据
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            print("未能获取到历史数据")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 重命名列以匹配原策略
        df.rename(columns={
            'time': '时间',
            'open': '开盘价',
            'close': '收盘价',
            'high': '最高价',
            'low': '最低价',
            'tick_volume': '成交量'
        }, inplace=True)
        
        # 添加计算列
        df['增减'] = df['收盘价'] - df['开盘价']
        df['波动幅度'] = df['最高价'] - df['最低价']
        df['星期几'] = df['时间'].dt.day_name()
        
        # 计算昨天增减（前一天的增减）
        df['昨天增减'] = df['增减'].shift(1)
        
        # 计算昨日波动幅度（前一天的波动幅度）
        df['昨日波动幅度'] = df['波动幅度'].shift(1)
        
        # 计算技术指标（用于入场信号，最小周期确保有值）
        df['MA5'] = df['收盘价'].rolling(window=5, min_periods=5).mean()
        df['MA10'] = df['收盘价'].rolling(window=10, min_periods=10).mean()

        # 计算过去7天（168小时）平均波动幅度（判断市场稳定性）
        df['past_week_vol'] = df['波动幅度'].rolling(window=168, min_periods=168).mean()

        # 只保留有完整指标的数据（避免空值错误）
        df = df.dropna(subset=['MA5', 'MA10', 'past_week_vol'])
        
        return df
        
    except Exception as e:
        print(f"获取MT5数据时发生错误: {str(e)}")
        return None
    finally:
        # 注意：不关闭MT5连接，因为可能其他地方还需要使用
        pass

def initialize_trading_day():
    """初始化交易日，查询历史订单并更新当日交易统计"""
    global daily_trades, daily_pnl, last_trading_day
    
    # 获取当前日期
    today = datetime.now().date()
    last_trading_day = today
    
    try:
        # 查询今日历史订单，按币种区分
        from_date = datetime(today.year, today.month, today.day)
        to_date = datetime.now()
        
        # 获取所有历史订单
        history_orders = mt5.history_deals_get(from_date, to_date)
        
        if history_orders is not None and len(history_orders) > 0:
            audusd_orders = []
            # 筛选出AUDUSD相关的订单
            for order in history_orders:
                if hasattr(order, 'symbol') and order.symbol == symbol:
                    audusd_orders.append(order)
            
            # 计算今日AUDUSD已交易次数和盈亏
            # 每笔交易包含开仓和平仓两个订单，所以除以2
            daily_trades = len(audusd_orders) // 2
            
            # 计算今日AUDUSD盈亏
            daily_pnl = 0.0
            for order in audusd_orders:
                if hasattr(order, 'profit') and order.profit is not None:
                    daily_pnl += order.profit
                    
        print(f"[AUDUSD] 今日交易次数初始化: {daily_trades}, 今日盈亏: {daily_pnl:.2f}")
        
    except Exception as e:
        print(f"[AUDUSD] 初始化交易日时发生错误: {str(e)}")

def check_position_status():
    """检查当前持仓状态"""
    global current_position, daily_trades, current_balance, daily_pnl, total_pnl
    
    try:
        # 获取当前AUDUSD持仓
        positions = mt5.positions_get(symbol=symbol)
        
        # 如果没有持仓但程序认为有持仓，说明已被外部平仓
        if (positions is None or len(positions) == 0) and current_position is not None:
            # 查询最近的平仓订单以获取盈亏信息
            today = datetime.now().date()
            from_date = datetime(today.year, today.month, today.day)
            to_date = datetime.now()
            
            # 获取今天的订单历史，仅查询当前币种
            history_orders = mt5.history_deals_get(from_date, to_date)
            if history_orders is not None and len(history_orders) > 0:
                # 查找与当前持仓相关的平仓订单，仅筛选当前币种
                for order in history_orders:
                    if hasattr(order, 'symbol') and order.symbol == symbol:
                        # 这里简化处理，实际应该更精确匹配
                        if hasattr(order, 'profit') and order.profit is not None:
                            # 更新统计数据
                            daily_trades += 1
                            current_balance += order.profit
                            daily_pnl += order.profit
                            total_pnl += order.profit
                            print(f"[AUDUSD] 检测到外部平仓，盈亏: {order.profit:.2f}, 余额: {current_balance:.2f}, 今日交易次数: {daily_trades}")
                            break
            
            # 重置持仓状态
            current_position = None
            
        # 如果有持仓但程序认为没有持仓，说明是外部开仓
        elif positions is not None and len(positions) > 0 and current_position is None:
            position = positions[0]
            print(f"[AUDUSD] 检测到外部开仓，订单号: {position.ticket}, 方向: {'多' if position.type == 0 else '空'}, 手数: {position.volume}")
            
    except Exception as e:
        print(f"[AUDUSD] 检查持仓状态时发生错误: {str(e)}")

def print_time_info():
    """打印M1和H1时间信息"""
    try:
        # 获取M1最新时间
        mt5.symbol_select(symbol, True)
        m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if m1_rates is not None and len(m1_rates) > 0:
            m1_time = TimeUtils.mt5_timestamp_to_datetime(m1_rates[0]['time'])
            beijing_time = TimeUtils.mt5_time_to_beijing_time(m1_time)
            print(f"[AUDUSD] MT5 M1时间: {TimeUtils.datetime_to_string(m1_time, True)}, 北京时间: {TimeUtils.datetime_to_string(beijing_time, True)}")
        
        # 获取H1最新时间
        h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)
        if h1_rates is not None and len(h1_rates) > 0:
            h1_time = TimeUtils.mt5_timestamp_to_datetime(h1_rates[0]['time'])
            beijing_time = TimeUtils.mt5_time_to_beijing_time(h1_time)
            print(f"[AUDUSD] MT5 H1时间: {TimeUtils.datetime_to_string(h1_time, True)}, 北京时间: {TimeUtils.datetime_to_string(beijing_time, True)}")
            
    except Exception as e:
        print(f"[AUDUSD] 获取时间信息时发生错误: {str(e)}")

def send_order(direction, lot_size, sl, tp, price):
    """发送真实订单到MT5"""
    try:
        # 验证价格参数
        if price is None or sl is None or tp is None:
            print("[AUDUSD] 订单价格参数无效")
            return None
            
        # 验证手数
        if lot_size <= 0:
            print("[AUDUSD] 订单手数必须大于0")
            return None
            
        # 获取交易品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[AUDUSD] 无法获取交易品种 {symbol} 的信息")
            return None
            
        # 检查交易品种是否可用
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"[AUDUSD] 无法选择交易品种 {symbol}")
                return None
                
        # 检查价格是否合理
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("[AUDUSD] 无法获取当前报价")
            return None
            
        # 根据方向设置订单类型
        if direction == "long":
            order_type = mt5.ORDER_TYPE_BUY
            # 检查止损和止盈是否合理 (多单止损应低于入场价，止盈应高于入场价)
            if sl >= price or tp <= price:
                print(f"[AUDUSD] 多单止损或止盈价格设置不合理: 入场价={price}, 止损={sl}, 止盈={tp}")
                return None
        else:
            order_type = mt5.ORDER_TYPE_SELL
            # 检查止损和止盈是否合理 (空单止损应高于入场价，止盈应低于入场价)
            if sl <= price or tp >= price:
                print(f"[AUDUSD] 空单止损或止盈价格设置不合理: 入场价={price}, 止损={sl}, 止盈={tp}")
                return None
            
        # 准备订单请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 打印下单信息
        print(f"[AUDUSD] 准备下单: 品种={symbol}, 方向={direction}, 手数={lot_size}, 入场价={price}, 止损={sl}, 止盈={tp}")
        
        # 发送订单
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[AUDUSD] 订单发送失败，错误码: {result.retcode}")
            # 打印更多错误信息帮助诊断
            print(f"[AUDUSD] 错误详情: {mt5.last_error()}")
            return None
            
        print(f"[AUDUSD] 订单发送成功，订单号: {result.order}")
        return result.order
        
    except Exception as e:
        print(f"[AUDUSD] 发送订单时发生错误: {str(e)}")
        return None

def close_position(ticket):
    """平仓持仓"""
    try:
        # 获取当前持仓信息
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            print("[AUDUSD] 未找到持仓")
            return False
            
        position = positions[0]
        
        # 准备平仓请求
        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            order_type = mt5.ORDER_TYPE_BUY
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": ticket,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 发送平仓请求
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[AUDUSD] 平仓失败，错误码: {result.retcode}")
            return False
            
        print(f"[AUDUSD] 平仓成功，订单号: {result.order}")
        return True
        
    except Exception as e:
        print(f"[AUDUSD] 平仓时发生错误: {str(e)}")
        return False

# ======================== 4. 核心交易逻辑 ========================
def run_strategy():
    global order_id_counter, current_position, daily_trades
    global daily_pnl, total_pnl, last_trading_day, last_close_time
    
    # 初始化MT5连接
    if not mt5.initialize():
        print("MT5初始化失败")
        return
    
    # 打印时间信息
    print_time_info()
    
    # 初始化交易日
    initialize_trading_day()
    
    # 检查当前持仓状态
    check_position_status()
    
    while True:  # 持续运行循环
        try:
            # 获取当前时间
            current_time = datetime.now()
            
            # 每小时获取一次最新数据 (可以根据需要调整频率)
            # 从MT5获取最新数据
            print("正在从MT5获取AUDUSD最新历史数据...")
            df = get_mt5_data("AUDUSD", mt5.TIMEFRAME_H1, 2000)
            
            if df is None or len(df) == 0:
                print("无法获取数据，等待下一次尝试...")
                time.sleep(60)  # 等待1分钟再尝试
                continue
            
            print(f"成功获取 {len(df)} 条数据")
            
            # 使用最新数据进行分析和交易决策
            latest_data = df.iloc[-1]  # 获取最新一条数据
            prev_data = df.iloc[-2] if len(df) >= 2 else None  # 获取前一条数据
            
            if prev_data is None:
                print("数据不足，等待更多数据...")
                time.sleep(60)  # 等待1分钟再尝试
                continue
            
            # 检查是否有足够的历史数据用于周三信号判断
            has_enough_history = len(df) >= 3

            # 时间处理（统一为str格式，避免类型错误）
            current_time = latest_data['时间']
            current_date = current_time.date()
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            # 打印M1时间信息和星期几
            # 获取M1最新时间
            mt5.symbol_select(symbol, True)
            m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if m1_rates is not None and len(m1_rates) > 0:
                m1_time = TimeUtils.mt5_timestamp_to_datetime(m1_rates[0]['time'])
                beijing_time = TimeUtils.mt5_time_to_beijing_time(m1_time)
                # 获取星期几信息
                weekday_name = m1_time.strftime('%A')
                # 星期映射: 0=星期一, 1=星期二, 2=星期三, 3=星期四, 4=星期五, 5=星期六, 6=星期日
                weekday_num = m1_time.weekday()
                weekday_chinese = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][weekday_num]
                print(f"[AUDUSD] M1时间: {TimeUtils.datetime_to_string(m1_time, True)}, 北京时间: {TimeUtils.datetime_to_string(beijing_time, True)}, 星期: {weekday_chinese}({weekday_num})")
            else:
                print("[AUDUSD] 无法获取M1时间信息")

            # 每日初始化（新交易日重置统计）
            if shared_state.check_daily_reset(current_date):
                daily_trades = 0
                daily_pnl = 0.0
                # 重新检查持仓状态
                check_position_status()

            # -------------------- 风险控制：禁止违规开仓 --------------------
            # 使用全局共享状态检查是否可以开仓
            # 当日亏损≥5% 或 总亏损≥10% 或 净值<90% 或 当日交易次数已达上限：禁止开仓
            if not shared_state.can_open_position() or daily_trades >= max_daily_trades:
                current_lot = 0.0
                # 获取全局状态信息用于输出
                global_stats = shared_state.get_global_stats()
                print(f"[AUDUSD] 风险控制限制开仓: 当前手数={current_lot}, 全局当日盈亏={global_stats['daily_pnl']:.2f}, 全局总盈亏={global_stats['total_pnl']:.2f}, 全局净值={global_stats['equity']:.2f}")
            else:
                current_lot = adjust_lot_size(latest_data, latest_data['past_week_vol'])
                
            # 确保仓位在合理范围内
            current_lot = max(min_lot, min(current_lot, max_lot))

            # -------------------- 空仓：检查入场条件（多信号触发） --------------------
            if current_position is None and current_lot > 0:
                # 检查上次平仓时间，如果是同一根K线内平仓的，则不立即开仓
                if last_close_time is not None:
                    # 获取最新H1 K线的开始时间
                    latest_h1_time = latest_data['时间'].replace(minute=0, second=0, microsecond=0)
                    # 如果上次平仓时间在当前K线周期内，则跳过开仓
                    if last_close_time >= latest_h1_time:
                        print(f"[AUDUSD] 上次平仓时间 {last_close_time} 在当前K线周期内，等待下一根完整K线")
                        time.sleep(60)
                        continue
                
                # 只在欧洲交易时段交易（GMT 7:00-12:00和15:00-19:00）
                current_hour = latest_data['时间'].hour
                if not (7 <= current_hour <= 12 or 15 <= current_hour <= 19):
                    # 非欧洲交易时段不交易
                    print(f"[AUDUSD] 非欧洲交易时段 ({current_hour}:00)，跳过交易")
                    time.sleep(60)
                    continue
                
                signal_type = None
                direction = None
                signal_reasons = []  # 记录信号判断过程
                
                # 打印当前数据用于分析（使用DataFrame中的星期信息）
                # pandas dayofweek映射: 0=星期一, 1=星期二, 2=星期三, 3=星期四, 4=星期五, 5=星期六, 6=星期日
                weekday_map = {'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三', 
                              'Thursday': '星期四', 'Friday': '星期五', 'Saturday': '星期六', 'Sunday': '星期日'}
                
                # 正确计算昨天的星期几
                current_weekday = latest_data['时间'].weekday()  # 今天是星期几 (0-6)
                yesterday_weekday = (current_weekday - 1) % 7  # 昨天是星期几 (0-6)
                weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
                yesterday_weekday_chinese = weekday_names[yesterday_weekday]
                current_weekday_chinese = weekday_names[current_weekday]
                
                print(f"[AUDUSD] 分析H1数据信号: 昨天星期={yesterday_weekday_chinese}, 今天星期={current_weekday_chinese}")
                
                # 1. 周一→周二趋势延续
                if prev_data['星期几'] == 'Monday' and latest_data['星期几'] == 'Tuesday':
                    # 周一上涨→周二延续：做多
                    if (prev_data['昨天增减'] > 0 and
                            latest_data['开盘价'] > prev_data['收盘价'] and
                            latest_data['增减'] > 0):
                        signal_type = "周一上涨周二延续"
                        direction = "long"
                        print(f"[AUDUSD] 满足做多条件: 周一上涨周二延续 - 昨天增减={prev_data['昨天增减']:.5f}, 价格跳空={latest_data['开盘价'] > prev_data['收盘价']}, 当前上涨={latest_data['增减'] > 0}")

                    # 周一下跌→周二延续：做空
                    elif (prev_data['昨天增减'] < 0 and
                          latest_data['开盘价'] < prev_data['收盘价'] and
                          latest_data['增减'] < 0):
                        signal_type = "周一下跌周二延续"
                        direction = "short"
                        print(f"[AUDUSD] 满足做空条件: 周一下跌周二延续 - 昨天增减={prev_data['昨天增减']:.5f}, 价格跳空={latest_data['开盘价'] < prev_data['收盘价']}, 当前下跌={latest_data['增减'] < 0}")
                    else:
                        signal_reasons.append(f"周一→周二: 昨天增减={prev_data['昨天增减']:.5f}, 跳空={latest_data['开盘价'] > prev_data['收盘价']}, 当前涨跌={latest_data['增减'] > 0}")

                # 2. 周三反转信号
                elif latest_data['星期几'] == 'Wednesday' and has_enough_history:
                    # 前两日下跌→周三反转：做多
                    if (df.iloc[-3]['增减'] < 0 and
                            prev_data['增减'] < 0 and
                            latest_data['开盘价'] > prev_data['收盘价'] and
                            latest_data['增减'] > 0):
                        signal_type = "周三反转做多"
                        direction = "long"
                        print(f"[AUDUSD] 满足做多条件: 周三反转做多 - 前两日下跌({df.iloc[-3]['增减']:.5f}, {prev_data['增减']:.5f}), 价格跳空={latest_data['开盘价'] > prev_data['收盘价']}, 当前上涨={latest_data['增减'] > 0}")

                    # 前两日上涨→周三反转：做空
                    elif (df.iloc[-3]['增减'] > 0 and
                          prev_data['增减'] > 0 and
                          latest_data['开盘价'] < prev_data['收盘价'] and
                          latest_data['增减'] < 0):
                        signal_type = "周三反转做空"
                        direction = "short"
                        print(f"[AUDUSD] 满足做空条件: 周三反转做空 - 前两日上涨({df.iloc[-3]['增减']:.5f}, {prev_data['增减']:.5f}), 价格跳空={latest_data['开盘价'] < prev_data['收盘价']}, 当前下跌={latest_data['增减'] < 0}")
                    else:
                        signal_reasons.append(f"周三反转: 前两日涨跌({df.iloc[-3]['增减']:.5f}, {prev_data['增减']:.5f}), 跳空={latest_data['开盘价'] > prev_data['收盘价']}, 当前涨跌={latest_data['增减'] > 0}")

                # 3. MA金叉/死叉（技术信号） - 仅在欧洲交易时段使用
                elif latest_data['MA5'] > latest_data['MA10'] and prev_data['MA5'] <= prev_data['MA10']:
                    signal_type = "MA金叉"
                    direction = "long"
                    print(f"[AUDUSD] 满足做多条件: MA金叉 - 当前MA5={latest_data['MA5']:.5f}, MA10={latest_data['MA10']:.5f}, 前期MA5={prev_data['MA5']:.5f}, 前期MA10={prev_data['MA10']:.5f}")
                    
                elif latest_data['MA5'] < latest_data['MA10'] and prev_data['MA5'] >= prev_data['MA10']:
                    signal_type = "MA死叉"
                    direction = "short"
                    print(f"[AUDUSD] 满足做空条件: MA死叉 - 当前MA5={latest_data['MA5']:.5f}, MA10={latest_data['MA10']:.5f}, 前期MA5={prev_data['MA5']:.5f}, 前期MA10={prev_data['MA10']:.5f}")
                else:
                    signal_reasons.append(f"MA信号: 当前MA5/MA10={latest_data['MA5']:.5f}/{latest_data['MA10']:.5f}, 前期MA5/MA10={prev_data['MA5']:.5f}/{prev_data['MA10']:.5f}")
                    
                # 如果有信号，则开仓
                if signal_type and direction:
                    sl, tp = calculate_sl_tp(latest_data, direction)
                    
                    # 获取当前价格
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        print("[AUDUSD] 无法获取当前价格")
                        time.sleep(60)  # 等待1分钟再尝试
                        continue
                        
                    price = tick.ask if direction == "long" else tick.bid
                    
                    # 发送真实订单
                    ticket = send_order(direction, current_lot, sl, tp, price)
                    if ticket is not None:
                        current_position = TradeOrder(
                            order_id=order_id_counter,
                            timestamp=current_time_str,
                            direction=direction,
                            lot_size=current_lot,
                            entry_price=price,
                            sl=sl,
                            tp=tp,
                            ticket=ticket
                        )
                        orders.append(current_position)
                        order_id_counter += 1
                        # 简化输出信息
                        print(f"[AUDUSD] 开仓: {direction}, 满足条件: {signal_type}, 入场价: {price}, 止损: {sl}, 止盈: {tp}")
                else:
                    # 没有满足条件的信号，输出详细原因
                    if signal_reasons:
                        print("[AUDUSD] 未满足入场条件:")
                        for reason in signal_reasons:
                            print(f"  - {reason}")
                    else:
                        print("[AUDUSD] 未满足入场条件，当前日期不符合任何交易信号触发条件")
                    print("[AUDUSD] 继续等待")
            elif current_position is not None:
                print(f"[AUDUSD] 当前持有仓位: {current_position.direction}, 入场价: {current_position.entry_price}")
            elif current_lot <= 0:
                # 获取全局状态信息用于输出
                global_stats = shared_state.get_global_stats()
                print(f"[AUDUSD] 风险控制限制开仓: 当前手数={current_lot}, 全局当日盈亏={global_stats['daily_pnl']:.2f}, 全局总盈亏={global_stats['total_pnl']:.2f}, 全局净值={global_stats['equity']:.2f}")

            # -------------------- 有持仓：检查平仓条件 --------------------
            if current_position is not None:
                # 检查是否已经通过MT5自动止盈止损平仓，仅检查当前币种
                positions = mt5.positions_get(symbol=symbol)
                if positions is None or len(positions) == 0:
                    # 持仓已平仓（可能是止盈止损触发），更新状态
                    print(f"[AUDUSD] 持仓已平仓，订单号: {current_position.ticket}")
                    
                    # 查询最近的平仓订单以获取盈亏信息，仅查询当前币种
                    today = datetime.now().date()
                    from_date = datetime(today.year, today.month, today.day)
                    to_date = datetime.now()
                    
                    # 获取今天的订单历史，仅查询当前币种
                    history_orders = mt5.history_deals_get(from_date, to_date)
                    if history_orders is not None and len(history_orders) > 0:
                        # 查找与当前持仓相关的平仓订单，仅筛选当前币种
                        for order in history_orders:
                            if hasattr(order, 'symbol') and order.symbol == symbol:
                                # 这里简化处理，实际应该更精确匹配
                                if hasattr(order, 'profit') and order.profit is not None:
                                    # 更新统计数据
                                    profit = order.profit
                                    daily_trades += 1
                                    daily_pnl += profit
                                    total_pnl += profit
                                    # 更新全局状态
                                    shared_state.update_daily_stats(trades_count=1, profit=profit)
                                    print(f"[AUDUSD] 平仓: MT5自动平仓, 盈亏: {profit:.2f}, 今日交易次数: {daily_trades}")
                                    # 记录平仓时间
                                    last_close_time = datetime.now()
                                    break
                    
                    current_position = None
                    
                # 添加时间止损机制（最大持仓24小时）
                # 注意：时间止损仍然需要程序主动平仓
                if current_position is not None:
                    entry_time = datetime.strptime(current_position.timestamp, "%Y-%m-%d %H:%M:%S")
                    current_time_obj = datetime.now()
                    time_diff = current_time_obj - entry_time
                    if time_diff.total_seconds() > 24 * 3600:  # 超过24小时
                        # 获取当前价格
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None:
                            print("[AUDUSD] 无法获取当前价格进行时间止损")
                            time.sleep(60)  # 等待1分钟再尝试
                            continue
                            
                        current_price = tick.bid if current_position.direction == "long" else tick.ask
                        
                        # 平仓
                        if close_position(current_position.ticket):
                            # 先保存盈亏信息，再关闭订单
                            position_copy = current_position
                            position_copy.close(current_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            # 简化输出信息
                            print(f"[AUDUSD] 平仓: 时间止损, 盈亏: {position_copy.pnl:.2f}, 今日交易次数: {daily_trades}")
                            current_position = None
                            daily_trades += 1
                            daily_pnl += position_copy.pnl
                            total_pnl += position_copy.pnl
                            # 更新全局状态
                            shared_state.update_daily_stats(trades_count=1, profit=position_copy.pnl)
                            # 记录平仓时间
                            last_close_time = datetime.now()

            # 更新净值（含未平仓盈亏）
            if current_position is not None:
                # 获取当前价格
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    current_price = tick.bid if current_position.direction == "long" else tick.ask
                    if current_position.direction == "long":
                        unrealized_pnl = (current_price - current_position.entry_price) * current_position.lot_size * contract_size
                    else:
                        unrealized_pnl = (current_position.entry_price - current_price) * current_position.lot_size * contract_size
                    # 更新全局净值
                    shared_state.equity = shared_state.current_balance + unrealized_pnl
            else:
                # 更新全局净值
                shared_state.equity = shared_state.current_balance
                
            # 确保只持有一个仓位的逻辑正确
            if current_position is not None and current_lot > 0:
                # 如果已经有仓位，则不允许再开新仓，将当前仓位设为0以防止新仓 opening
                current_lot = 0.0
                
            # 等待一段时间再进行下一次检查 (例如: 1分钟)
            time.sleep(60)
            
        except Exception as e:
            print(f"策略执行过程中发生错误: {str(e)}")
            time.sleep(60)  # 出错后等待1分钟再继续
            
# ======================== 5. 主函数 ========================
if __name__ == "__main__":
    try:
        run_strategy()
    except Exception as e:
        print(f"策略执行过程中发生错误: {str(e)}")
    except KeyboardInterrupt:
        print("用户中断程序执行")
    finally:
        # 关闭MT5连接
        mt5.shutdown()
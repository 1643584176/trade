import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import MetaTrader5 as mt5
import logging
import os

# 配置日志
logging.basicConfig(
    filename='../trading_log.log',
    level=logging.INFO,
    format='%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 交易参数 - 修改手数为1
SYMBOL = os.getenv("TRADE_SYMBOL", "XAUUSD")
LOT_SIZE = float(os.getenv("LOT_SIZE", 1.0))  # 手数从0.1改为1


# 风控参数
MAX_CONSECUTIVE_LOSSES = 2
DAILY_MAX_LOSS = 1200
TOTAL_MAX_LOSS = 1800
MAX_DAILY_TRADES = 3
TIME_STOP_LOSS = 10  # H1周期下的10小时
ATR_PERIOD = 14
RSI_PERIOD = 14

# 全局状态变量
current_position = 0  # 0=无仓, 1=多仓, -1=空仓
entry_price = 0.0
entry_time = None
daily_trades = 0
daily_loss = 0.0
consecutive_losses = 0
total_loss = 0.0
historical_data = pd.DataFrame()
# 添加反向交易所需变量
trade_sequence = 0  # 交易序列计数器
last_trade_signal = 0  # 上一次交易信号
last_close_reason = ""  # 上一次平仓原因

# 全局交易状态变量
current_position_ticket = None

class FTMORealTimeTrader:
    def __init__(self):
        self.connect_mt5()
        self.load_initial_data()
        self.reset_daily_stats()
        # 获取点值和合约大小（用于计算盈亏）
        self.point = mt5.symbol_info(SYMBOL).point
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is not None and hasattr(symbol_info, 'trade_contract_size'):
            self.contract_size = symbol_info.trade_contract_size
        else:
            print(f"[{self.get_current_time()}] 无法获取 {SYMBOL} 的合约大小，使用默认值 100")
            logging.warning(f"无法获取 {SYMBOL} 的合约大小，使用默认值 100")
            self.contract_size = 100  # 黄金通常为100盎司/手

        # 初始化持仓ticket
        self.current_position_ticket = None
        
        # 初始化上一次平仓原因
        global last_close_reason
        last_close_reason = ""

        # 初始化历史交易状态
        self.initialize_trade_state_from_history()

        # 打印初始化信息
        self.log_and_print(f"初始化完成 - 交易品种: {SYMBOL}, 手数: {LOT_SIZE}, 合约大小: {self.contract_size}")

    def get_current_time(self):
        """获取MT5服务器时间"""
        try:
            # 获取当前报价，其中包含服务器时间
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is not None:
                server_time = datetime.fromtimestamp(tick.time).strftime('%Y-%m-%d %H:%M:%S')
                return server_time
            else:
                return "获取服务器时间失败"
        except Exception as e:
            return f"获取服务器时间异常: {str(e)}"

    def log_and_print(self, message):
        """同时打印到控制台和日志文件"""
        # 使用MT5服务器时间
        server_time = self.get_current_time()
        print(f"{server_time} {message}")
        logging.info(f"{server_time} {message}")

    def connect_mt5(self):
        """连接MT5平台"""
        if not mt5.initialize():
            error_msg = "MT5初始化失败"
            print(f"[{self.get_current_time()}] {error_msg}")
            logging.error(error_msg)
            raise Exception("MT5初始化失败")

        # 检查连接状态
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            error_msg = "MT5连接失败或未连接"
            print(f"[{self.get_current_time()}] {error_msg}")
            logging.error(error_msg)
            raise Exception("MT5连接失败或未连接")

        # 检查交易品种
        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            error_msg = f"交易品种 {SYMBOL} 不存在"
            print(f"[{self.get_current_time()}] {error_msg}")
            logging.error(error_msg)
            raise Exception(f"交易品种 {SYMBOL} 不存在")

        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                error_msg = f"无法选择交易品种 {SYMBOL}"
                print(f"[{self.get_current_time()}] {error_msg}")
                logging.error(error_msg)
                raise Exception(f"无法选择交易品种 {SYMBOL}")

        print(f"[{self.get_current_time()}] MT5连接成功！")
        print(f"[{self.get_current_time()}] 交易品种{SYMBOL}已就绪")
        logging.info("MT5连接成功！")
        logging.info(f"交易品种{SYMBOL}已就绪")

    def load_initial_data(self):
        """加载初始历史数据"""
        global historical_data
        try:
            # 获取最近30根H1 K线数据
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 30)
            if rates is None or len(rates) == 0:
                error_msg = "无法获取历史数据"
                print(f"[{self.get_current_time()}] {error_msg}")
                logging.error(error_msg)
                raise Exception("无法获取历史数据")

            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # 重命名列以匹配现有代码
            df.rename(columns={
                'open': '当前开盘价',
                'high': '当前最高价',
                'low': '当前最低价',
                'close': '当前收盘价',
                'tick_volume': '成交量'
            }, inplace=True)
            
            # 添加计算指标所需的基础列
            df["时间点"] = df.index
            df["星期数"] = df.index.dayofweek
            df["小时"] = df.index.hour
            
            # 计算相对开盘价变化
            df["相对开盘价变化"] = df["当前收盘价"] - df["当前开盘价"]
            
            # 计算动量指标
            df["1周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(1)
            df["3周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(3)
            
            # 计算K线形态所需数据
            df["实体"] = abs(df["当前收盘价"] - df["当前开盘价"])
            df["上影线"] = df["当前最高价"] - np.maximum(df["当前开盘价"], df["当前收盘价"])
            df["下影线"] = np.minimum(df["当前开盘价"], df["当前收盘价"]) - df["当前最低价"]
            df["是否阳线"] = df["当前收盘价"] > df["当前开盘价"]
            
            # 计算ATR指标
            df["真实波幅"] = np.maximum(
                df["当前最高价"] - df["当前最低价"],
                np.maximum(
                    abs(df["当前最高价"] - df["当前收盘价"].shift(1)),
                    abs(df["当前最低价"] - df["当前收盘价"].shift(1))
                )
            )
            df["ATR"] = df["真实波幅"].rolling(ATR_PERIOD).mean()
            
            # 计算周间趋势
            df["周间趋势"] = 0
            for i in range(len(df)):
                if i >= 4:  # 需要前面4根K线来计算周间趋势
                    week_start_idx = max(0, i - 4)
                    week_data = df.iloc[week_start_idx:i+1]
                    week_open = week_data["当前开盘价"].iloc[0]
                    week_close = week_data["当前收盘价"].iloc[-1]
                    df.loc[df.index[i], "周间趋势"] = 1 if week_close > week_open else -1
            
            historical_data = df
            
            # 打印数据加载信息
            latest_time = df.index[-1]
            # 确保latest_time是datetime类型
            if isinstance(latest_time, pd.Timestamp):
                print(f"[{self.get_current_time()}] 初始数据加载完成，共{len(df)}根H1 K线，最新K线服务器时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logging.info(f"初始数据加载完成，共{len(df)}根H1 K线，最新K线服务器时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"[{self.get_current_time()}] 初始数据加载完成，共{len(df)}根H1 K线，最新K线服务器时间: {latest_time}")
                logging.info(f"初始数据加载完成，共{len(df)}根H1 K线，最新K线服务器时间: {latest_time}")
            
        except Exception as e:
            error_msg = f"加载初始数据时发生错误: {str(e)}"
            print(f"[{self.get_current_time()}] {error_msg}")
            logging.error(error_msg)
            raise

    def reset_daily_stats(self):
        """重置每日统计"""
        global daily_trades, daily_loss
        daily_trades = 0
        daily_loss = 0.0
        print(f"[{self.get_current_time()}] 当日交易统计重置：今日交易次数={daily_trades}，当日亏损={daily_loss:.2f}美元")
        logging.info(f"当日交易统计重置：今日交易次数={daily_trades}，当日亏损={daily_loss:.2f}美元")

    def update_real_time_data(self):
        """更新实时数据"""
        global historical_data
        try:
            # 获取最新的2根K线数据（确保当前K线已完成）
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 2)
            if rates is None or len(rates) < 2:
                return

            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # 重命名列
            df.rename(columns={
                'open': '当前开盘价',
                'high': '当前最高价',
                'low': '当前最低价',
                'close': '当前收盘价',
                'tick_volume': '成交量'
            }, inplace=True)
            
            # 只保留最新的一根已完成K线
            latest_complete_bar = df.iloc[0:1].copy()  # 添加.copy()避免SettingWithCopyWarning
            
            # 使用.loc方法安全地添加计算指标所需的基础列
            idx = latest_complete_bar.index[0]
            latest_complete_bar.loc[idx, "时间点"] = idx
            latest_complete_bar.loc[idx, "星期数"] = idx.dayofweek
            latest_complete_bar.loc[idx, "小时"] = idx.hour
            
            # 计算相对开盘价变化
            latest_complete_bar.loc[idx, "相对开盘价变化"] = latest_complete_bar.loc[idx, "当前收盘价"] - latest_complete_bar.loc[idx, "当前开盘价"]
            
            # 计算动量指标
            latest_close = historical_data["当前收盘价"].iloc[-1]
            latest_complete_bar.loc[idx, "1周期动量"] = latest_complete_bar.loc[idx, "当前收盘价"] - latest_close
            if len(historical_data) >= 3:
                close_3_periods_ago = historical_data["当前收盘价"].iloc[-3]
                latest_complete_bar.loc[idx, "3周期动量"] = latest_complete_bar.loc[idx, "当前收盘价"] - close_3_periods_ago
            else:
                latest_complete_bar.loc[idx, "3周期动量"] = 0
            
            # 计算K线形态所需数据
            open_price = latest_complete_bar.loc[idx, "当前开盘价"]
            close_price = latest_complete_bar.loc[idx, "当前收盘价"]
            high_price = latest_complete_bar.loc[idx, "当前最高价"]
            low_price = latest_complete_bar.loc[idx, "当前最低价"]
            
            latest_complete_bar.loc[idx, "实体"] = abs(close_price - open_price)
            latest_complete_bar.loc[idx, "上影线"] = high_price - np.maximum(open_price, close_price)
            latest_complete_bar.loc[idx, "下影线"] = np.minimum(open_price, close_price) - low_price
            latest_complete_bar.loc[idx, "是否阳线"] = close_price > open_price
            
            # 计算ATR指标
            # 先合并历史数据和新K线
            combined_df = pd.concat([historical_data, latest_complete_bar])
            combined_df["真实波幅"] = np.maximum(
                combined_df["当前最高价"] - combined_df["当前最低价"],
                np.maximum(
                    abs(combined_df["当前最高价"] - combined_df["当前收盘价"].shift(1)),
                    abs(combined_df["当前最低价"] - combined_df["当前收盘价"].shift(1))
                )
            )
            combined_df["ATR"] = combined_df["真实波幅"].rolling(ATR_PERIOD).mean()
            
            # 只保留最新的30根K线
            historical_data = combined_df.tail(30)
            
            # 移除K线时间打印，增加持仓信息透明度
            self.log_and_print("数据更新完成")
            self.log_current_position_info()
            
        except Exception as e:
            error_msg = f"更新实时数据时发生错误: {str(e)}"
            self.log_and_print(error_msg)
            logging.error(error_msg)

    def calculate_indicators(self):
        """计算实时指标"""
        global historical_data
        try:
            df = historical_data.copy()
            
            # 计算周间趋势
            if len(df) >= 5:
                week_data = df.tail(5)
                week_open = week_data["当前开盘价"].iloc[0]
                week_close = week_data["当前收盘价"].iloc[-1]
                df.loc[df.index[-1], "周间趋势"] = 1 if week_close > week_open else -1
            
            historical_data = df
            
            # 移除K线时间打印，增加持仓信息透明度
            self.log_and_print("指标计算完成")
            self.log_current_position_info()
            
            return df.iloc[-1]
            
        except Exception as e:
            error_msg = f"计算指标时发生错误: {str(e)}"
            self.log_and_print(error_msg)
            logging.error(error_msg)
            return None

    def log_current_position_info(self):
        """记录当前持仓信息以增加透明度"""
        global current_position, entry_price, entry_time, daily_trades
        
        # 获取MT5实际持仓信息
        position_info = None
        if hasattr(self, 'current_position_ticket') and self.current_position_ticket:
            position_info = mt5.positions_get(ticket=self.current_position_ticket)
        
        position_status = "无持仓"
        if current_position == 1:
            position_status = "多仓"
        elif current_position == -1:
            position_status = "空仓"
            
        self.log_and_print(f"持仓状态: {position_status}, 今日交易次数: {daily_trades}")
        
        if current_position != 0 and entry_price != 0.0:
            # 获取当前市场价格
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is not None:
                current_price = tick.bid if current_position == 1 else tick.ask
                # 计算浮动盈亏
                profit = (current_price - entry_price) * self.contract_size * LOT_SIZE if current_position == 1 else \
                        (entry_price - current_price) * self.contract_size * LOT_SIZE
                self.log_and_print(f"开仓价格: {entry_price:.2f}, 当前价格: {current_price:.2f}, 浮动盈亏: {profit:.2f}")
        
        # 显示MT5实际持仓信息
        if position_info is not None and len(position_info) > 0:
            pos = position_info[0]
            self.log_and_print(f"MT5持仓信息 - 订单号: {pos.ticket}, 方向: {'BUY' if pos.type == 0 else 'SELL'}, "
                              f"价格: {pos.price_open:.2f}, 手数: {pos.volume}")
        elif hasattr(self, 'current_position_ticket') and self.current_position_ticket:
            self.log_and_print(f"MT5持仓信息: 未找到订单号 {self.current_position_ticket} 的持仓")

    def get_session_type(self, hour):
        if 0 <= hour <= 6:
            return "亚洲"
        elif 7 <= hour <= 14:
            return "欧洲"
        elif 15 <= hour <= 23:
            return "美洲"
        return "其他"

    def is_reversal_pattern(self, latest_data):
        df = historical_data
        idx = len(df) - 1  # 使用索引位置而不是时间戳
        if idx < 2:
            return None

        current = latest_data
        prev1 = df.iloc[idx - 1]
        prev2 = df.iloc[idx - 2]

        # 锤子线/上吊线
        if current["下影线"] > current["实体"] * 2 and current["上影线"] < current["实体"] * 0.5:
            return "bullish" if current["是否阳线"] else "bearish"

        # 吞没形态
        if prev1["是否阳线"] != current["是否阳线"]:
            if not prev1["是否阳线"] and current["是否阳线"]:
                if current["当前开盘价"] < prev1["当前收盘价"] and current["当前收盘价"] > prev1["当前开盘价"]:
                    return "bullish"
            if prev1["是否阳线"] and not current["是否阳线"]:
                if current["当前开盘价"] > prev1["当前收盘价"] and current["当前收盘价"] < prev1["当前开盘价"]:
                    return "bearish"

        # 早晨之星/黄昏之星
        if idx >= 3:
            prev3 = df.iloc[idx - 3]
            if not prev3["是否阳线"] and abs(prev2["实体"]) < prev3["实体"] * 0.3 and current["是否阳线"]:
                if current["当前收盘价"] > prev2["当前最高价"]:
                    return "bullish"
            if prev3["是否阳线"] and abs(prev2["实体"]) < prev3["实体"] * 0.3 and not current["是否阳线"]:
                if current["当前收盘价"] < prev2["当前最低价"]:
                    return "bearish"

        return None

    def get_todays_history_trades(self):
        """获取当天的历史交易数据并按时间排序"""
        try:
            # 获取今天的开始和结束时间
            today = datetime.now().date()
            from_time = datetime(today.year, today.month, today.day)
            to_time = from_time + timedelta(days=1)
            
            # 获取历史交易
            history_deals = mt5.history_deals_get(from_time, to_time)
            
            if history_deals is None or len(history_deals) == 0:
                self.log_and_print("今日无历史交易记录")
                return []
            
            # 过滤出当前交易对的交易，并按时间排序
            symbol_deals = []
            for deal in history_deals:
                if hasattr(deal, 'symbol') and deal.symbol == SYMBOL:
                    symbol_deals.append(deal)
            
            # 按时间排序
            symbol_deals.sort(key=lambda x: x.time)
            
            self.log_and_print(f"今日{SYMBOL}历史交易记录数量: {len(symbol_deals)}")
            return symbol_deals
            
        except Exception as e:
            self.log_and_print(f"获取历史交易数据时出错: {str(e)}")
            return []

    def is_last_trade_profitable(self, history_trades):
        """检查最近一笔交易是否盈利"""
        if not history_trades:
            return False, None
            
        # 获取最近一笔交易
        last_trade = history_trades[-1]
        
        # 检查是否盈利
        if hasattr(last_trade, 'profit'):
            is_profit = last_trade.profit > 0
            close_reason = "止盈" if is_profit else "止损" if last_trade.profit < 0 else "平仓"
            self.log_and_print(f"最近一笔交易盈亏: {last_trade.profit:.2f}, 平仓原因: {close_reason}")
            return is_profit, close_reason
            
        return False, None

    def generate_real_time_signal(self, latest_data):
        global trade_sequence, last_trade_signal, last_close_reason
        df = historical_data
        weekday_num = latest_data["星期数"]
        hour = latest_data["小时"]
        session_type = self.get_session_type(hour)
        open_direction = 1 if latest_data["相对开盘价变化"] > 0 else -1
        momentum_1 = 1 if latest_data["1周期动量"] > 0 else -1
        momentum_3 = 1 if latest_data["3周期动量"] > 0 else -1
        reversal = self.is_reversal_pattern(latest_data)
        atr = latest_data["ATR"] if not np.isnan(latest_data["ATR"]) else 0.0
        avg_atr = df["ATR"].mean() if not df["ATR"].isna().all() else 0.0

        # 显示信号分析时间和相关信息
        latest_time = latest_data["时间点"].strftime('%Y-%m-%d %H:%M:%S')
        self.log_and_print(f"信号分析 - K线时间: {latest_time}, 时段: {session_type}, 星期: {weekday_num}, 小时: {hour}")
        self.log_and_print(f"开仓方向: {open_direction}, 1周期动量: {momentum_1}, 3周期动量: {momentum_3}, ATR: {atr:.4f}")
        # 增加持仓信息透明度
        self.log_current_position_info()

        if session_type not in ["欧洲", "美洲"]:
            return 0, ["非交易时段"]

        if atr < avg_atr * 0.5 or atr > avg_atr * 2:
            return 0, [f"波动率异常 (ATR: {atr:.4f}, 平均ATR: {avg_atr:.4f})"]

        signal = 0
        reasons = []

        # 星期几逻辑
        if weekday_num == 4:  # 周五
            if open_direction == momentum_1 == momentum_3:
                signal = open_direction
                reasons.append(f"周五趋势强化({signal})")
        elif weekday_num == 2:  # 周三
            if reversal == "bullish":
                signal = 1
                reasons.append("周三看涨反转")
            elif reversal == "bearish":
                signal = -1
                reasons.append("周三看跌反转")
        elif weekday_num in [0, 1]:  # 周一、周二
            weekly_trend = latest_data["周间趋势"]
            if not np.isnan(weekly_trend) and weekly_trend != open_direction:
                signal = weekly_trend
                reasons.append(f"周间趋势修正({signal})")
        elif weekday_num == 3:  # 周四
            weekly_trend = latest_data["周间趋势"]
            if not np.isnan(weekly_trend) and weekly_trend == open_direction:
                signal = weekly_trend
                reasons.append(f"周间趋势确认({signal})")

        # 基础信号
        if signal == 0:
            if reversal == "bullish":
                signal = 1
                reasons.append("看涨反转形态")
            elif reversal == "bearish":
                signal = -1
                reasons.append("看跌反转形态")
            elif open_direction == momentum_1 == momentum_3 and abs(latest_data["相对开盘价变化"]) > atr * 0.5:
                signal = open_direction
                reasons.append(f"动量一致({signal})")

        # 根据上一次平仓原因决定交易方向（原有的逻辑）
        if trade_sequence >= 1 and signal != 0:
            if last_close_reason == "止盈":
                # 止盈后反向交易
                signal = -last_trade_signal
                reasons.append(f"止盈后反向交易({signal})")
            else:
                # 止损后顺着原方向交易或默认信号
                signal = last_trade_signal
                reasons.append(f"止损后顺势交易({signal})")

        # 添加根据当日历史交易数据决定是否反向的逻辑（新逻辑）
        history_trades = self.get_todays_history_trades()
        if len(history_trades) > 0:
            is_profitable, close_reason = self.is_last_trade_profitable(history_trades)
            if close_reason is not None:
                if close_reason == "止盈":
                    # 如果最近一笔交易是止盈，则反向
                    signal = -signal if signal != 0 else signal
                    reasons.append(f"基于历史交易止盈反向({signal})")
                # 如果是止损则保持原方向，这里不需要额外处理，因为默认就是原方向

        if len(reasons) < 1:
            signal = 0
            reasons.append("信号不足")

        return signal, reasons

    def send_order(self, action, volume, price, sl=0, tp=0):
        """发送交易订单"""
        # 检查MT5连接状态
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            self.log_and_print("MT5未连接，无法发送订单")
            logging.error("MT5未连接，无法发送订单")
            return False

        # 订单类型：买入=0，卖出=1
        order_type = mt5.ORDER_TYPE_BUY if action == 1 else mt5.ORDER_TYPE_SELL
        order_type_str = "买入" if action == 1 else "卖出"
        
        # 价格精度调整（黄金通常为2位小数）
        def round_price(price):
            return round(price, 2)

        # 准备订单请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": round_price(price),
            "sl": round_price(sl) if sl != 0 else 0,
            "tp": round_price(tp) if tp != 0 else 0,
            "deviation": 10,  # 允许的价格偏差（点）
            "magic": 123456,  # 策略ID
            "type_time": mt5.ORDER_TIME_GTC,  # 订单有效期
            "type_filling": mt5.ORDER_FILLING_IOC,  # 改为立即成交或取消
        }

        # 打印订单信息，增加透明度
        self.log_and_print(f"发送{order_type_str}订单 - 手数: {volume}, 价格: {price:.2f}, "
                           f"止损: {sl:.2f}, 止盈: {tp:.2f}")

        # 发送订单
        result = mt5.order_send(request)
        if result is None:
            error_msg = "订单发送失败：MT5返回None，可能连接中断或请求无效"
            self.log_and_print(error_msg)
            logging.error(error_msg)
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"订单发送失败！错误代码：{result.retcode}, 描述：{mt5.last_error()}"
            self.log_and_print(error_msg)
            logging.error(error_msg)
            return False

        # 保存当前持仓的订单编号
        self.current_position_ticket = result.order

        # 打印订单成功信息
        self.log_and_print(f"{order_type_str}订单执行成功！订单号: {result.order}, 订单编号: {result.order}")
        return True

    def open_position(self, signal, latest_data):
        """开仓操作"""
        global current_position, entry_price, entry_time, daily_trades, trade_sequence, last_trade_signal

        # 获取当前市场价格
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            self.log_and_print("获取实时报价失败")
            return False

        # 确定开仓价格
        price = tick.ask if signal == 1 else tick.bid

        # 计算止损止盈
        atr = latest_data["ATR"] if not np.isnan(latest_data["ATR"]) else 0.001
        sl_multiplier = 0.8
        tp_multiplier = 1.6
        
        # 修复点差计算：使用point属性获取正确价格单位
        spread = mt5.symbol_info(SYMBOL).spread
        spread_value = spread * self.point  # 正确计算点差的实际值
        
        # 价格精度调整（黄金通常为2位小数）
        def round_price(price):
            return round(price, 2)

        if signal == 1:  # 买入
            # 修正点差影响计算：止损需额外扣除点差，止盈需额外增加点差
            sl = round_price(price - (atr * sl_multiplier + spread_value * 1.5))
            tp = round_price(price + (atr * tp_multiplier - spread_value * 1.5))
            success = self.send_order(1, LOT_SIZE, price, sl, tp)
        else:  # 卖出
            # 修正点差影响计算：止损需额外增加点差，止盈需额外扣除点差
            sl = round_price(price + (atr * sl_multiplier + spread_value * 1.5))
            tp = round_price(price - (atr * tp_multiplier - spread_value * 1.5))
            success = self.send_order(-1, LOT_SIZE, price, sl, tp)

        if success:
            current_position = signal
            entry_price = price  # 使用正确的开仓价格
            entry_time = datetime.now()
            daily_trades += 1
            trade_sequence += 1  # 更新交易序列
            last_trade_signal = signal  # 记录当前交易信号
            # 使用send_order返回的订单号
            self.log_and_print(f"开仓成功 - 方向: {signal}, 价格: {price}, 手数: {LOT_SIZE}, "
                               f"当前交易次数: {daily_trades}, 订单编号: {self.current_position_ticket}, 交易序列: {trade_sequence}")
            # 增加持仓信息透明度
            self.log_current_position_info()
            return True
        return False

    def close_position(self, reason="策略平仓"):
        """平仓操作"""
        global current_position, entry_price, entry_time, daily_loss, total_loss, consecutive_losses
        
        if current_position == 0:
            self.log_and_print("无持仓，无法平仓")
            return False

        # 获取当前价格
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return False

        # 记录开仓时的价格信息用于后续计算
        initial_entry_price = entry_price
        initial_position = current_position

        # 发送订单
        close_success = self.send_order(-current_position, LOT_SIZE, tick.ask if current_position == -1 else tick.bid)
        if not close_success:
            return False

        # 查询交易历史以确定实际的平仓原因
        try:
            # 获取最近几分钟的交易历史
            from_time = datetime.now() - timedelta(minutes=2)
            to_time = datetime.now() + timedelta(minutes=1)
            history_deals = mt5.history_deals_get(from_time, to_time)
            
            actual_reason = reason  # 默认使用传入的原因
            
            if history_deals is not None:
                # 按时间倒序查找最近的平仓交易
                for deal in reversed(history_deals):
                    # 确认这是针对我们持仓的平仓交易
                    if (hasattr(deal, 'position_id') and 
                        deal.position_id == self.current_position_ticket and
                        deal.type in [mt5.DEAL_TYPE_SELL, mt5.DEAL_TYPE_BUY] and  # 平仓类型
                        abs(deal.volume) == LOT_SIZE):
                        
                        # 根据盈亏判断实际原因
                        if deal.profit < 0:
                            actual_reason = "止损"
                        elif deal.profit > 0:
                            actual_reason = "止盈"
                        else:
                            actual_reason = "平仓"
                        break
                        
            final_reason = actual_reason
            
        except Exception as e:
            self.log_and_print(f"查询交易历史确定平仓原因时出错: {str(e)}")
            final_reason = reason  # 如果查询失败，使用原始原因

        # 计算盈亏（使用实际成交价格）
        current_price = tick.bid if current_position == 1 else tick.ask
        profit = (current_price - initial_entry_price) * self.contract_size * LOT_SIZE if initial_position == 1 else \
            (initial_entry_price - current_price) * self.contract_size * LOT_SIZE

        # 更新风控统计
        if profit < 0:
            daily_loss += profit
            total_loss += profit
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        self.log_and_print(f"平仓成功 - 方向: {-current_position}, 价格: {current_price:.2f}, 盈亏: {profit:.2f}, 原因: {final_reason}")
        
        # 根据平仓原因设置下次交易方向
        global last_close_reason
        last_close_reason = final_reason
        
        # 更新内部状态
        current_position = 0
        entry_price = 0.0
        entry_time = None
        self.current_position_ticket = None
        
        # 增加持仓信息透明度
        self.log_and_print("平仓完成，当前无持仓")
        self.log_current_position_info()
        
        return True

    def check_close_conditions(self, latest_data):
        """检查平仓条件"""
        global current_position, entry_price, entry_time

        # 检查MT5实际持仓状态，同步内部状态
        if current_position != 0 and hasattr(self, 'current_position_ticket') and self.current_position_ticket:
            # 检查持仓是否还存在
            position_info = mt5.positions_get(ticket=self.current_position_ticket)
            if position_info is None or len(position_info) == 0:
                # 持仓已不存在，可能是MT5自动平仓（止损/止盈触发）
                # 查询MT5历史交易记录来确定平仓原因
                global trade_sequence, last_trade_signal, last_close_reason
                
                # 获取最近的交易历史记录
                deal_history = mt5.history_deals_get(datetime.now() - timedelta(minutes=5), datetime.now())
                close_reason = "未知"
                if deal_history is not None and len(deal_history) > 0:
                    # 查找与当前订单相关的平仓记录
                    for deal in deal_history:
                        if hasattr(deal, 'position_id') and deal.position_id == self.current_position_ticket:
                            # 根据盈亏判断平仓原因
                            if deal.profit < 0:
                                close_reason = "止损"
                            elif deal.profit > 0:
                                close_reason = "止盈"
                            else:
                                close_reason = "平仓"
                            break
                
                # 确定后续交易方向
                next_signal = current_position  # 止损后顺着原方向交易
                if close_reason == "止盈":  # 止盈后反向交易
                    next_signal = -current_position
                
                # 更新内部状态
                current_position = 0
                entry_price = 0.0
                entry_time = None
                self.current_position_ticket = None
                
                # 记录平仓原因
                global last_close_reason
                last_close_reason = close_reason
                
                # 执行后续交易（如果有信号）
                if next_signal != 0:
                    self.log_and_print(f"检测到MT5自动平仓，原因: {close_reason}，准备后续交易: {next_signal}")
                else:
                    self.log_and_print(f"检测到MT5自动平仓，原因: {close_reason}，无后续交易信号")
                # 重置交易序列和信号
                trade_sequence = 0
                last_trade_signal = 0
                return True

        if current_position == 0:
            return False

        # 获取MT5实际持仓信息
        if not hasattr(self, 'current_position_ticket') or not self.current_position_ticket:
            return False
            
        position_info = mt5.positions_get(ticket=self.current_position_ticket)
        if not position_info:
            # 更新内部状态
            current_position = 0
            entry_price = 0.0
            entry_time = None
            self.current_position_ticket = None
            return False

        # 检查时间止损
        if entry_time and (datetime.now() - entry_time).total_seconds() / 3600 >= TIME_STOP_LOSS:
            self.close_position("时间止损")
            return True

        return False

    def end_of_day_summary(self):
        """输出每日交易统计信息"""
        global daily_trades, daily_loss
        self.log_and_print("\n当日交易总结：")
        self.log_and_print(f"- 总交易次数: {daily_trades}")
        self.log_and_print(f"- 当日亏损: ${daily_loss:.2f}")

        # 重置每日统计变量
        daily_trades = 0
        daily_loss = 0.0

    def run(self):
        """运行实时交易策略"""
        self.log_and_print("开始实时交易...")
        try:
            while True:
                # 检查风控限制
                if total_loss >= TOTAL_MAX_LOSS:
                    self.log_and_print(f"达到总最大亏损({TOTAL_MAX_LOSS}美元)，停止交易")
                    if current_position != 0:
                        self.close_position("达到总亏损限制")
                    break

                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                    self.log_and_print(f"达到最大连续亏损({MAX_CONSECUTIVE_LOSSES}次)，暂停交易30分钟")
                    time.sleep(1800)  # 暂停30分钟
                    continue

                if daily_loss >= DAILY_MAX_LOSS:
                    self.log_and_print(f"达到当日最大亏损({DAILY_MAX_LOSS}美元)，今日停止交易")
                    if current_position != 0:
                        self.close_position("达到当日亏损限制")
                    # 等待到次日
                    tomorrow = datetime.now() + timedelta(days=1)
                    next_day = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 1)
                    sleep_time = (next_day - datetime.now()).total_seconds()
                    self.log_and_print(f"等待至次日，休眠 {sleep_time / 3600:.2f} 小时")
                    time.sleep(sleep_time)
                    self.reset_daily_stats()
                    continue

                # 更新数据和指标
                self.update_real_time_data()
                latest_data = self.calculate_indicators()

                # 检查平仓条件
                self.check_close_conditions(latest_data)

                # 开仓逻辑
                if current_position == 0 and daily_trades < MAX_DAILY_TRADES:
                    signal, reasons = self.generate_real_time_signal(latest_data)
                    if signal != 0:
                        self.log_and_print(f"生成信号：{signal}，理由：{'; '.join(reasons)}")
                        self.log_and_print(f"交易序列: {trade_sequence}, 上次信号: {last_trade_signal}, 上次平仓原因: '{last_close_reason}'")
                        self.open_position(signal, latest_data)
                    else:
                        self.log_and_print(f"无交易信号，原因：{'; '.join(reasons)}")

                # 每分钟检查一次
                self.log_and_print("等待下一次检查...")
                time.sleep(60)

        except KeyboardInterrupt:
            self.log_and_print("用户中断交易")
            if current_position != 0:
                self.close_position("用户中断")
        except Exception as e:
            error_msg = f"发生未预期错误: {str(e)}"
            self.log_and_print(error_msg)
            logging.error(error_msg, exc_info=True)
            if current_position != 0:
                self.close_position("策略错误")
        finally:
            mt5.shutdown()
            self.log_and_print("交易结束，断开MT5连接")
            self.end_of_day_summary()

    def initialize_trade_state_from_history(self):
        """从历史交易记录初始化交易状态"""
        global trade_sequence, last_trade_signal, last_close_reason, current_position
        try:
            # 获取最近30分钟的历史交易数据
            to_time = datetime.now()
            from_time = to_time - timedelta(minutes=30)
            
            # 获取历史交易记录
            history_deals = mt5.history_deals_get(from_time, to_time)
            
            if history_deals is None or len(history_deals) == 0:
                self.log_and_print("最近30分钟内无历史交易记录，使用默认初始化")
                return
            
            # 过滤出当前交易对的交易，并按时间排序
            symbol_deals = []
            for deal in history_deals:
                # 只考虑平仓交易（ DEAL_TYPE_BUY 或 DEAL_TYPE_SELL ）
                if (hasattr(deal, 'symbol') and deal.symbol == SYMBOL and 
                    hasattr(deal, 'type') and deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]):
                    symbol_deals.append(deal)
            
            # 按时间排序
            symbol_deals.sort(key=lambda x: x.time)
            
            if len(symbol_deals) == 0:
                self.log_and_print("最近30分钟内无相关平仓交易记录，使用默认初始化")
                return
            
            # 获取最近的平仓交易
            last_deal = symbol_deals[-1]
            
            # 检查是否包含必要的属性
            if not hasattr(last_deal, 'profit'):
                self.log_and_print("历史交易记录缺少盈利属性，使用默认初始化")
                return
            
            # 根据最近平仓交易的盈亏情况设置last_close_reason
            if last_deal.profit > 0:
                last_close_reason = "止盈"
                self.log_and_print(f"从历史交易初始化状态 - 上次平仓原因: {last_close_reason} (盈利: {last_deal.profit:.2f})")
            elif last_deal.profit < 0:
                last_close_reason = "止损"
                self.log_and_print(f"从历史交易初始化状态 - 上次平仓原因: {last_close_reason} (亏损: {last_deal.profit:.2f})")
            else:
                last_close_reason = "平仓"
                self.log_and_print(f"从历史交易初始化状态 - 上次平仓原因: {last_close_reason} (盈亏平衡: {last_deal.profit:.2f})")
            
        except Exception as e:
            self.log_and_print(f"从历史交易初始化状态时出错: {str(e)}")

if __name__ == "__main__":
    trader = FTMORealTimeTrader()
    trader.run()
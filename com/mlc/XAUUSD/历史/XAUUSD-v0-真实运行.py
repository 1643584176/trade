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
    format='%(asctime)s - %(levelname)s - %(message)s',
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

        # 打印初始化信息
        self.log_and_print(f"初始化完成 - 交易品种: {SYMBOL}, 手数: {LOT_SIZE}, 合约大小: {self.contract_size}")

    def get_current_time(self):
        """获取MT5服务器时间"""
        try:
            # 获取当前报价，其中包含服务器时间
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                # 可以记录日志或返回默认值
                return "无法获取服务器时间"

            return datetime.fromtimestamp(tick.time).strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            # 处理可能的异常，如时间转换错误等
            return f"获取时间出错: {str(e)}"

    def log_and_print(self, message):
        """同时打印到控制台和日志文件"""
        print(f"[{self.get_current_time()}] {message}")
        logging.info(message)

    def connect_mt5(self):
        current_time = self.get_current_time()
        if not mt5.initialize():
            error_msg = f"MT5连接失败！错误代码：{mt5.last_error()}"
            print(f"[{current_time}] {error_msg}")
            logging.error(error_msg)
            quit()
        self.log_and_print("MT5连接成功！")

        if not mt5.symbol_select(SYMBOL, True):
            error_msg = f"交易品种{SYMBOL}不可用！错误代码：{mt5.last_error()}"
            print(f"[{current_time}] {error_msg}")
            logging.error(error_msg)
            mt5.shutdown()
            quit()
        self.log_and_print(f"交易品种{SYMBOL}已就绪")

    def load_initial_data(self):
        global historical_data
        current_time = self.get_current_time()
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 30)
        if rates is None:
            error_msg = f"获取历史数据失败！错误代码：{mt5.last_error()}"
            print(f"[{current_time}] {error_msg}")
            logging.error(error_msg)
            mt5.shutdown()
            quit()

        historical_data = pd.DataFrame(rates)
        historical_data["time"] = pd.to_datetime(historical_data["time"], unit="s")
        historical_data.rename(columns={
            "time": "时间点",
            "open": "当前开盘价",
            "high": "当前最高价",
            "low": "当前最低价",
            "close": "当前收盘价",
            "tick_volume": "成交量"
        }, inplace=True)

        historical_data["日期"] = historical_data["时间点"].dt.date
        historical_data["小时"] = historical_data["时间点"].dt.hour
        historical_data["星期数"] = historical_data["时间点"].dt.weekday
        historical_data["周数"] = historical_data["时间点"].dt.isocalendar().week
        self.log_and_print(f"初始数据加载完成，共{len(historical_data)}根H1 K线")

    def reset_daily_stats(self):
        global daily_trades, daily_loss
        today = date.today()
        if entry_time and entry_time.date() != today:
            daily_trades = 0
            daily_loss = 0.0
        self.log_and_print(f"当日交易统计重置：今日交易次数={daily_trades}，当日亏损={daily_loss:.2f}美元")

    def update_real_time_data(self):
        global historical_data
        latest_rate = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, 1)
        if latest_rate is None:
            error_msg = f"更新实时数据失败！错误代码：{mt5.last_error()}"
            self.log_and_print(error_msg)
            logging.error(error_msg)
            return False

        latest_df = pd.DataFrame(latest_rate)
        latest_df["time"] = pd.to_datetime(latest_df["time"], unit="s")
        latest_df.rename(columns={
            "time": "时间点",
            "open": "当前开盘价",
            "high": "当前最高价",
            "low": "当前最低价",
            "close": "当前收盘价",
            "tick_volume": "成交量"
        }, inplace=True)
        latest_df["日期"] = latest_df["时间点"].dt.date
        latest_df["小时"] = latest_df["时间点"].dt.hour
        latest_df["星期数"] = latest_df["时间点"].dt.weekday
        latest_df["周数"] = latest_df["时间点"].dt.isocalendar().week

        if not historical_data.empty:
            last_existing_time = historical_data["时间点"].iloc[-1]
            if latest_df["时间点"].iloc[0] > last_existing_time:
                historical_data = pd.concat([historical_data, latest_df], ignore_index=True)
                historical_data = historical_data.tail(30)
                self.log_and_print(f"新增K线：{latest_df['时间点'].iloc[0]}，当前数据总量：{len(historical_data)}")
        else:
            # 如果历史数据为空，直接赋值
            historical_data = latest_df
            self.log_and_print(f"初始化数据：{len(historical_data)}根H1 K线")

        return True

    def calculate_indicators(self):
        global historical_data
        df = historical_data.copy()

        # 计算ATR
        high_low = df["当前最高价"] - df["当前最低价"]
        high_close = np.abs(df["当前最高价"] - df["当前收盘价"].shift())
        low_close = np.abs(df["当前最低价"] - df["当前收盘价"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df["ATR"] = pd.Series(np.max(ranges, axis=1)).rolling(ATR_PERIOD).mean()

        # 计算RSI
        delta = df["当前收盘价"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rs = gain / loss.replace(0, 0.001)
        df["RSI"] = 100 - (100 / (1 + rs))

        # 计算动量指标
        df["1周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(1)
        df["3周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(3)
        df["5周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(5)
        df["10周期动量"] = df["当前收盘价"] - df["当前收盘价"].shift(10)

        # 计算K线形态
        df["实体"] = np.abs(df["当前收盘价"] - df["当前开盘价"])
        df["上影线"] = df["当前最高价"] - np.maximum(df["当前开盘价"], df["当前收盘价"])
        df["下影线"] = np.minimum(df["当前开盘价"], df["当前收盘价"]) - df["当前最低价"]
        df["是否阳线"] = df["当前收盘价"] > df["当前开盘价"]

        # 计算当日开盘价与相对变化
        daily_open = df.groupby("日期")["当前开盘价"].first()
        df["当日开盘价"] = df["日期"].map(daily_open)
        df["相对开盘价变化"] = df["当前收盘价"] - df["当日开盘价"]

        # 计算周间关联 - 修复合并错误
        weekly_close = df.groupby("周数")["当前收盘价"].last().reset_index()
        weekly_close.columns = ["周数", "周收盘价"]
        weekly_close["前一周收盘价"] = weekly_close["周收盘价"].shift(1)
        weekly_close["周间趋势"] = np.where(weekly_close["周收盘价"] - weekly_close["前一周收盘价"] > 0, 1, -1)

        # 检查是否已有"周间趋势"列，如果有则先删除
        if "周间趋势" in df.columns:
            df = df.drop(columns=["周间趋势"])

        # 合并数据并指定后缀避免冲突
        df = df.merge(weekly_close[["周数", "周间趋势"]], on="周数", how="left", suffixes=('', '_y'))

        # 删除可能的重复列
        df = df.drop(columns=[col for col in df.columns if col.endswith('_y')])

        historical_data = df
        return df.iloc[-1]

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
        idx = df.index[-1]
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

    def generate_real_time_signal(self, latest_data):
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

        self.log_and_print(f"信号分析 - 时段: {session_type}, 星期: {weekday_num}, 小时: {hour}")

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

        if len(reasons) < 1:
            signal = 0
            reasons.append("信号不足")

        return signal, reasons

    def send_order(self, action, volume, price, sl=0, tp=0):
        """发送交易订单"""
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
        global current_position, entry_price, entry_time, daily_trades

        # 获取当前市场价格
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            self.log_and_print("获取实时报价失败")
            return False

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
            price = tick.ask
            # 修正点差影响计算：止损需额外扣除点差，止盈需额外增加点差
            sl = round_price(price - (atr * sl_multiplier + spread_value * 1.5))
            tp = round_price(price + (atr * tp_multiplier - spread_value * 1.5))
            success = self.send_order(1, LOT_SIZE, price, sl, tp)
        else:  # 卖出
            price = tick.bid
            # 修正点差影响计算：止损需额外增加点差，止盈需额外扣除点差
            sl = round_price(price + (atr * sl_multiplier + spread_value * 1.5))
            tp = round_price(price - (atr * tp_multiplier - spread_value * 1.5))
            success = self.send_order(-1, LOT_SIZE, price, sl, tp)

        if success:
            current_position = signal
            entry_price = price
            entry_time = datetime.now()
            daily_trades += 1
            # 使用send_order返回的订单号
            self.log_and_print(f"开仓成功 - 方向: {signal}, 价格: {price}, 手数: {LOT_SIZE}, "
                               f"当前交易次数: {daily_trades}, 订单编号: {self.current_position_ticket}")
            return True
        return False

    def close_position(self, reason):
        """平仓操作"""
        global current_position, entry_price, entry_time, daily_loss, consecutive_losses, total_loss

        # 检查MT5连接状态
        if not mt5.connected():
            self.log_and_print("MT5未连接，无法执行平仓操作")
            logging.error("MT5未连接，无法执行平仓操作")
            return False

        # 获取当前持仓信息
        if not hasattr(self, 'current_position_ticket') or not self.current_position_ticket:
            self.log_and_print("没有可平仓的持仓")
            logging.warning("没有可平仓的持仓")
            return False

        position_info = mt5.positions_get(ticket=self.current_position_ticket)
        if not position_info:
            self.log_and_print("无法获取持仓信息，可能已平仓或连接中断")
            logging.warning("无法获取持仓信息，可能已平仓或连接中断")
            return False
        
        # 获取当前价格
        symbol_info = mt5.symbol_info(position_info[0].symbol)
        if not symbol_info:
            self.log_and_print("无法获取交易品种信息，使用最后已知价格")
            logging.warning("无法获取交易品种信息，使用最后已知价格")
            # 修复未定义变量current_price问题
            if hasattr(self, 'current_price'):
                price = self.current_price
            else:
                self.log_and_print("无法获取当前价格，平仓操作失败")
                return False
        else:
            price = symbol_info.bid if position_info[0].type == mt5.POSITION_TYPE_BUY else symbol_info.ask

        # 继续执行平仓逻辑
        action = -1 if position_info[0].type == mt5.POSITION_TYPE_BUY else 1
        success = self.send_order(action, position_info[0].volume, price)

        if success:
            # 计算盈亏
            if position_info[0].type == mt5.POSITION_TYPE_BUY:
                profit = (price - position_info[0].price_open) * self.contract_size * position_info[0].volume
            else:
                profit = (position_info[0].price_open - price) * self.contract_size * position_info[0].volume

            self.log_and_print(f"平仓成功！原因：{reason}，盈亏：{profit:.2f}美元, "
                               f"持仓时间: {datetime.now() - entry_time}")
            # 更新风控统计
            if profit < 0:
                daily_loss += abs(profit)
                consecutive_losses += 1
                total_loss += abs(profit)
                self.log_and_print(f"亏损交易 - 当日累计亏损: {daily_loss:.2f}, "
                                   f"连续亏损次数: {consecutive_losses}, 总亏损: {total_loss:.2f}")
            else:
                consecutive_losses = 0
                self.log_and_print(f"盈利交易 - 盈利: {profit:.2f}美元")

            # 重置持仓状态
            current_position = 0
            entry_price = 0.0
            entry_time = None
            return True
        return False

    def check_close_conditions(self, latest_data):
        """检查平仓条件"""
        global current_position, entry_time

        if current_position == 0:
            return False

        # 获取当前价格
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            return False

        # 计算ATR
        atr = latest_data["ATR"] if not np.isnan(latest_data["ATR"]) else 0.001
        sl_multiplier = 0.8
        tp_multiplier = 1.6

        # 打印当前持仓状态
        current_price = tick.bid if current_position == 1 else tick.ask
        current_profit = (current_price - entry_price) * self.contract_size * LOT_SIZE if current_position == 1 else \
            (entry_price - current_price) * self.contract_size * LOT_SIZE
        self.log_and_print(f"当前持仓 - 方向: {current_position}, 开仓价: {entry_price:.4f}, "
                           f"当前价: {current_price:.4f}, 浮盈: {current_profit:.2f}美元, "
                           f"持仓时间: {datetime.now() - entry_time}")

        # 检查止盈止损
        if current_position == 1:  # 多仓
            sl = entry_price - atr * sl_multiplier
            tp = entry_price + atr * tp_multiplier
            if tick.bid <= sl:
                self.close_position("止损")
                return True
            if tick.bid >= tp:
                self.close_position("止盈")
                return True
        else:  # 空仓
            sl = entry_price + atr * sl_multiplier
            tp = entry_price - atr * tp_multiplier
            if tick.ask >= sl:
                self.close_position("止损")
                return True
            if tick.ask <= tp:
                self.close_position("止盈")
                return True

        # 检查时间止损
        if (datetime.now() - entry_time) >= timedelta(hours=TIME_STOP_LOSS):
            self.close_position("时间止损")
            return True

        return False

    def execute_trade(self, direction):
        """执行交易并打印详细信息"""
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            self.log_and_print("获取实时报价失败")
            return

        current_price = tick.ask if direction > 0 else tick.bid

        # 计算ATR用于设置止损止盈
        latest_data = self.calculate_indicators()
        atr = latest_data["ATR"] if not np.isnan(latest_data["ATR"]) else 0.001
        sl_multiplier = 0.8
        tp_multiplier = 1.6

        stop_loss = current_price - atr * sl_multiplier if direction > 0 else current_price + atr * sl_multiplier
        take_profit = current_price + atr * tp_multiplier if direction > 0 else current_price - atr * tp_multiplier

        # 打印交易详细信息
        self.log_and_print(f"执行交易: {'买入' if direction > 0 else '卖出'}")
        self.log_and_print(f"- 价格: {current_price}")
        self.log_and_print(f"- 止损: {stop_loss:.4f}")
        self.log_and_print(f"- 止盈: {take_profit:.4f}")

        # 执行交易
        if direction == 1:
            self.send_order(1, LOT_SIZE, current_price, stop_loss, take_profit)
        else:
            self.send_order(-1, LOT_SIZE, current_price, stop_loss, take_profit)

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
                        self.open_position(signal, latest_data)

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



if __name__ == "__main__":
    trader = FTMORealTimeTrader()
    trader.run()

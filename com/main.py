"""
FTMO XAUUSD 日内交易系统
时间关系：北京时间 = XAUUSD时间 + 6小时
交易时间：北京时间 9:00-19:00 = XAUUSD时间 3:00-13:00
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import json
import os


# ==================== 配置部分 ====================
class Config:
    """交易系统配置"""

    # 交易品种
    SYMBOL = "XAUUSD"
    LOT_SIZE = 100.0  # 黄金标准手大小（盎司）
    POINT = 0.01  # 黄金最小报价单位

    # 交易时间（基于XAUUSD时间）
    XAUUSD_TRADING_HOURS = {
        "start": 3,   # XAUUSD时间 03:00 (北京09:00)
        "end": 13     # XAUUSD时间 13:00 (北京19:00)
    }

    # 时段细分（XAUUSD时间）
    XAUUSD_TIME_WINDOWS = {
        "asia_morning": (3, 7),      # XAUUSD 03:00-07:00 (北京09:00-13:00)
        "london_preopen": (7, 10),   # XAUUSD 07:00-10:00 (北京13:00-16:00)
        "london_open": (10, 13),     # XAUUSD 10:00-13:00 (北京16:00-19:00)
    }

    # 固定手数设置
    FIXED_LOT_SIZES = {  # 变量名改成FIXED_LOT_SIZES（复数）
        "asia_morning": 0.10,    # 0.10手
        "london_preopen": 0.15,  # 0.15手
        "london_open": 0.20,     # 0.20手
        "off_hours": 0.00        # 不交易
    }

    # 各时段市场特征 - 需要调整止损以适应0.1-0.2手
    SESSION_CHARACTERISTICS = {
        "asia_morning": {
            "volatility": "低-中",
            "strategy": "延续/区间交易",
            "min_sl": 2.0,    # 减小到2美元（配合0.10手）
            "max_sl": 4.0,    # 减小到4美元
            "position_multiplier": 1.0  # 改为1.0
        },
        "london_preopen": {
            "volatility": "中",
            "strategy": "突破/动量",
            "min_sl": 3.0,    # 减小到3美元（配合0.15手）
            "max_sl": 6.0,    # 减小到6美元
            "position_multiplier": 1.0  # 改为1.0
        },
        "london_open": {
            "volatility": "高",
            "strategy": "动量突破",
            "min_sl": 4.0,    # 减小到4美元（配合0.20手）
            "max_sl": 8.0,    # 减小到8美元
            "position_multiplier": 1.0  # 改为1.0
        }
    }

    # 风险管理 - 需要提高风险比例
    ACCOUNT_BALANCE = 10000  # FTMO模拟账户余额
    RISK_PER_TRADE = 0.02    # 提高到2.0% 每笔交易风险（适应0.1-0.2手）
    DAILY_MAX_RISK = 0.04    # 4% 每日最大风险
    MAX_POSITIONS = 2        # 最大同时持仓数

    # 策略参数 - 减小ATR乘数
    STOP_LOSS_ATR_MULTIPLIER = 1.2  # 减小到1.2
    TAKE_PROFIT_RATIO = 1.5         # 降低到1.5:1（适应小止损）
    TRAILING_STOP_ACTIVATE = 8      # 美元盈利后激活追踪止损
    TRAILING_STOP_DISTANCE = 4      # 减小追踪止损距离

    # 技术指标参数
    EMA_FAST = 9
    EMA_SLOW = 21
    EMA_TREND = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.0

    # 交易限制
    TRADE_ON_FRIDAY = False  # 周五是否交易
    MAX_HOLDING_HOURS = 4    # 最大持仓时间


# ==================== 时间管理 ====================
class TimeManager:
    """时间管理器"""

    @staticmethod
    def get_mt5_time():
        current_beijing = datetime.now()  # 假设你的电脑是北京时间
        return current_beijing - timedelta(hours=6)

    @staticmethod
    def get_current_hour():
        """获取当前XAUUSD时间（小时）"""
        mt5_time = TimeManager.get_mt5_time()
        return mt5_time.hour + mt5_time.minute/60

    @staticmethod
    def get_beijing_time():
        """获取北京时间"""
        mt5_time = TimeManager.get_mt5_time()
        # 北京时间 = XAUUSD时间 + 6小时
        return mt5_time + timedelta(hours=6)

    @staticmethod
    def get_current_session():
        """获取当前交易时段"""
        xauusd_hour = TimeManager.get_current_hour()

        # 检查是否在交易时间内
        if Config.XAUUSD_TRADING_HOURS["start"] <= xauusd_hour < Config.XAUUSD_TRADING_HOURS["end"]:
            if Config.XAUUSD_TIME_WINDOWS["asia_morning"][0] <= xauusd_hour < Config.XAUUSD_TIME_WINDOWS["asia_morning"][1]:
                return "asia_morning"
            elif Config.XAUUSD_TIME_WINDOWS["london_preopen"][0] <= xauusd_hour < Config.XAUUSD_TIME_WINDOWS["london_preopen"][1]:
                return "london_preopen"
            elif Config.XAUUSD_TIME_WINDOWS["london_open"][0] <= xauusd_hour < Config.XAUUSD_TIME_WINDOWS["london_open"][1]:
                return "london_open"
        return "off_hours"

    @staticmethod
    def is_trading_time():
        """检查是否为交易时间"""
        return TimeManager.get_current_session() != "off_hours"

    @staticmethod
    def get_session_params(session):
        """获取时段参数"""
        return Config.SESSION_CHARACTERISTICS.get(session, {
            "volatility": "无",
            "strategy": "不交易",
            "min_sl": 0,
            "max_sl": 0,
            "position_multiplier": 0
        })

    @staticmethod
    def should_close_before_end():
        """检查是否应在交易结束前平仓"""
        xauusd_hour = TimeManager.get_current_hour()
        # XAUUSD时间 12:30开始准备平仓 (北京18:30)
        return xauusd_hour >= 12.5


# ==================== 日志系统 ====================
class TradeLogger:
    """交易日志系统"""

    def __init__(self):
        self.setup_logging()
        self.trade_log = []

    def setup_logging(self):
        """配置日志"""
        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger('FTMOTrader')
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = RotatingFileHandler('logs/trading.log', maxBytes=10485760, backupCount=5)
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_trade(self, trade_type, price, lot_size, sl, tp):
        """记录交易"""
        trade_record = {
            "timestamp": datetime.utcnow(),
            "type": trade_type,
            "symbol": Config.SYMBOL,
            "price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp
        }
        self.trade_log.append(trade_record)
        self.logger.info(f"交易执行: {trade_record}")


# ==================== 数据管理 ====================
class DataManager:
    """数据获取和管理"""

    def __init__(self):
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }

    def get_data(self, timeframe, n_bars=100):
        """获取K线数据"""
        tf = self.timeframes.get(timeframe, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, tf, 0, n_bars)

        if rates is None:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        # 移动平均线
        df['ema_fast'] = df['close'].ewm(span=Config.EMA_FAST).mean()
        df['ema_slow'] = df['close'].ewm(span=Config.EMA_SLOW).mean()
        df['ema_trend'] = df['close'].ewm(span=Config.EMA_TREND).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(Config.ATR_PERIOD).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=Config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=Config.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df


# ==================== 风险管理 ====================
class RiskManager:
    """风险管理器"""

    def __init__(self):
        self.initial_balance = Config.ACCOUNT_BALANCE

    def calculate_position_size(self, stop_loss_distance, session):
        """使用固定手数 + 风险检查"""
        # 直接从配置获取固定手数
        fixed_lot = Config.FIXED_LOT_SIZES.get(session, 0.10)

        # 检查风险是否超标
        account_info = mt5.account_info()
        if account_info is None:
            current_equity = Config.ACCOUNT_BALANCE
        else:
            current_equity = account_info.equity

        # 计算实际风险
        pip_value = 10  # 美元/标准手
        pip_distance = stop_loss_distance / Config.POINT
        actual_risk = fixed_lot * pip_distance * pip_value

        # 计算允许的最大风险（2%）
        max_allowed_risk = current_equity * Config.RISK_PER_TRADE

        # 如果实际风险超过允许风险，按比例减小手数
        if actual_risk > max_allowed_risk and pip_distance > 0:
            fixed_lot = fixed_lot * (max_allowed_risk / actual_risk)
            fixed_lot = round(fixed_lot, 2)  # 保留2位小数
            print(f"⚠️ 手数调整: 从{Config.FIXED_LOT_SIZES.get(session, 0.10)}调整为{fixed_lot}手")

        # 确保最小手数0.01
        return max(0.01, fixed_lot)

    def validate_trade_risk(self, lot_size, stop_loss_distance, session):
        """验证交易风险是否安全"""
        account_info = mt5.account_info()
        if account_info is None:
            return True, ""

        current_equity = account_info.equity

        # 计算实际风险
        pip_value = 10
        pip_distance = stop_loss_distance / Config.POINT
        actual_risk = lot_size * pip_distance * pip_value

        # 计算允许的最大风险
        max_allowed_risk = current_equity * Config.RISK_PER_TRADE

        if actual_risk > max_allowed_risk:
            return False, f"实际风险${actual_risk:.2f}超过限制${max_allowed_risk:.2f}"

        return True, ""

    def check_daily_limit(self):
        """检查每日亏损限制"""
        account_info = mt5.account_info()
        if account_info is None:
            return True, ""

        current_equity = account_info.equity
        daily_loss = self.initial_balance - current_equity

        # 检查FTMO规则（5%亏损限制）
        ftmo_max_loss = Config.ACCOUNT_BALANCE * 0.05
        if daily_loss > ftmo_max_loss:
            return False, f"达到FTMO每日亏损限制"

        # 检查内部限制（4%）
        internal_max_loss = Config.ACCOUNT_BALANCE * Config.DAILY_MAX_RISK
        if daily_loss > internal_max_loss:
            return False, f"达到内部日亏损限制"

        return True, ""

    def check_trading_time(self):
        """检查交易时间"""
        if not TimeManager.is_trading_time():
            return False, "非交易时段"
        return True, ""

    def check_max_positions(self):
        """检查最大持仓数"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if positions is None:
            return True, ""

        if len(positions) >= Config.MAX_POSITIONS:
            return False, f"达到最大持仓限制"
        return True, ""


# ==================== 交易策略 ====================
class XAUUSDStrategy:
    """黄金交易策略"""

    def __init__(self, data_manager):
        self.dm = data_manager

    def analyze_market(self):
        """分析市场状态"""
        session = TimeManager.get_current_session()

        # 获取数据
        df_m5 = self.dm.get_data('M5', 100)
        df_m15 = self.dm.get_data('M15', 50)

        if df_m5 is None or df_m15 is None:
            return {"error": "无法获取数据"}

        # 计算指标
        df_m5 = self.dm.calculate_indicators(df_m5)
        df_m15 = self.dm.calculate_indicators(df_m15)

        # 获取当前价格和ATR
        current_price = df_m5['close'].iloc[-1]
        atr = df_m5['atr'].iloc[-1] if not pd.isna(df_m5['atr'].iloc[-1]) else 5.0

        # 检查交易信号
        signal = self._check_trading_signal(df_m5, df_m15, current_price, session)

        return {
            "session": session,
            "current_price": current_price,
            "atr": atr,
            "signal": signal,
            "session_params": TimeManager.get_session_params(session)
        }

    def _check_trading_signal(self, df_m5, df_m15, price, session):
        """检查交易信号"""
        if session == "asia_morning":
            return self._asia_morning_signal(df_m5, df_m15, price)
        elif session == "london_preopen":
            return self._london_preopen_signal(df_m5, df_m15, price)
        elif session == "london_open":
            return self._london_open_signal(df_m5, df_m15, price)
        return None

    def _asia_morning_signal(self, df_m5, df_m15, price):
        """亚洲上午信号"""
        if len(df_m15) < 20:
            return None

        # 获取隔夜区间
        overnight_high = df_m15['high'].iloc[-20:-1].max()
        overnight_low = df_m15['low'].iloc[-20:-1].min()
        rsi = df_m5['rsi'].iloc[-1]

        # 突破策略
        if price > overnight_high and rsi < 65:
            return {"type": "BUY", "reason": "突破隔夜高点"}
        elif price < overnight_low and rsi > 35:
            return {"type": "SELL", "reason": "跌破隔夜低点"}

        return None

    def _london_preopen_signal(self, df_m5, df_m15, price):
        """伦敦开盘前信号"""
        if len(df_m15) < 12:
            return None

        # 获取亚洲时段区间
        asian_high = df_m15['high'].iloc[-12:-1].max()
        asian_low = df_m15['low'].iloc[-12:-1].min()
        rsi = df_m5['rsi'].iloc[-1]

        # 突破亚洲区间
        if price > asian_high and rsi < 70:
            return {"type": "BUY", "reason": "突破亚洲时段区间"}
        elif price < asian_low and rsi > 30:
            return {"type": "SELL", "reason": "跌破亚洲时段区间"}

        return None

    def _london_open_signal(self, df_m5, df_m15, price):
        """伦敦开盘信号"""
        if len(df_m15) < 8:
            return None

        # 获取开盘后区间
        london_high = df_m15['high'].iloc[-8:-1].max()
        london_low = df_m15['low'].iloc[-8:-1].min()
        rsi = df_m5['rsi'].iloc[-1]

        # 动量突破
        if price > london_high and rsi < 75:
            return {"type": "BUY", "reason": "伦敦开盘动量突破"}
        elif price < london_low and rsi > 25:
            return {"type": "SELL", "reason": "伦敦开盘动量跌破"}

        return None


# ==================== 交易执行 ====================
class TradeExecutor:
    """交易执行器"""

    def __init__(self, strategy, risk_manager, logger):
        self.strategy = strategy
        self.rm = risk_manager
        self.logger = logger

    def execute_trade(self, signal, market_data):
        """执行交易"""
        # 检查各种限制
        checks = [
            self.rm.check_trading_time(),
            self.rm.check_daily_limit(),
            self.rm.check_max_positions()
        ]

        for check_result, check_msg in checks:
            if not check_result:
                return False, check_msg

        if signal is None:
            return False, "无交易信号"

        current_price = market_data['current_price']
        atr = market_data['atr']
        session = market_data['session']
        session_params = market_data['session_params']

        # 先获取固定手数
        fixed_lot = Config.FIXED_LOT_SIZES.get(session, 0.10)

        # 计算允许的最大止损（基于固定手数和风险限制）
        account_info = mt5.account_info()
        if account_info is None:
            current_equity = Config.ACCOUNT_BALANCE
        else:
            current_equity = account_info.equity

        # 允许的最大风险金额（2%）
        max_allowed_risk = current_equity * Config.RISK_PER_TRADE

        # 基于固定手数反推最大止损距离
        pip_value = 10  # 美元/标准手
        max_pip_distance = max_allowed_risk / (fixed_lot * pip_value)
        max_sl_from_risk = max_pip_distance * Config.POINT  # 转换为美元

        # 计算ATR止损
        atr_sl = atr * Config.STOP_LOSS_ATR_MULTIPLIER

        # 获取时段限制
        min_sl = session_params.get("min_sl", 2.0)
        max_sl = session_params.get("max_sl", 6.0)

        # 最终止损：取多个限制中的最小值
        stop_loss_distance = min(
            atr_sl,               # ATR计算
            max_sl,               # 时段最大
            max_sl_from_risk      # 风险控制最大
        )

        # 确保不小于最小止损
        stop_loss_distance = max(stop_loss_distance, min_sl)

        # 验证风险是否安全
        is_valid, risk_msg = self.rm.validate_trade_risk(fixed_lot, stop_loss_distance, session)
        if not is_valid:
            return False, f"风险控制: {risk_msg}"

        # 计算止盈
        if signal['type'] == "BUY":
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + (stop_loss_distance * Config.TAKE_PROFIT_RATIO)
        else:
            stop_loss = current_price + stop_loss_distance
            take_profit = current_price - (stop_loss_distance * Config.TAKE_PROFIT_RATIO)

        # 计算实际仓位（可能被风险控制调整过）
        lot_size = self.rm.calculate_position_size(stop_loss_distance, session)

        # 准备订单
        order_type = mt5.ORDER_TYPE_BUY if signal['type'] == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Config.SYMBOL,
            "volume": lot_size,
            "type": order_type,
            "price": current_price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,
            "magic": 234000,
            "comment": signal['reason'],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 发送订单
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"订单失败: {result.comment}"

        # 记录交易
        self.logger.log_trade(signal['type'], current_price, lot_size, stop_loss, take_profit)
        return True, "交易成功"

    def manage_positions(self):
        """管理现有仓位"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if positions is None:
            return

        for position in positions:
            self._check_time_exit(position)

    def _check_time_exit(self, position):
        """检查时间退出"""
        # 交易结束前平仓
        if TimeManager.should_close_before_end():
            self._close_position(position, "交易时段结束前平仓")

        # 持仓超时平仓
        position_time = pd.to_datetime(position.time, unit='s')
        holding_hours = (datetime.utcnow() - position_time).total_seconds() / 3600
        if holding_hours > Config.MAX_HOLDING_HOURS:
            self._close_position(position, "持仓超时")

    def _close_position(self, position, reason):
        """平仓"""
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": position.ticket,
            "symbol": Config.SYMBOL,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid,
            "deviation": 10,
            "magic": 234000,
            "comment": reason,
        }
        mt5.order_send(request)


# ==================== 主交易系统 ====================
class FTMOTradingSystem:
    """主交易系统"""

    def __init__(self):
        self.logger = TradeLogger()
        self.dm = DataManager()
        self.rm = RiskManager()
        self.strategy = XAUUSDStrategy(self.dm)
        self.executor = TradeExecutor(self.strategy, self.rm, self.logger)
        self.is_running = False

    def initialize_mt5(self):
        """初始化MT5连接"""
        if not mt5.initialize():
            self.logger.logger.error("MT5初始化失败")
            return False
        return True

    def run(self):
        """运行主循环"""
        self.is_running = True

        while self.is_running:
            try:
                # 管理仓位
                self.executor.manage_positions()

                # 每30秒检查一次信号
                if TimeManager.is_trading_time():
                    self._check_trading_signal()

                # 显示状态
                self._display_status()

                time.sleep(30)

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.logger.error(f"运行错误: {e}")
                time.sleep(10)

    def _check_trading_signal(self):
        """检查交易信号"""
        market_data = self.strategy.analyze_market()
        if "error" in market_data:
            return

        signal = market_data.get("signal")
        if signal:
            success, message = self.executor.execute_trade(signal, market_data)
            if success:
                self.logger.logger.info(f"交易成功: {message}")
            else:
                self.logger.logger.info(f"交易未执行: {message}")

    def _display_status(self):
        """显示当前状态"""
        try:
            # 获取时间信息
            xauusd_time = TimeManager.get_mt5_time()
            beijing_time = TimeManager.get_beijing_time()
            session = TimeManager.get_current_session()
            session_params = TimeManager.get_session_params(session)

            # 获取账户信息
            account_info = mt5.account_info()
            if account_info:
                equity = account_info.equity
                balance = account_info.balance
                daily_pnl = equity - balance
            else:
                equity = balance = daily_pnl = 0

            # 获取持仓
            positions = mt5.positions_get(symbol=Config.SYMBOL)
            position_count = len(positions) if positions else 0

            # 清屏显示
            os.system('cls' if os.name == 'nt' else 'clear')


            print(f"FTMO XAUUSD 交易系统 | 状态: {'运行中' if self.is_running else '已停止'} | XAUUSD时间: {xauusd_time.strftime('%H:%M:%S')} | 北京时间: {beijing_time.strftime('%H:%M:%S')} | 交易时段: {session} | 策略: {session_params.get('strategy', '不交易')}")
            print(f"账户净值: ${equity:.2f} | 余额: ${balance:.2f} | 当日盈亏: ${daily_pnl:.2f} | 持仓数量: {position_count}")

            if position_count > 0:
                print("当前持仓:")
                for pos in positions:
                    print(f"  #{pos.ticket}: {pos.type_desc()} {pos.volume}手 @ {pos.price_open:.2f} | 盈亏: ${pos.profit:.2f}")

            # FTMO规则状态
            ftmo_limit = Config.ACCOUNT_BALANCE * 0.05
            ftmo_used = abs(min(daily_pnl, 0))
            ftmo_remaining = ftmo_limit - ftmo_used

            # 显示固定手数信息
            fixed_lot = Config.FIXED_LOT_SIZES.get(session, 0.00)
            max_sl = session_params.get('max_sl', 0)


            print(f"[FTMO规则] 限额: ${ftmo_limit:.2f} | 已用: ${ftmo_used:.2f} | 剩余: ${ftmo_remaining:.2f}")

            if session != "off_hours":
                print(f"[交易参数] 固定手数: {fixed_lot}手 | 最大止损: ${max_sl:.1f} | 风险比例: {Config.RISK_PER_TRADE*100:.1f}%")



        except Exception as e:
            print(f"状态显示错误: {e}")

    def stop(self):
        """停止系统"""
        self.is_running = False
        mt5.shutdown()
        self.logger.logger.info("交易系统已停止")


# ==================== 启动脚本 ====================
if __name__ == "__main__":
    system = FTMOTradingSystem()

    # 显示初始时间信息
    xauusd_time = TimeManager.get_mt5_time()
    beijing_time = TimeManager.get_beijing_time()

    if system.initialize_mt5():
        try:
            system.run()
        except Exception as e:
            system.logger.logger.error(f"系统运行异常: {e}")
        finally:
            system.stop()
    else:
        print("MT5初始化失败")
"""
FTMO XAUUSD 日内交易系统 - GMT时间版本
作者：交易系统助手
版本：2.0
说明：专为GMT 01:00-11:00（北京时间9:00-19:00）黄金交易设计
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
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

    # 交易时间（GMT时间，对应北京时间9:00-19:00）
    GMT_TRADING_HOURS = {
        "start": 1,   # GMT 01:00 (北京09:00)
        "end": 11     # GMT 11:00 (北京19:00)
    }

    # 时段细分（GMT时间）
    GMT_TIME_WINDOWS = {
        "asia_to_london": (1, 5),      # GMT 01:00-05:00 (北京09:00-13:00)
        "london_morning": (5, 8),      # GMT 05:00-08:00 (北京13:00-16:00)
        "london_open": (8, 11),        # GMT 08:00-11:00 (北京16:00-19:00)
    }

    # 各时段市场特征
    SESSION_CHARACTERISTICS = {
        "asia_to_london": {
            "volatility": "低-中",
            "strategy": "延续/区间交易",
            "max_sl": 4,  # 最大止损(美元)
            "position_multiplier": 0.5  # 仓位乘数
        },
        "london_morning": {
            "volatility": "中",
            "strategy": "突破/动量",
            "max_sl": 6,
            "position_multiplier": 0.75
        },
        "london_open": {
            "volatility": "高",
            "strategy": "动量突破",
            "max_sl": 8,
            "position_multiplier": 1.0
        }
    }

    # 风险管理
    ACCOUNT_BALANCE = 10000  # FTMO模拟账户余额
    RISK_PER_TRADE = 0.005   # 0.5% 每笔交易风险
    DAILY_MAX_RISK = 0.03    # 3% 每日最大风险（留2%缓冲给FTMO的5%限制）
    MAX_POSITIONS = 2        # 最大同时持仓数

    # 策略参数
    STOP_LOSS_ATR_MULTIPLIER = 1.5
    TAKE_PROFIT_RATIO = 1.8  # 盈亏比
    TRAILING_STOP_ACTIVATE = 10  # 美元盈利后激活追踪止损
    TRAILING_STOP_DISTANCE = 5   # 追踪止损距离

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
class XAUUSDTimeManager:
    """XAUUSD时间管理器（基于GMT时间）"""

    @staticmethod
    def get_gmt_time():
        """获取当前GMT时间"""
        return datetime.utcnow()

    @staticmethod
    def get_current_gmt_hour():
        """获取当前GMT小时（含分钟）"""
        utc_now = datetime.utcnow()
        return utc_now.hour + utc_now.minute/60

    @staticmethod
    def get_current_session():
        """获取当前交易时段"""
        gmt_hour = XAUUSDTimeManager.get_current_gmt_hour()

        # 检查是否在交易时间内
        if Config.GMT_TRADING_HOURS["start"] <= gmt_hour < Config.GMT_TRADING_HOURS["end"]:
            if Config.GMT_TIME_WINDOWS["asia_to_london"][0] <= gmt_hour < Config.GMT_TIME_WINDOWS["asia_to_london"][1]:
                return "asia_to_london"
            elif Config.GMT_TIME_WINDOWS["london_morning"][0] <= gmt_hour < Config.GMT_TIME_WINDOWS["london_morning"][1]:
                return "london_morning"
            elif Config.GMT_TIME_WINDOWS["london_open"][0] <= gmt_hour < Config.GMT_TIME_WINDOWS["london_open"][1]:
                return "london_open"
        return "off_hours"

    @staticmethod
    def is_trading_time():
        """检查是否为交易时间"""
        return XAUUSDTimeManager.get_current_session() != "off_hours"

    @staticmethod
    def get_session_params(session):
        """获取时段参数"""
        return Config.SESSION_CHARACTERISTICS.get(session, {
            "volatility": "低",
            "strategy": "观望",
            "max_sl": 3,
            "position_multiplier": 0
        })

    @staticmethod
    def get_beijing_time(gmt_time=None):
        """GMT转北京时间"""
        if gmt_time is None:
            gmt_time = datetime.utcnow()
        return gmt_time + timedelta(hours=8)

    @staticmethod
    def format_time_display():
        """格式化时间显示"""
        gmt_now = datetime.utcnow()
        beijing_now = gmt_now + timedelta(hours=8)
        gmt_hour = gmt_now.hour + gmt_now.minute/60

        session = XAUUSDTimeManager.get_current_session()
        session_params = XAUUSDTimeManager.get_session_params(session)

        return {
            "gmt_time": gmt_now.strftime("%H:%M:%S"),
            "beijing_time": beijing_now.strftime("%H:%M:%S"),
            "gmt_hour": round(gmt_hour, 2),
            "session": session,
            "strategy": session_params["strategy"],
            "volatility": session_params["volatility"]
        }

    @staticmethod
    def should_close_before_end():
        """检查是否应在交易结束前平仓"""
        gmt_hour = XAUUSDTimeManager.get_current_gmt_hour()
        # GMT 10:30开始准备平仓 (北京18:30)
        return gmt_hour >= 10.5

    @staticmethod
    def is_friday_close_time():
        """检查是否是周五收盘时间"""
        gmt_now = datetime.utcnow()
        if gmt_now.weekday() == 4:  # 周五
            gmt_hour = gmt_now.hour + gmt_now.minute/60
            # GMT 10:45开始强制平仓 (北京18:45)
            return gmt_hour >= 10.75
        return False


# ==================== 日志系统 ====================
class TradeLogger:
    """交易日志系统"""

    def __init__(self):
        self.setup_logging()
        self.trade_log = []
        self.daily_stats = {
            "date": datetime.utcnow().date().isoformat(),
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0,
            "max_drawdown": 0
        }

    def setup_logging(self):
        """配置日志"""
        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger('FTMOTrader')
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        fh = RotatingFileHandler(
            'logs/trading.log',
            maxBytes=10485760,
            backupCount=5
        )
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_trade(self, trade_type, price, lot_size, sl, tp, result=None):
        """记录交易"""
        trade_record = {
            "timestamp": datetime.utcnow(),
            "type": trade_type,
            "symbol": Config.SYMBOL,
            "price": price,
            "lot_size": lot_size,
            "sl": sl,
            "tp": tp,
            "result": result
        }
        self.trade_log.append(trade_record)
        self.logger.info(f"Trade Executed: {trade_record}")

        # 更新统计
        self.daily_stats["trades"] += 1

        # 保存到文件
        date_str = datetime.utcnow().strftime("%Y%m%d")
        with open(f'logs/trades_{date_str}.json', 'w') as f:
            json.dump(self.trade_log, f, default=str, indent=2)


# ==================== 数据管理 ====================
class DataManager:
    """数据获取和管理"""

    def __init__(self):
        self.timeframes = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }

    def get_data(self, timeframe, n_bars=100):
        """获取K线数据"""
        tf = self.timeframes.get(timeframe, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, tf, 0, n_bars)

        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        return df

    def get_session_data(self, session):
        """根据时段获取不同数据"""
        if session == "asia_to_london":
            # 亚洲时段：需要更多历史数据
            df_m5 = self.get_data('M5', 200)  # 16小时数据
            df_m15 = self.get_data('M15', 80)  # 20小时数据
            df_h1 = self.get_data('H1', 24)    # 1天数据
        elif session == "london_morning":
            # 伦敦上午：中等数据
            df_m5 = self.get_data('M5', 150)
            df_m15 = self.get_data('M15', 60)
            df_h1 = self.get_data('H1', 12)
        else:  # london_open
            # 伦敦开盘：最新数据
            df_m5 = self.get_data('M5', 100)
            df_m15 = self.get_data('M15', 40)
            df_h1 = self.get_data('H1', 6)

        return df_m5, df_m15, df_h1

    def calculate_indicators(self, df):
        """计算技术指标"""
        if df is None or len(df) < 20:
            return df

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

        # 布林带
        df['bb_middle'] = df['close'].rolling(Config.BB_PERIOD).mean()
        bb_std = df['close'].rolling(Config.BB_PERIOD).std()
        df['bb_upper'] = df['bb_middle'] + (Config.BB_STD * bb_std)
        df['bb_lower'] = df['bb_middle'] - (Config.BB_STD * bb_std)

        # 支撑阻力
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 成交量加权平均价
        if 'tick_volume' in df.columns:
            df['vwap'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

        return df


# ==================== 风险管理 ====================
class RiskManager:
    """风险管理器"""

    def __init__(self, account_balance):
        self.initial_balance = account_balance
        self.daily_pnl = 0
        self.today_trades = 0
        self.max_daily_loss = account_balance * Config.DAILY_MAX_RISK
        self.ftmo_daily_limit = account_balance * 0.05  # FTMO 5%限制

        # 加载历史表现
        self.load_performance()

    def load_performance(self):
        """加载历史表现数据"""
        try:
            with open('logs/performance.json', 'r') as f:
                self.performance = json.load(f)
        except:
            self.performance = {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "best_day": 0,
                "worst_day": 0
            }

    def calculate_position_size(self, stop_loss_distance, session):
        """计算仓位大小"""
        if stop_loss_distance <= 0:
            return 0.01  # 最小手数

        # 获取当前账户净值
        account_info = mt5.account_info()
        if account_info is None:
            current_equity = self.initial_balance
        else:
            current_equity = account_info.equity

        # 计算风险金额
        risk_amount = current_equity * Config.RISK_PER_TRADE

        # 应用时段仓位乘数
        session_params = XAUUSDTimeManager.get_session_params(session)
        risk_amount *= session_params["position_multiplier"]

        # 计算每点价值（黄金：1标准手 = 10美元/点）
        pip_value = 10  # 美元

        # 计算手数
        pip_distance = stop_loss_distance / Config.POINT
        if pip_distance == 0:
            return 0.01

        position_size = risk_amount / (pip_distance * pip_value)

        # 限制最小和最大手数
        position_size = max(0.01, min(position_size, 0.3))  # 最大0.3手

        return round(position_size, 2)

    def check_daily_limit(self):
        """检查每日亏损限制"""
        account_info = mt5.account_info()
        if account_info is None:
            return True, ""

        current_equity = account_info.equity
        daily_loss = self.initial_balance - current_equity

        # 检查我们的内部限制
        if daily_loss > self.max_daily_loss:
            return False, f"达到内部亏损限制：损失${daily_loss:.2f}"

        # 检查FTMO规则（5%亏损限制）
        if daily_loss > self.ftmo_daily_limit:
            return False, f"达到FTMO每日亏损限制"

        return True, ""

    def check_trading_time(self):
        """检查交易时间"""
        if not XAUUSDTimeManager.is_trading_time():
            return False, "非交易时段"

        # 周五是否交易
        if datetime.utcnow().weekday() == 4 and not Config.TRADE_ON_FRIDAY:
            return False, "周五不交易"

        return True, ""

    def check_max_positions(self):
        """检查最大持仓数"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if positions is None:
            return True, ""

        if len(positions) >= Config.MAX_POSITIONS:
            return False, f"达到最大持仓限制：{len(positions)}个"

        return True, ""


# ==================== 交易策略 ====================
class XAUUSDStrategy:
    """黄金交易策略 - 基于GMT时间"""

    def __init__(self, data_manager, risk_manager):
        self.dm = data_manager
        self.rm = risk_manager
        self.current_session = None
        self.last_signal_time = None

    def analyze_market(self):
        """分析市场状态"""
        # 获取当前GMT时段
        current_session = XAUUSDTimeManager.get_current_session()
        self.current_session = current_session

        # 根据时段获取数据
        df_m5, df_m15, df_h1 = self.dm.get_session_data(current_session)

        if df_m5 is None or df_m15 is None or df_h1 is None:
            return {"error": "无法获取数据"}

        # 计算指标
        df_m5 = self.dm.calculate_indicators(df_m5)
        df_m15 = self.dm.calculate_indicators(df_m15)
        df_h1 = self.dm.calculate_indicators(df_h1)

        # 获取当前价格
        current_price = df_m5['close'].iloc[-1]
        atr = df_m5['atr'].iloc[-1] if not pd.isna(df_m5['atr'].iloc[-1]) else 5.0

        # 获取时段参数
        session_params = XAUUSDTimeManager.get_session_params(current_session)

        # 检查交易信号
        signal = self._check_trading_signal(df_m5, df_m15, df_h1, current_price)

        return {
            "session": current_session,
            "current_price": current_price,
            "atr": atr,
            "signal": signal,
            "session_params": session_params,
            "rsi": df_m5['rsi'].iloc[-1] if 'rsi' in df_m5.columns else 50,
            "trend": self._get_trend_direction(df_h1)
        }

    def _get_trend_direction(self, df):
        """判断趋势方向"""
        if df is None or len(df) < 5:
            return "NEUTRAL"

        price = df['close'].iloc[-1]

        # 简单趋势判断
        if len(df) >= 10:
            price_change = (price - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100

            if price_change > 0.5:
                return "UP"
            elif price_change < -0.5:
                return "DOWN"

        return "NEUTRAL"

    def _check_trading_signal(self, df_m5, df_m15, df_h1, price):
        """检查交易信号 - 基于GMT时段"""
        current_session = self.current_session

        if current_session == "asia_to_london":
            return self._asia_to_london_signal(df_m5, df_m15, price)
        elif current_session == "london_morning":
            return self._london_morning_signal(df_m5, df_m15, price)
        elif current_session == "london_open":
            return self._london_open_signal(df_m5, df_m15, price)
        else:
            return None

    def _asia_to_london_signal(self, df_m5, df_m15, price):
        """GMT 01:00-05:00 (北京09:00-13:00) - 亚洲到伦敦过渡"""
        if len(df_m15) < 32:
            return None

        # 获取隔夜区间（前8小时）
        overnight_high = df_m15['high'].iloc[-32:-1].max()  # 前8小时高点
        overnight_low = df_m15['low'].iloc[-32:-1].min()    # 前8小时低点
        overnight_range = overnight_high - overnight_low

        rsi = df_m5['rsi'].iloc[-1] if 'rsi' in df_m5.columns else 50

        # 策略1：突破隔夜区间
        if overnight_range > 8:  # 有足够波动
            if price > overnight_high and rsi < 65:
                return {"type": "BUY", "reason": "突破隔夜高点"}
            elif price < overnight_low and rsi > 35:
                return {"type": "SELL", "reason": "跌破隔夜低点"}

        # 策略2：均值回归（区间中部）
        if overnight_low < price < overnight_high:
            if rsi > 70:
                return {"type": "SELL", "reason": "亚洲时段超买回归"}
            elif rsi < 30:
                return {"type": "BUY", "reason": "亚洲时段超卖回归"}

        return None

    def _london_morning_signal(self, df_m5, df_m15, price):
        """GMT 05:00-08:00 (北京13:00-16:00) - 伦敦上午"""
        if len(df_m15) < 20:
            return None

        # 获取伦敦开盘后区间
        london_high = df_m15['high'].iloc[-20:-1].max()  # 前5小时高点
        london_low = df_m15['low'].iloc[-20:-1].min()    # 前5小时低点

        # 计算亚洲时段区间（GMT 01:00-05:00）
        asian_high = df_m15['high'].iloc[-32:-20].max() if len(df_m15) >= 32 else 0
        asian_low = df_m15['low'].iloc[-32:-20].min() if len(df_m15) >= 32 else 0

        rsi = df_m5['rsi'].iloc[-1] if 'rsi' in df_m5.columns else 50

        # 策略1：突破亚洲时段区间
        if asian_high > 0 and asian_low > 0:
            if price > asian_high and rsi < 70:
                return {"type": "BUY", "reason": "突破亚洲时段区间"}
            elif price < asian_low and rsi > 30:
                return {"type": "SELL", "reason": "跌破亚洲时段区间"}

        # 策略2：伦敦上午回调交易
        if london_high > 0 and london_low > 0:
            london_range = london_high - london_low

            if london_range > 6:  # 有足够波动
                # 在区间边缘寻找反转
                if abs(price - london_high) < 2 and rsi > 65:
                    return {"type": "SELL", "reason": "伦敦上午区间顶部反转"}
                elif abs(price - london_low) < 2 and rsi < 35:
                    return {"type": "BUY", "reason": "伦敦上午区间底部反转"}

        return None

    def _london_open_signal(self, df_m5, df_m15, price):
        """GMT 08:00-11:00 (北京16:00-19:00) - 伦敦开盘动量"""
        if len(df_m15) < 12:
            return None

        # 获取伦敦开盘后区间
        london_open_high = df_m15['high'].iloc[-12:-1].max()  # 前3小时高点
        london_open_low = df_m15['low'].iloc[-12:-1].min()    # 前3小时低点
        london_range = london_open_high - london_open_low

        # 需要足够波动性
        if london_range < 5:
            return None

        rsi = df_m5['rsi'].iloc[-1] if 'rsi' in df_m5.columns else 50

        # 动量突破策略
        if price > london_open_high and rsi < 75:
            return {"type": "BUY", "reason": "伦敦开盘动量突破"}
        elif price < london_open_low and rsi > 25:
            return {"type": "SELL", "reason": "伦敦开盘动量跌破"}

        return None


# ==================== 交易执行 ====================
class TradeExecutor:
    """交易执行器"""

    def __init__(self, strategy, risk_manager, logger):
        self.strategy = strategy
        self.rm = risk_manager
        self.logger = logger
        self.pending_orders = []

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
                self.logger.logger.info(f"交易被阻止: {check_msg}")
                return False, check_msg

        if signal is None:
            return False, "无交易信号"

        current_price = market_data['current_price']
        atr = market_data['atr']
        session = market_data['session']
        session_params = market_data['session_params']

        # 根据时段调整止损
        max_sl = session_params["max_sl"]
        atr_sl = atr * Config.STOP_LOSS_ATR_MULTIPLIER
        stop_loss_distance = min(atr_sl, max_sl)

        # 计算止损止盈
        if signal['type'] == "BUY":
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + (stop_loss_distance * Config.TAKE_PROFIT_RATIO)
        else:  # SELL
            stop_loss = current_price + stop_loss_distance
            take_profit = current_price - (stop_loss_distance * Config.TAKE_PROFIT_RATIO)

        # 计算仓位大小（考虑时段仓位乘数）
        lot_size = self.rm.calculate_position_size(stop_loss_distance, session)

        # 准备订单请求
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
            "comment": f"{session}:{signal['reason']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 发送订单
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.logger.error(f"订单失败: {result.comment}")
            return False, result.comment

        # 记录交易
        self.logger.log_trade(
            signal['type'],
            current_price,
            lot_size,
            stop_loss,
            take_profit
        )

        self.logger.logger.info(f"交易成功: {signal['type']} {Config.SYMBOL} "
                              f"手数:{lot_size} 价格:{current_price}")

        return True, "交易成功"

    def manage_positions(self):
        """管理现有仓位"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)

        if positions is None or len(positions) == 0:
            return

        for position in positions:
            self._check_trailing_stop(position)
            self._check_time_exit(position)

    def _check_trailing_stop(self, position):
        """检查追踪止损"""
        current_price = mt5.symbol_info_tick(Config.SYMBOL).ask

        if position.type == mt5.ORDER_TYPE_BUY:
            profit = current_price - position.price_open

            # 激活追踪止损
            if profit >= Config.TRAILING_STOP_ACTIVATE:
                new_sl = current_price - Config.TRAILING_STOP_DISTANCE
                if new_sl > position.sl:
                    self._modify_position(position.ticket, new_sl, position.tp)

        elif position.type == mt5.ORDER_TYPE_SELL:
            profit = position.price_open - current_price

            if profit >= Config.TRAILING_STOP_ACTIVATE:
                new_sl = current_price + Config.TRAILING_STOP_DISTANCE
                if new_sl < position.sl:
                    self._modify_position(position.ticket, new_sl, position.tp)

    def _check_time_exit(self, position):
        """检查时间退出"""
        # 交易结束前平仓
        if XAUUSDTimeManager.should_close_before_end():
            self._close_position(position, reason="交易时段结束前平仓")
            return

        # 周五强制平仓
        if XAUUSDTimeManager.is_friday_close_time():
            self._close_position(position, reason="周五收盘前强制平仓")
            return

        # 持仓超过4小时强制平仓
        position_time = pd.to_datetime(position.time, unit='s')
        holding_hours = (datetime.utcnow() - position_time).total_seconds() / 3600

        if holding_hours > Config.MAX_HOLDING_HOURS:
            self._close_position(position, reason="持仓超时")

    def _modify_position(self, ticket, new_sl, new_tp):
        """修改订单"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.logger.info(f"修改订单 {ticket}: SL={new_sl}, TP={new_tp}")

    def _close_position(self, position, reason="收盘平仓"):
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
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.logger.info(f"平仓 {position.ticket}: {reason}")


# ==================== 主交易系统 ====================
class FTMOTradingSystem:
    """主交易系统"""

    def __init__(self):
        self.logger = TradeLogger()
        self.dm = DataManager()
        self.rm = RiskManager(Config.ACCOUNT_BALANCE)
        self.strategy = XAUUSDStrategy(self.dm, self.rm)
        self.executor = TradeExecutor(self.strategy, self.rm, self.logger)

        self.is_running = False
        self.last_check_time = None

        self.logger.logger.info("=" * 50)
        self.logger.logger.info("FTMO XAUUSD交易系统启动 - GMT时间版本")
        self.logger.logger.info(f"品种: {Config.SYMBOL}")

        # 显示时间转换信息
        gmt_start = Config.GMT_TRADING_HOURS["start"]
        gmt_end = Config.GMT_TRADING_HOURS["end"]
        self.logger.logger.info(f"交易时间: GMT {gmt_start}:00-{gmt_end}:00")
        self.logger.logger.info(f"对应北京时间: {gmt_start+8}:00-{gmt_end+8}:00")
        self.logger.logger.info("=" * 50)

    def initialize_mt5(self):
        """初始化MT5连接"""
        if not mt5.initialize():
            self.logger.logger.error("MT5初始化失败")
            return False

        # 启用自动交易模式
        mt5.terminal_info()

        return True

    def run(self):
        """运行主循环"""
        self.is_running = True

        while self.is_running:
            try:
                # 检查连接
                if not mt5.terminal_info():
                    self.logger.logger.warning("MT5连接断开，尝试重连...")
                    if not self.initialize_mt5():
                        time.sleep(60)
                        continue

                # 管理现有仓位
                self.executor.manage_positions()

                # 每30秒检查一次交易信号
                current_time = time.time()
                if self.last_check_time is None or current_time - self.last_check_time > 30:
                    if XAUUSDTimeManager.is_trading_time():
                        self._check_trading_signal()
                    self.last_check_time = current_time

                # 显示状态
                self._display_status()

                time.sleep(5)  # 主循环间隔

            except KeyboardInterrupt:
                self.logger.logger.info("收到停止信号，关闭系统...")
                self.stop()
                break
            except Exception as e:
                self.logger.logger.error(f"运行错误: {str(e)}")
                time.sleep(30)

    def _check_trading_signal(self):
        """检查交易信号"""
        # 获取市场分析
        market_data = self.strategy.analyze_market()

        if "error" in market_data:
            self.logger.logger.error(f"分析错误: {market_data['error']}")
            return

        # 获取交易信号
        signal = market_data.get("signal")

        if signal:
            self.logger.logger.info(f"发现信号: {signal}")

            # 执行交易
            success, message = self.executor.execute_trade(signal, market_data)

            if success:
                self.logger.logger.info(f"交易执行成功: {message}")
            else:
                self.logger.logger.info(f"交易未执行: {message}")

    def _display_status(self):
        """显示当前状态"""
        try:
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is None:
                print("无法获取账户信息")
                return

            equity = account_info.equity
            balance = account_info.balance

            # 获取持仓
            positions = mt5.positions_get(symbol=Config.SYMBOL)
            position_count = len(positions) if positions else 0

            # 获取时间信息
            time_info = XAUUSDTimeManager.format_time_display()

            # 计算当日盈亏
            daily_pnl = equity - balance

            # 清屏并显示状态
            os.system('cls' if os.name == 'nt' else 'clear')

            print("=" * 70)
            print("FTMO XAUUSD 日内交易系统 - GMT时间版本")
            print("=" * 70)
            print(f"GMT时间: {time_info['gmt_time']} | 北京时间: {time_info['beijing_time']}")
            print(f"交易时段: {time_info['session']} | 波动性: {time_info['volatility']}")
            print(f"当前策略: {time_info['strategy']}")
            print("-" * 70)
            print(f"账户净值: ${equity:.2f} | 余额: ${balance:.2f}")
            print(f"当日盈亏: ${daily_pnl:.2f} ({daily_pnl/balance*100:.2f}%)")
            print(f"持仓数量: {position_count}")
            print(f"交易状态: {'运行中' if self.is_running else '已停止'}")
            print("-" * 70)

            # 显示持仓详情
            if position_count > 0:
                print("当前持仓:")
                for pos in positions:
                    profit = pos.profit
                    open_time = pd.to_datetime(pos.time, unit='s')
                    holding_hours = (datetime.utcnow() - open_time).total_seconds() / 3600

                    print(f"  #{pos.ticket}: {pos.type_desc()} {pos.volume}手 "
                          f"@ {pos.price_open:.2f} | 盈亏: ${profit:.2f} "
                          f"| 持仓: {holding_hours:.1f}小时")

            # 显示FTMO规则状态
            ftmo_daily_limit = Config.ACCOUNT_BALANCE * 0.05  # 5%
            ftmo_used = abs(min(daily_pnl, 0))
            ftmo_remaining = ftmo_daily_limit - ftmo_used

            print("-" * 70)
            print(f"FTMO规则监控:")
            print(f"  日亏损限额: ${ftmo_daily_limit:.2f}")
            print(f"  已用额度: ${ftmo_used:.2f}")
            print(f"  剩余额度: ${ftmo_remaining:.2f}")
            print(f"  状态: {'✓ 正常' if ftmo_remaining > 0 else '✗ 超过限制'}")

            # 显示时段参数
            session_params = XAUUSDTimeManager.get_session_params(time_info['session'])
            print(f"  时段仓位乘数: {session_params['position_multiplier']}")
            print(f"  最大止损: ${session_params['max_sl']}")

            # 显示下一个检查时间
            if self.last_check_time:
                next_check = self.last_check_time + 30
                remaining = max(0, next_check - time.time())
                print(f"\n下一个信号检查: {int(remaining)}秒后")

            print("=" * 70)
            print("按 Ctrl+C 停止系统")

        except Exception as e:
            self.logger.logger.error(f"状态显示错误: {str(e)}")

    def stop(self):
        """停止系统"""
        self.is_running = False

        # 平仓所有持仓
        self._close_all_positions()

        # 断开MT5连接
        mt5.shutdown()

        self.logger.logger.info("交易系统已停止")

    def _close_all_positions(self):
        """平仓所有持仓"""
        positions = mt5.positions_get()

        if positions:
            self.logger.logger.info(f"开始平仓所有持仓，共{len(positions)}个")

            for position in positions:
                self.executor._close_position(position, reason="系统关闭")

            self.logger.logger.info("所有持仓已平仓")


# ==================== 启动脚本 ====================
if __name__ == "__main__":
    # 创建交易系统
    system = FTMOTradingSystem()

    # 初始化MT5连接
    if system.initialize_mt5():
        try:
            # 运行系统
            system.run()
        except Exception as e:
            system.logger.logger.error(f"系统运行异常: {str(e)}")
        finally:
            system.stop()
    else:
        print("MT5初始化失败，请检查配置")
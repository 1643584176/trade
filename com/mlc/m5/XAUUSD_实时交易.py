import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys
import os
import time
from datetime import datetime, timedelta, timezone
import logging
from threading import Thread, Event
import warnings

warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))

# 配置文件抽离（模拟config模块）
CONFIG = {
    "SYMBOL": "XAUUSD",
    "LOT_SIZE": 0.2,
    "MAGIC_NUMBER": 10000005,
    "MODEL_WEIGHTS": {"m1": 0.15, "m5": 0.55, "m15": 0.30},
    "FTMO_RULES": {
        "MAX_DRAWDOWN": 0.045,
        "PROFIT_TARGET": 0.10,
        "MIN_BALANCE": 99020,
        "INITIAL_BALANCE": 100000
    },
    "HISTORY_BARS": {
        "m1": 50,
        "m5": 120,
        "m15": 200
    },
    "ATR_MULTIPLIERS": {
        "stop_loss": 2.0,
        "take_profit": 3.0,  # 风险回报比1.5
        "vol_high": 1.2,  # 高波动系数
        "vol_low": 0.8  # 低波动系数
    },
    "SIGNAL_THRESHOLD": {
        "base": 0.7,
        "min": 0.6,
        "max": 0.8
    },
    "TRADING_CYCLE": {
        "m1": 60,
        "m5": 300,
        "m15": 900
    },
    "LOG_LEVEL": "INFO",
    "MAX_RETRIES": 3,
    "RETRY_INTERVAL": 1
}

try:
    import m5_feature_engineering

    M5FeatureEngineer = m5_feature_engineering.M5FeatureEngineer
except ImportError:
    # 兜底实现基础特征工程
    class M5FeatureEngineer:
        def add_core_features(self, df):
            df['volatility_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['hour_of_day'] = df.index.hour
            df['is_peak_hour'] = df['hour_of_day'].isin([8, 9, 10, 14, 15, 16, 20, 21, 22]).astype(int)
            return df

        def add_enhanced_features(self, df):
            return df

# 配置日志
# 由于项目中多处配置日志，使用更可靠的方式确保日志文件被创建
logger = logging.getLogger('xauusd_trader')  # 使用特定的logger名称
logger.setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))

# 清除已有的处理器，避免重复日志
if logger.hasHandlers():
    logger.handlers.clear()

# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建文件处理器
file_handler = logging.FileHandler('xauusd_multi_period_trading.log', encoding='utf-8')
file_handler.setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))
file_handler.setFormatter(formatter)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 禁止向上级logger传播，避免重复输出
logger.propagate = False


class MultiPeriodRealTimeTrader:
    def __init__(self, m1_model_path="xauusd_m1_model.json",
                 m5_model_path="xauusd_m5_model.json",
                 m15_model_path="xauusd_m15_model.json"):
        """
        初始化多周期实时交易器
        """
        # 基础配置
        self.SYMBOL = CONFIG["SYMBOL"]
        self.M1_TIMEFRAME = mt5.TIMEFRAME_M1
        self.M5_TIMEFRAME = mt5.TIMEFRAME_M5
        self.M15_TIMEFRAME = mt5.TIMEFRAME_M15
        self.M1_MODEL_PATH = m1_model_path
        self.M5_MODEL_PATH = m5_model_path
        self.M15_MODEL_PATH = m15_model_path
        self.LOT_SIZE = CONFIG["LOT_SIZE"]
        self.MAGIC_NUMBER = CONFIG["MAGIC_NUMBER"]

        # 历史K线配置
        self.HISTORY_M1_BARS = CONFIG["HISTORY_BARS"]["m1"]
        self.HISTORY_M5_BARS = CONFIG["HISTORY_BARS"]["m5"]
        self.HISTORY_M15_BARS = CONFIG["HISTORY_BARS"]["m15"]

        # 模型权重
        self.MODEL_WEIGHTS = CONFIG["MODEL_WEIGHTS"]

        # FTMO规则
        self.FTMO_MAX_DRAWDOWN = CONFIG["FTMO_RULES"]["MAX_DRAWDOWN"]
        self.FTMO_PROFIT_TARGET = CONFIG["FTMO_RULES"]["PROFIT_TARGET"]
        self.FTMO_MIN_BALANCE = CONFIG["FTMO_RULES"]["MIN_BALANCE"]
        self.INITIAL_BALANCE = CONFIG["FTMO_RULES"]["INITIAL_BALANCE"]

        # ATR乘数配置
        self.ATR_STOP_LOSS = CONFIG["ATR_MULTIPLIERS"]["stop_loss"]
        self.ATR_TAKE_PROFIT = CONFIG["ATR_MULTIPLIERS"]["take_profit"]
        self.VOL_HIGH_COEFF = CONFIG["ATR_MULTIPLIERS"]["vol_high"]
        self.VOL_LOW_COEFF = CONFIG["ATR_MULTIPLIERS"]["vol_low"]

        # 信号阈值配置
        self.BASE_THRESHOLD = CONFIG["SIGNAL_THRESHOLD"]["base"]
        self.MIN_THRESHOLD = CONFIG["SIGNAL_THRESHOLD"]["min"]
        self.MAX_THRESHOLD = CONFIG["SIGNAL_THRESHOLD"]["max"]

        # 重试配置
        self.MAX_RETRIES = CONFIG["MAX_RETRIES"]
        self.RETRY_INTERVAL = CONFIG["RETRY_INTERVAL"]

        # 交易状态
        self.current_position = None
        self.is_running = False
        self.stop_event = Event()

        # 特征工程实例
        self.feature_engineer = M5FeatureEngineer()

        # 模型自检特征
        self.prediction_history = []
        self.max_history_length = 20
        self.daily_trades = []
        self.daily_start_balance = None

        # 特征配置（精简核心特征，确保与模型训练时一致）
        self.FEATURE_CONFIG = {
            'm1': [
                'open', 'high', 'low', 'close', 'tick_volume',
                'rsi_7',
                'ma3', 'ma7',
                'atr_7',
                'volatility_pct',
                'hour_of_day', 'is_peak_hour',
                'hammer', 'shooting_star', 'engulfing',
                'rsi_14', 'macd', 'macd_hist',
                'bollinger_position',
                'ma5', 'ma10', 'ma20', 'ma10_direction', 'ma20_direction',
                'tick_vol_zscore',
                'up_down_count_10',
                'hl_spread_zscore',
                'volatility_intensity',
                'ma5_deviation',
                'volume_impulse',
                'price_direction_consistency',
                'dynamic_activity',
                'high_activity',
                'up_momentum_3',
                'down_momentum_3',
                'down_volume_ratio',
                'momentum_3',
                'momentum_5',
                'volume_price_divergence',
                'consecutive_up',
                'consecutive_down',
                'volume_up_ratio',
                'up_momentum_5',
                'volume_up_ratio_enhanced',
                'activity_trend_up',
                'ma5_deviation_up',
                'down_momentum_5',
                'down_volume_impulse',
                'high_activity_up_weight',
                'activity_trend',
                'up_down_activity_diff',
                'activity_trend_down',
                'ma5_deviation_down'
            ],
            'm5': [
                'open', 'high', 'low', 'close', 'tick_volume',
                'price_position', 'volatility_pct',
                'm15_trend', 'm30_support', 'm30_resistance',
                'volatility_change', 'tick_density',
                'hammer', 'shooting_star', 'engulfing',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_position',
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                'rsi_price_consistency',
                'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                'trend_consistency',
                'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                'volatility_regime', 'vol_cluster',
                'm15_trend_ma_consistency',
                'm5_m1_volume_correlation',
                'trend_strength_m5_m15',
                'cycle_alignment_score',
                'm5_m15_volume_correlation',
                'volatility_diff_m5_m1',
                'adx',
                'ma5_ma20_alignment',
                'momentum_3',
                'momentum_5',
                'volume_price_divergence',
                'consecutive_up',
                'consecutive_down',
                'body_strength',
                'upper_shadow',
                'lower_shadow',
                'price_position_5',
                'dynamic_activity',
                'activity_level',
                'volume_up_ratio',
                'atr_down_prob',
                'atr_14',
                'hl_ratio',
            ],
            'm15': [
                'open', 'close', 'tick_volume',
                'rsi_21',
                'ma21',
                'ma21_direction',
                'atr_21',
                'trend_strength',
                'volatility_pct',
                'm60_trend_consistency',
                'hammer', 'shooting_star', 'engulfing',
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_position',
                'ma5', 'ma5_direction', 'ma20_direction',
                'adx',
                'ma_trend_alignment',
                'trend_duration',
                'dynamic_activity',
                'activity_level',
                'consecutive_up_momentum',
                'up_prob_when_ma21_up',
                'up_prob_when_atr_contraction',
                'dynamic_activity_up_mean',
                'up_after_high_volatility',
                'consecutive_down_momentum',
                'atr_down_prob',
                'high_activity_up_weight',
            ]
        }

        # 初始化MT5连接
        self.init_mt5_connection()

        # 加载模型和标准化器
        self.load_models()
        self.load_scalers()
        self.load_label_mapping()

        # 检查现有持仓
        self.check_existing_positions()

        # 初始化当日余额
        self.update_daily_balance()

        logger.info(f"✅ MT5连接成功")
        logger.info(f"📈 开始多周期实时交易 {self.SYMBOL}，手数: {self.LOT_SIZE}")
        logger.info(
            f"⚖️ 模型权重 - M1: {self.MODEL_WEIGHTS['m1']:.2f}, M5: {self.MODEL_WEIGHTS['m5']:.2f}, M15: {self.MODEL_WEIGHTS['m15']:.2f}")

    def init_mt5_connection(self):
        """初始化MT5连接（带重试）"""
        for retry in range(self.MAX_RETRIES):
            if mt5.initialize():
                # 确保交易品种被选中
                if mt5.symbol_select(self.SYMBOL, True):
                    return
                else:
                    logger.error(f"❌ 无法选择交易品种 {self.SYMBOL}")
            else:
                logger.error(f"❌ MT5初始化失败（重试{retry + 1}/{self.MAX_RETRIES}）: {mt5.last_error()}")

            if retry < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_INTERVAL)

        raise Exception(f"❌ MT5连接失败，已重试{self.MAX_RETRIES}次")

    def load_models(self):
        """加载所有模型（带重试）"""
        models = {
            'm1': (self.M1_MODEL_PATH, 'M1'),
            'm5': (self.M5_MODEL_PATH, 'M5'),
            'm15': (self.M15_MODEL_PATH, 'M15')
        }

        self.models = {}
        for key, (path, name) in models.items():
            for retry in range(self.MAX_RETRIES):
                try:
                    model = xgb.Booster()
                    model.load_model(path)
                    self.models[key] = model
                    logger.debug(f"✅ {name}模型已从 {path} 加载")
                    break
                except Exception as e:
                    logger.error(f"❌ 加载{name}模型失败（重试{retry + 1}/{self.MAX_RETRIES}）: {e}")
                    if retry == self.MAX_RETRIES - 1:
                        raise e
                    time.sleep(self.RETRY_INTERVAL)

    def load_scalers(self):
        """加载特征标准化器"""
        self.scalers = {}
        scaler_paths = {
            'm1': "m1_scaler.pkl",
            'm5': "m5_scaler.pkl",
            'm15': "m15_scaler.pkl"
        }

        for key, path in scaler_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.scalers[key] = pickle.load(f)
                    logger.debug(f"✅ {key}标准化器已加载")
                else:
                    self.scalers[key] = None
                    logger.warning(f"⚠️ {key}标准化器文件不存在: {path}")
            except Exception as e:
                logger.error(f"❌ 加载{key}标准化器失败: {e}")
                self.scalers[key] = None

    def load_label_mapping(self):
        """加载标签映射"""
        self.label_mappings = {}
        mapping_paths = {
            'm1': "m1_label_mapping.pkl",
            'm5': "m5_label_mapping.pkl",
            'm15': "m15_label_mapping.pkl"
        }

        for key, path in mapping_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.label_mappings[key] = pickle.load(f)
                    logger.debug(f"✅ {key}标签映射已加载")
                else:
                    # 默认映射
                    self.label_mappings[key] = {-1: 0, 0: 1, 1: 2}
                    logger.warning(f"⚠️ 使用{key}默认标签映射")
            except Exception as e:
                logger.error(f"❌ 加载{key}标签映射失败: {e}")
                self.label_mappings[key] = {-1: 0, 0: 1, 1: 2}

    def calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(self, high, low, close, window=14):
        """计算ATR"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()

    def calculate_direction(self, series):
        """计算方向特征"""
        return (series - series.shift(1)) / (series.shift(1) + 1e-8)

    def calculate_adx(self, high, low, close, window=14):
        """计算ADX指标"""
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr_rolling = tr.rolling(window=window).mean()
        
        # 计算+DM和-DM
        hd = high - high.shift()
        ld = low.shift() - low
        
        pdm = np.where((hd > 0) & (hd > ld), hd, 0)
        ndm = np.where((ld > 0) & (ld > hd), ld, 0)
        
        pdm = pd.Series(pdm, index=high.index)
        ndm = pd.Series(ndm, index=high.index)
        
        pdm_rolling = pdm.rolling(window=window).mean()
        ndm_rolling = ndm.rolling(window=window).mean()
        
        # 计算+DI和-DI
        pdi = (pdm_rolling / tr_rolling) * 100
        ndi = (ndm_rolling / tr_rolling) * 100
        
        # 计算DX
        dx = (abs(pdi - ndi) / abs(pdi + ndi)) * 100
        dx = dx.replace([np.inf, -np.inf], np.nan)
        
        # 计算ADX
        adx = dx.rolling(window=window).mean()
        adx = adx.fillna(method='bfill')
        
        return adx

    def calculate_dynamic_activity(self, df):
        """为M5数据计算动态活跃度特征"""
        # 计算短期波动率（最近3根M5波动率）- 平滑短期波动
        df['volatility_short'] = df['close'].pct_change().rolling(window=3).std()  # 3根M5波动率
        
        # 计算长期波动率（过去24小时平均波动率）
        df['volatility_long_avg'] = df['volatility_short'].rolling(window=288, min_periods=24).mean()  # 24小时=288个M5周期
        
        # 计算动态活跃度（短期波动率/长期平均波动率）
        df['dynamic_activity'] = df['volatility_short'] / (df['volatility_long_avg'] + 1e-8)
        
        return df['dynamic_activity']

    def calculate_dynamic_activity_m15(self, df):
        """为M15数据计算动态活跃度特征"""
        # 计算短期波动率（最近3根M15K线波动率）- 优化活跃度计算
        df['volatility_short'] = df['close'].pct_change().rolling(window=3).std()  # 3根M15波动率
        
        # 计算长期波动率（过去24小时平均波动率）
        df['volatility_long_avg'] = df['volatility_short'].rolling(window=96, min_periods=24).mean()  # 24小时=96个M15周期
        
        # 计算动态活跃度（短期波动率/长期平均波动率）
        df['dynamic_activity_raw'] = df['volatility_short'] / (df['volatility_long_avg'] + 1e-8)
        
        # 重构dynamic_activity计算逻辑：从"单根M15活跃度"改为"最近3根M15的平均活跃度"
        df['dynamic_activity_avg'] = df['dynamic_activity_raw'].rolling(window=3).mean()  # 3根M15的平均活跃度
        
        # 计算活跃度环比变化
        df['dynamic_activity_change'] = df['dynamic_activity_raw'].pct_change()
        
        # 综合平均活跃度和环比变化作为最终活跃度
        df['dynamic_activity'] = df['dynamic_activity_avg'] + 0.3 * df['dynamic_activity_change']
        
        # 创建活跃度分类（高/中/低活跃度）
        df['activity_level'] = 1  # 默认为中等活跃度
        df.loc[df['dynamic_activity'] > 1.2, 'activity_level'] = 2  # 高活跃度
        df.loc[df['dynamic_activity'] < 0.8, 'activity_level'] = 0  # 低活跃度
        
        return df

    def add_micro_features(self, df):
        """为M1数据添加完整的微观交易特征"""
        # Tick成交量脉冲特征
        df['tick_vol_zscore'] = (df['tick_volume'] - df['tick_volume'].rolling(window=10).mean()) / df['tick_volume'].rolling(window=10).std()
        df['tick_vol_zscore'] = df['tick_vol_zscore'].fillna(0)
        
        # 成交量脉冲特征（当前成交量 / 前3根均值，更适合M1超短期周期）
        df['volume_impulse'] = df['tick_volume'] / df['tick_volume'].rolling(window=3).mean()
        df['volume_impulse'] = df['volume_impulse'].fillna(1.0)  # 用1.0填充初始值
        
        # 涨跌延续性特征（连续2根M1的涨跌幅方向是否一致）
        df['price_change'] = df['close'].pct_change()
        df['price_direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))
        df['price_direction_consistency'] = (df['price_direction'] == df['price_direction'].shift(1)).astype(int)
        
        # 1分钟内涨跌次数特征（通过价格变化方向统计）
        df['price_change'] = df['close'].diff()
        df['price_direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))
        df['up_down_count_10'] = df['price_direction'].rolling(window=10).sum().abs()
        
        # 盘口买卖价差特征（通过高低价差异近似）
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        df['hl_spread_zscore'] = (df['high_low_spread'] - df['high_low_spread'].rolling(window=20).mean()) / df['high_low_spread'].rolling(window=20).std()
        df['hl_spread_zscore'] = df['hl_spread_zscore'].fillna(0)
        
        # 价格波动强度特征
        df['volatility_intensity'] = abs(df['close'] - df['open']) / df['close']
        
        # 短期趋势强度（基于移动平均偏离度）
        df['ma5_deviation'] = abs(df['close'] - df['close'].rolling(window=5).mean()) / df['close']
        df['ma5_trend_strength'] = (df['close'] - df['close'].rolling(window=5).mean()) / df['close']
        
        # 清理可能的无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 更新dynamic_activity特征：保留 "最近 5 根 M1 平均活跃度"，新增 "涨 / 跌活跃度差异" 特征
        df['volatility_5m'] = df['close'].pct_change().rolling(window=5).std()  # 5分钟波动率
        df['volatility_60m_avg'] = df['volatility_5m'].rolling(window=12).mean()  # 60分钟（12个5分钟）平均波动率
        df['dynamic_activity_raw'] = df['volatility_5m'] / (df['volatility_60m_avg'] + 1e-8)  # 防止除零
        df['dynamic_activity'] = df['dynamic_activity_raw'].rolling(window=5).mean()  # 最近5根M1平均活跃度
        
        # 新增"涨/跌活跃度差异"特征
        df['price_change_direction'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
        df['up_activity'] = df['dynamic_activity_raw'] * (df['price_change_direction'] == 1).astype(int)  # 上涨时活跃度
        df['down_activity'] = df['dynamic_activity_raw'] * (df['price_change_direction'] == -1).astype(int)  # 下跌时活跃度
        df['up_down_activity_diff'] = df['up_activity'].rolling(window=5).mean() - df['down_activity'].rolling(window=5).mean()  # 涨跌活跃度差异
        
        df['high_activity'] = (df['dynamic_activity'] > 1.2).astype(int)  # 高活跃度标记
        
        # 对高活跃时段的涨类样本额外加权（1.1），帮助模型识别高波动下的上涨信号
        df['high_activity_up_weight'] = df['high_activity'] * (df['price_change'] > 0).astype(int) * 1.1
        
        # 新增涨类动能特征（涨类专属特征补充）
        # 连续3根M1涨跌幅之和（仅计算上涨）
        df['price_change'] = df['close'].pct_change()
        df['up_momentum_3'] = df['price_change'].rolling(window=3).apply(lambda x: sum([i for i in x if i > 0]), raw=True)  # 仅计算上涨部分
        df['up_momentum_3'] = df['up_momentum_3'].fillna(0)
        
        # volume_up_ratio 强化版
        df['volume_up_ratio_enhanced'] = df['tick_volume'] / df['tick_volume'].rolling(window=10).mean()  # 成交量相对均值的比值
        df['volume_up_impulse_enhanced'] = df['volume_up_ratio_enhanced'] * (df['price_change'] > 0).astype(int)  # 放量上涨占比
        
        # activity_trend 上涨趋势
        df['activity_trend_up'] = df['dynamic_activity'] - df['dynamic_activity'].shift(5)  # 当前活跃度 - 前5根平均活跃度
        df['activity_trend_up'] = df['activity_trend_up'].fillna(0)
        
        # ma5_deviation 向上偏离
        df['ma5_deviation_up'] = np.where(df['ma5_trend_strength'] > 0, df['ma5_deviation'], 0)  # 仅当趋势向上时考虑偏离度
        
        # 强化跌类动能特征：连续3根M1下跌动能 + 跌时成交量占比
        df['down_momentum_3'] = df['price_change'].rolling(window=3).apply(lambda x: abs(sum([i for i in x if i < 0])), raw=True)  # 仅计算下跌部分
        df['down_momentum_3'] = df['down_momentum_3'].fillna(0)
        
        # 跌时成交量占比
        df['price_direction'] = np.where(df['price_change'] < 0, 1, 0)  # 价格下跌标记
        df['down_volume_ratio'] = df['tick_volume'] * df['price_direction']  # 跌时成交量
        df['down_volume_ratio'] = df['down_volume_ratio'].rolling(window=10).sum() / df['tick_volume'].rolling(window=10).sum()  # 跌时成交量占比
        df['down_volume_ratio'] = df['down_volume_ratio'].fillna(0)
        
        # 新增涨类专属特征：volume_impulse 放量上涨占比
        df['volume_up_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=10).mean()  # 成交量相对均值的比值
        df['up_volume_impulse'] = df['volume_up_ratio'] * (df['price_change'] > 0).astype(int)  # 放量上涨占比
        
        # 新增涨类专属特征：momentum_5 上涨强度
        df['up_momentum_5'] = df['price_change'].rolling(window=5).apply(lambda x: sum([i for i in x if i > 0]), raw=True)  # 5根K线仅计算上涨部分
        df['up_momentum_5'] = df['up_momentum_5'].fillna(0)
        
        # 新增跌类专属特征：down_momentum_5
        df['down_momentum_5'] = df['price_change'].rolling(window=5).apply(lambda x: abs(sum([i for i in x if i < 0])), raw=True)  # 5根K线仅计算下跌部分
        df['down_momentum_5'] = df['down_momentum_5'].fillna(0)
        
        # 新增跌类专属特征：volume_down_ratio
        df['volume_down_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(window=10).mean()  # 成交量相对均值的比值
        df['down_volume_impulse'] = df['volume_down_ratio'] * (df['price_change'] < 0).astype(int)  # 放量下跌占比
        
        # dynamic_activity 特征优化：新增"活跃度趋势"特征
        df['activity_trend'] = df['dynamic_activity'] - df['dynamic_activity'].shift(5)  # 当前活跃度 - 前5根平均活跃度
        df['activity_trend'] = df['activity_trend'].fillna(0)
        
        # 新增跌类专属特征：activity_trend 下跌趋势
        df['activity_trend_down'] = np.where(df['activity_trend'] < 0, abs(df['activity_trend']), 0)  # 仅当活跃度趋势向下时考虑
        
        # 新增跌类专属特征：ma5_deviation 向下偏离
        df['ma5_deviation_down'] = np.where(df['ma5_trend_strength'] < 0, df['ma5_deviation'], 0)  # 仅当趋势向下时考虑偏离度
        
        return df

    def add_trend_features(self, df):
        """添加M15趋势特征（简化版）"""
        try:
            # 计算ma21_direction特征，如果不存在ma21则先计算
            if 'ma21' in df.columns:
                ma21_diff = df['close'] - df['ma21']
            else:
                ma21 = df['close'].rolling(21).mean()
                ma21_diff = df['close'] - ma21
            
            # 使用shift方法替代rolling.apply，避免潜在错误
            df['ma21_direction'] = np.where(
                ma21_diff > ma21_diff.shift(1), 1,
                np.where(ma21_diff < ma21_diff.shift(1), -1, 0)
            )
            
            # 计算趋势持续时间
            df['trend_duration'] = df['ma21_direction'].rolling(10).sum().abs()
            
            # 计算连续涨跌动量
            price_diff = df['close'].diff()
            df['consecutive_up_momentum'] = np.where(price_diff > 0, price_diff, 0)
            df['consecutive_down_momentum'] = np.where(price_diff < 0, -price_diff, 0)
            
            # 用0填充NaN值
            df['ma21_direction'] = df['ma21_direction'].fillna(0).astype(int)
            df['trend_duration'] = df['trend_duration'].fillna(0)
            df['consecutive_up_momentum'] = df['consecutive_up_momentum'].fillna(0)
            df['consecutive_down_momentum'] = df['consecutive_down_momentum'].fillna(0)
        except Exception as e:
            logger.warning(f"⚠️ 添加趋势特征失败: {e}，使用默认值")
            # 设置默认值
            df['ma21_direction'] = 0
            df['trend_duration'] = 0
            df['consecutive_up_momentum'] = 0
            df['consecutive_down_momentum'] = 0
        
        return df

    def update_daily_balance(self):
        """更新当日初始余额"""
        try:
            account_info = mt5.account_info()
            # 获取XAUUSD市场数据时间，严格遵守时间源使用规范
            current_tick = mt5.symbol_info_tick(self.SYMBOL)
            if current_tick:
                current_market_time = datetime.fromtimestamp(current_tick.time)
                if account_info and (self.daily_start_balance is None or current_market_time.hour == 0):
                    self.daily_start_balance = account_info.balance
                    logger.info(f"📅 当日初始余额更新为: {self.daily_start_balance}")
            else:
                logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                raise Exception("无法获取XAUUSD市场数据时间")
        except Exception as e:
            logger.error(f"❌ 更新当日余额失败: {e}")

    def check_existing_positions(self):
        """检查现有持仓（实时同步）"""
        try:
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions:
                pos = positions[0]
                direction = "做多" if pos.type == mt5.POSITION_TYPE_BUY else "做空"
                self.current_position = {
                    'ticket': pos.ticket,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'time': pos.time,
                    'direction': direction,
                    'profit': pos.profit
                }
                logger.info(f"📌 检测到现有持仓: {direction}, 手数: {pos.volume}, 盈亏: {pos.profit:.2f}")
            else:
                self.current_position = None
                logger.info("📌 未检测到现有持仓")
        except Exception as e:
            logger.error(f"❌ 检查现有持仓失败: {e}")
            self.current_position = None

    def get_current_market_data(self, timeframe, bars_count: int):
        """获取指定时间周期的市场数据（带重试和异常值处理）"""
        for retry in range(self.MAX_RETRIES):
            try:
                # 记录尝试获取的数据周期
                timeframe_name = {mt5.TIMEFRAME_M1: 'M1', mt5.TIMEFRAME_M5: 'M5', mt5.TIMEFRAME_M15: 'M15'}.get(timeframe, str(timeframe))
                logger.debug(f"📊 开始获取{timeframe_name}数据，K线索取数量: {bars_count + 1}")
                
                # 从MT5获取实时数据，获取额外一根K线以确保我们有足够数据
                rates = mt5.copy_rates_from_pos(self.SYMBOL, timeframe, 0, bars_count + 1)

                if rates is None or len(rates) == 0:
                    logger.error(
                        f"❌ 获取{timeframe_name}({timeframe})历史数据失败（重试{retry + 1}/{self.MAX_RETRIES}）: {mt5.last_error()}")
                    time.sleep(self.RETRY_INTERVAL)
                    continue
                
                logger.debug(f"📊 成功获取{timeframe_name}原始数据，共{len(rates)}根K线")

                # 转换为DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # 移除最后一根K线，因为它可能是未完成的K线
                # 这确保我们只使用已完成的K线进行分析
                if len(df) > 1:
                    df = df[:-1]  # 移除最后一行
                elif len(df) == 1:
                    # 如果只有一根K线，则使用它（虽然可能未完成）
                    pass
                
                # 确保有足够的数据用于分析
                timeframe_name = {mt5.TIMEFRAME_M1: 'M1', mt5.TIMEFRAME_M5: 'M5', mt5.TIMEFRAME_M15: 'M15'}.get(timeframe, str(timeframe))
                if len(df) < bars_count * 0.8:  # 至少需要80%的数据
                    logger.warning(f"⚠️ {timeframe_name}数据不足，需要{bars_count}根，实际{len(df)}根（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                # 添加基础特征
                logger.debug(f"📊 开始为{timeframe_name}数据添加基础特征")
                df = self.feature_engineer.add_core_features(df)
                logger.debug(f"📊 {timeframe_name}基础特征添加完成，当前列数: {len(df.columns)}")

                # 根据周期添加特征
                if timeframe == mt5.TIMEFRAME_M1:
                    df['rsi_7'] = self.calculate_rsi(df['close'], 7)
                    df['ma3'] = df['close'].rolling(window=3).mean()
                    df['ma7'] = df['close'].rolling(window=7).mean()
                    df['atr_7'] = self.calculate_atr(df['high'], df['low'], df['close'], 7)
                    df = self.add_micro_features(df)
                    # 添加其他M1需要的特征
                    df['rsi_14'] = self.calculate_rsi(df['close'], 14)
                    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # 计算布林带位置
                    bb_middle = df['close'].rolling(window=20).mean()
                    bb_std = df['close'].rolling(window=20).std()
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std
                    df['bollinger_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    
                    # 计算ma5, ma10, ma20
                    df['ma5'] = df['close'].rolling(window=5).mean()
                    df['ma10'] = df['close'].rolling(window=10).mean()
                    df['ma20'] = df['close'].rolling(window=20).mean()
                    
                    # 计算方向特征
                    df['ma5_direction'] = (df['ma5'] - df['ma5'].shift(1)) / (df['ma5'].shift(1) + 1e-8)
                    df['ma10_direction'] = (df['ma10'] - df['ma10'].shift(1)) / (df['ma10'].shift(1) + 1e-8)
                    df['ma20_direction'] = (df['ma20'] - df['ma20'].shift(1)) / (df['ma20'].shift(1) + 1e-8)
                    
                    # 计算momentum特征
                    df['price_change_pct'] = df['close'].pct_change()
                    df['momentum_3'] = df['price_change_pct'].rolling(window=3).sum()
                    df['momentum_5'] = df['price_change_pct'].rolling(window=5).sum()
                    
                    # 添加K线形态特征
                    df['body_size'] = abs(df['close'] - df['open'])
                    df['upper_shadow'] = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
                    df['lower_shadow'] = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
                    df['hammer'] = np.where((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < df['body_size']), 1, 0)
                    df['shooting_star'] = np.where((df['upper_shadow'] > 2 * df['body_size']) & (df['lower_shadow'] < df['body_size']), 1, 0)
                    df['engulfing'] = np.where((df['body_size'] > 0) & (df['close'].shift(1) - df['open'].shift(1) < 0) & (df['close'] - df['open'] > 0) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)), 1, 0)
                    
                    # 添加一致性特征
                    df['rsi_price_consistency'] = np.where((df['rsi_14'] > 70) & (df['close'] > df['close'].shift(1)), 1, np.where((df['rsi_14'] < 30) & (df['close'] < df['close'].shift(1)), -1, 0))
                    
                    # 添加跨周期特征
                    df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(5)
                    df['vol_short_vs_medium'] = df['tick_volume'] / (df['tick_volume'].rolling(5).mean() + 1e-8)
                    df['vol_medium_vs_long'] = df['tick_volume'].rolling(5).mean() / (df['tick_volume'].rolling(20).mean() + 1e-8)
                    df['vol_short_vs_long'] = df['tick_volume'] / (df['tick_volume'].rolling(20).mean() + 1e-8)
                    
                    # 添加信号特征
                    df['rsi_signal_strength'] = np.where(df['rsi_14'] > 70, df['rsi_14'] - 70, np.where(df['rsi_14'] < 30, 30 - df['rsi_14'], 0))
                    
                    # 添加风险特征
                    df['volatility_regime'] = np.where(df['volatility_pct'] > df['volatility_pct'].rolling(20).mean(), 1, 0)
                    df['vol_cluster'] = np.where(df['tick_volume'] > df['tick_volume'].rolling(10).mean(), 1, 0)
                    
                    # 添加涨跌动能特征
                    df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).rolling(window=5).sum()
                    df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).rolling(window=5).sum()
                    
                    # 添加其他M1专用特征
                    df['volume_price_divergence'] = (df['tick_volume'] - df['tick_volume'].shift(1)) * (df['close'] - df['close'].shift(1))
                    df['rsi_signal_strength'] = np.where(df['rsi_14'] > 70, df['rsi_14'] - 70, np.where(df['rsi_14'] < 30, 30 - df['rsi_14'], 0))
                    df['short_long_signal_consistency'] = np.where((df['rsi_14'] > 50) & (df['rsi_14'].shift(5) > 50), 1, np.where((df['rsi_14'] < 50) & (df['rsi_14'].shift(5) < 50), -1, 0))
                    
                    # 添加趋势一致性特征
                    df['trend_consistency'] = np.where((df['ma5_direction'] > 0) & (df['ma20_direction'] > 0), 1, np.where((df['ma5_direction'] < 0) & (df['ma20_direction'] < 0), -1, 0))

                elif timeframe == mt5.TIMEFRAME_M5:
                    df = self.feature_engineer.add_enhanced_features(df)
                    df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'], 14)
                    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
                    # 确保momentum特征被计算
                    df['price_change_pct'] = df['close'].pct_change()
                    df['momentum_3'] = df['price_change_pct'].rolling(window=3).sum()
                    df['momentum_5'] = df['price_change_pct'].rolling(window=5).sum()
                    
                    # 添加K线形态特征
                    df['body_size'] = abs(df['close'] - df['open'])
                    df['upper_shadow'] = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
                    df['lower_shadow'] = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
                    df['hammer'] = np.where((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < df['body_size']), 1, 0)
                    df['shooting_star'] = np.where((df['upper_shadow'] > 2 * df['body_size']) & (df['lower_shadow'] < df['body_size']), 1, 0)
                    df['engulfing'] = np.where((df['body_size'] > 0) & (df['close'].shift(1) - df['open'].shift(1) < 0) & (df['close'] - df['open'] > 0) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)), 1, 0)
                    
                    # 添加技术指标
                    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # 计算ma5, ma10, ma20和方向
                    df['ma5'] = df['close'].rolling(window=5).mean()
                    df['ma10'] = df['close'].rolling(window=10).mean()
                    df['ma20'] = df['close'].rolling(window=20).mean()
                    df['ma5_direction'] = (df['ma5'] - df['ma5'].shift(1)) / (df['ma5'].shift(1) + 1e-8)
                    df['ma10_direction'] = (df['ma10'] - df['ma10'].shift(1)) / (df['ma10'].shift(1) + 1e-8)
                    df['ma20_direction'] = (df['ma20'] - df['ma20'].shift(1)) / (df['ma20'].shift(1) + 1e-8)
                    
                    # 添加一致性特征
                    df['rsi_price_consistency'] = np.where((df['rsi_14'] > 70) & (df['close'] > df['close'].shift(1)), 1, np.where((df['rsi_14'] < 30) & (df['close'] < df['close'].shift(1)), -1, 0))
                    
                    # 添加跨周期特征
                    df['rsi_divergence'] = df['rsi_14'] - df['rsi_14'].shift(5)
                    df['vol_short_vs_medium'] = df['tick_volume'] / (df['tick_volume'].rolling(5).mean() + 1e-8)
                    df['vol_medium_vs_long'] = df['tick_volume'].rolling(5).mean() / (df['tick_volume'].rolling(20).mean() + 1e-8)
                    df['vol_short_vs_long'] = df['tick_volume'] / (df['tick_volume'].rolling(20).mean() + 1e-8)
                    
                    # 添加信号特征
                    df['rsi_signal_strength'] = np.where(df['rsi_14'] > 70, df['rsi_14'] - 70, np.where(df['rsi_14'] < 30, 30 - df['rsi_14'], 0))
                    df['macd_signal_strength'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
                    df['short_long_signal_consistency'] = np.where((df['rsi_14'] > 50) & (df['rsi_14'].shift(5) > 50), 1, np.where((df['rsi_14'] < 50) & (df['rsi_14'].shift(5) < 50), -1, 0))
                    
                    # 添加风险特征
                    df['volatility_regime'] = np.where(df['volatility_pct'] > df['volatility_pct'].rolling(20).mean(), 1, 0)
                    df['vol_cluster'] = np.where(df['tick_volume'] > df['tick_volume'].rolling(10).mean(), 1, 0)
                    
                    # 添加M5专用周期共振特征
                    df['m15_trend_ma_consistency'] = 0  # Placeholder, would need M15 data
                    df['m5_m1_volume_correlation'] = df['tick_volume'].rolling(window=5).corr(df['tick_volume'].shift(5)).fillna(0)
                    df['trend_strength_m5_m15'] = abs(df['ma5_direction'])  # Placeholder
                    df['cycle_alignment_score'] = (df['ma5_direction'] + df['ma10_direction'] + df['ma20_direction']) / 3
                    
                    # 添加跨周期联动特征
                    df['m5_m15_volume_correlation'] = df['tick_volume'].rolling(window=10).corr(df['tick_volume'].shift(10)).fillna(0)
                    df['volatility_diff_m5_m1'] = df['volatility_pct'] - df['volatility_pct'].shift(5)
                    
                    # 添加趋势强度特征
                    df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], 14)
                    df['ma5_ma20_alignment'] = np.where(
                        (df['ma5_direction'] > 0) & (df['ma20_direction'] > 0), 1,  # 多头排列
                        np.where(
                            (df['ma5_direction'] < 0) & (df['ma20_direction'] < 0), -1,  # 空头排列
                            0  # 方向不一致
                        )
                    )
                    
                    # 添加涨跌动能特征
                    df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int).rolling(window=5).sum()
                    df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int).rolling(window=5).sum()
                    
                    # 添加K线实体强度和影线特征
                    df['body_strength'] = df['body_size'] / (df['high'] - df['low'] + 1e-8)
                    df['upper_shadow'] = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
                    df['lower_shadow'] = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
                    df['price_position_5'] = (df['close'] - df['low'].rolling(5).min()) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-8)
                    
                    # 添加动态活跃度特征
                    df['dynamic_activity'] = self.calculate_dynamic_activity(df)
                    df['activity_level'] = 1  # Placeholder
                    
                    # 添加跌类专属特征
                    df['volume_up_ratio'] = (df['tick_volume'] * (df['price_change_pct'] < 0)).rolling(window=10).sum() / (df['tick_volume'].rolling(window=10).sum() + 1e-8)
                    df['atr_down_prob'] = np.where(
                        (df['atr_14'] / df['atr_14'].rolling(window=10).mean() > 1.2) & (df['price_change_pct'] < 0), 1, 0
                    )

                elif timeframe == mt5.TIMEFRAME_M15:
                    df['rsi_21'] = self.calculate_rsi(df['close'], 21)
                    df['ma21'] = df['close'].rolling(window=21).mean()
                    df['atr_21'] = self.calculate_atr(df['high'], df['low'], df['close'], 21)
                    df['trend_strength'] = abs(df['ma21'] - df['close']) / df['close']
                    # 添加M15需要的其他特征
                    df['rsi_14'] = self.calculate_rsi(df['close'], 14)
                    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
                    df['macd_signal'] = df['macd'].ewm(span=9).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    # 计算布林带位置
                    bb_middle = df['close'].rolling(window=20).mean()
                    bb_std = df['close'].rolling(window=20).std()
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std
                    df['bollinger_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
                    
                    # 计算ma5, ma20和方向
                    df['ma5'] = df['close'].rolling(window=5).mean()
                    df['ma20'] = df['close'].rolling(window=20).mean()
                    df['ma5_direction'] = (df['ma5'] - df['ma5'].shift(1)) / (df['ma5'].shift(1) + 1e-8)
                    df['ma20_direction'] = (df['ma20'] - df['ma20'].shift(1)) / (df['ma20'].shift(1) + 1e-8)
                    
                    # 添加K线形态特征
                    df['body_size'] = abs(df['close'] - df['open'])
                    df['upper_shadow'] = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
                    df['lower_shadow'] = np.where(df['close'] > df['open'], df['open'] - df['low'], df['close'] - df['low'])
                    df['hammer'] = np.where((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < df['body_size']), 1, 0)
                    df['shooting_star'] = np.where((df['upper_shadow'] > 2 * df['body_size']) & (df['lower_shadow'] < df['body_size']), 1, 0)
                    df['engulfing'] = np.where((df['body_size'] > 0) & (df['close'].shift(1) - df['open'].shift(1) < 0) & (df['close'] - df['open'] > 0) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1)), 1, 0)
                    
                    # 添加趋势强度特征
                    try:
                        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], 14)
                    except Exception as e:
                        logger.warning(f"⚠️ M15计算adx特征失败: {e}")
                        df['adx'] = 0  # 设置默认值
                    
                    df['ma_trend_alignment'] = np.where(
                        (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']), 1,  # 多头排列
                        np.where(
                            (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']), -1,  # 空头排列
                            0  # 无明显排列
                        )
                    )
                    
                    # 趋势持续时长
                    df['trend_direction'] = np.where(df['close'] > df['open'], 1, np.where(df['close'] < df['open'], -1, 0))
                    df['trend_duration'] = 0
                    current_trend = 0
                    duration = 0
                    trend_durations = []
                    for direction in df['trend_direction']:
                        if direction == current_trend:
                            duration += 1
                        else:
                            current_trend = direction
                            duration = 1
                        trend_durations.append(duration)
                    df['trend_duration'] = trend_durations
                    
                    # 动态活跃度特征
                    try:
                        df = self.calculate_dynamic_activity_m15(df)
                    except Exception as e:
                        logger.warning(f"⚠️ M15计算动态活跃度特征失败: {e}")
                        # 设置默认的动态活跃度特征值
                        df['dynamic_activity'] = 0
                        df['activity_level'] = 1
                        df['dynamic_activity_up_mean'] = 0
                        df['high_activity_up_weight'] = 1.0
                    
                    # 新增跌类专属趋势特征
                    close_pct_change = df['close'].pct_change()
                    df['consecutive_down_momentum'] = np.where(close_pct_change < 0, abs(close_pct_change), 0)
                    df['consecutive_down_momentum'] = df['consecutive_down_momentum'].fillna(0)
                    
                    # ATR21扩张时的下跌概率
                    df['atr_expansion'] = df['atr_21'] / df['atr_21'].rolling(window=10).mean()  # ATR扩张比例
                    df['atr_down_prob'] = np.where(
                        (df['atr_expansion'] > 1.2) & (df['close'].pct_change() < 0), 1, 0
                    )  # ATR扩张且价格下跌
                    
                    # 新增涨类专属趋势特征
                    df['consecutive_up_momentum'] = df['close'].pct_change().rolling(window=2).apply(
                        lambda x: sum([i for i in x if i > 0]), raw=True)  # 仅计算上涨部分
                    df['consecutive_up_momentum'] = df['consecutive_up_momentum'].fillna(0)
                    
                    # MA21向上时的涨概率
                    df['ma21_direction'] = np.where(df['ma21'] > df['ma21'].shift(1), 1, 0)  # MA21向上为1，向下为0
                    df['up_prob_when_ma21_up'] = np.where(
                        (df['ma21_direction'] == 1) & (df['close'].pct_change() > 0), 1, 0
                    )  # MA21向上且价格上涨
                    
                    # ATR21收缩时的涨概率
                    df['atr_contraction'] = np.where(df['atr_21'] < df['atr_21'].rolling(window=10).mean(), 1, 0)  # ATR收缩标记
                    df['up_prob_when_atr_contraction'] = np.where(
                        (df['atr_contraction'] == 1) & (df['close'].pct_change() > 0), 1, 0
                    )  # ATR收缩且价格上涨
                    
                    # dynamic_activity上涨区间均值
                    df['dynamic_activity_up_mean'] = np.where(
                        df['close'].pct_change() > 0, df['dynamic_activity'], np.nan
                    )  # 仅取上涨时的dynamic_activity值
                    df['dynamic_activity_up_mean'] = df['dynamic_activity_up_mean'].rolling(window=21).mean()  # 上涨时的21周期均值
                    df['dynamic_activity_up_mean'] = df['dynamic_activity_up_mean'].fillna(0)
                    
                    # 高波动后上涨概率
                    df['high_volatility_prev'] = np.where(df['volatility_pct'] > df['volatility_pct'].rolling(window=21).mean(), 1, 0)
                    df['up_after_high_volatility'] = np.where(
                        (df['high_volatility_prev'].shift(1) == 1) & (df['close'].pct_change() > 0), 1, 0
                    )  # 前一周期高波动后上涨
                    
                    # 高活跃度涨类加权特征
                    df['high_activity_up_weight'] = np.where((df['activity_level'] == 2) & (df['close'].pct_change() > 0), 1.2, 1.0)
                    
                    # 风险特征
                    df['volatility_regime'] = np.where(df['volatility_pct'] > df['volatility_pct'].rolling(21).mean(), 1, 0)
                    
                    # 添加M15专用的趋势特征
                    df = self.add_trend_features(df)
                    
                    # 添加缺失的特征 - m60_trend_consistency（跨周期趋势特征）
                    # 由于需要M60数据，我们用M15数据的简单替代方案
                    df['m60_trend_consistency'] = 0  # Placeholder, would need M60 data

                # 异常值处理（3σ原则）
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = np.clip(df[col], mean - 3 * std, mean + 3 * std)

                # 清理数据
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna()

                # 只保留需要的特征
                period_key = 'm1' if timeframe == mt5.TIMEFRAME_M1 else 'm5' if timeframe == mt5.TIMEFRAME_M5 else 'm15'
                feature_list = self.FEATURE_CONFIG[period_key]
                available_features = [f for f in feature_list if f in df.columns]
                
                if not available_features:
                    logger.error(f"❌ {period_key.upper()}无可用特征列")
                    return None
                
                df = df[available_features]

                return df

            except Exception as e:
                logger.error(f"Line: {e.__traceback__.tb_lineno}")
                logger.error(f"❌ 获取市场数据失败（重试{retry + 1}/{self.MAX_RETRIES}）: {e}")
                if retry < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_INTERVAL)

        return None

    def get_all_period_data(self):
        """一次性获取所有周期数据，避免重复调用"""
        # 增加获取数据量以满足M15数据需求
        initial_bars = max(self.HISTORY_M1_BARS, self.HISTORY_M5_BARS, self.HISTORY_M15_BARS) + 200

        # 为不同周期分别获取数据，对M15周期使用更多数据
        data = {}
        
        # 获取M1数据
        data['m1'] = self.get_current_market_data(self.M1_TIMEFRAME, initial_bars)
        
        # 获取M5数据
        data['m5'] = self.get_current_market_data(self.M5_TIMEFRAME, initial_bars)
        
        # 获取M15数据 - 使用更多数据并增加重试
        m15_data_retries = 0
        m15_initial_bars = initial_bars + 100
        m15_data = None
        
        while m15_data is None and m15_data_retries < 3:
            m15_data = self.get_current_market_data(self.M15_TIMEFRAME, m15_initial_bars)
            if m15_data is None:
                logger.warning(f"⚠️ 第{m15_data_retries + 1}次获取M15数据失败，增加数据量重试")
                m15_initial_bars += 100  # 增加数据量
                m15_data_retries += 1
            elif len(m15_data) < self.HISTORY_M15_BARS:
                logger.warning(f"⚠️ M15数据不足，需要{self.HISTORY_M15_BARS}根，实际{len(m15_data)}根，增加数据量重试")
                m15_initial_bars += 100  # 增加数据量
                m15_data_retries += 1
                m15_data = None  # 重置数据，重新获取
        
        data['m15'] = m15_data
        
        # 验证数据完整性 - 确保获取到足够的数据
        for period_key, period_data in data.items():
            if period_data is not None:
                min_required = getattr(self, f'HISTORY_{period_key.upper()}_BARS')
                if len(period_data) < min_required:
                    logger.warning(f"⚠️ {period_key.upper()}数据不足，需要{min_required}根，实际{len(period_data)}根")
                else:
                    logger.debug(f"📊 {period_key.upper()}数据获取成功，共{len(period_data)}根K线")

        # 特征标准化
        for period in ['m1', 'm5', 'm15']:
            if data[period] is not None:
                # logger.info(f"📊 {period.upper()}标准化前特征列数: {len(data[period].columns) if data[period] is not None else 0}")
                # 
                if self.scalers.get(period) is not None:
                    feature_cols = [col for col in self.FEATURE_CONFIG[period] if col in data[period].columns]
                    # logger.info(f"📊 {period.upper()}匹配的特征数: {len(feature_cols)}, 配置中定义的特征数: {len(self.FEATURE_CONFIG[period])}")
                    # 
                    if feature_cols:
                        # 检查特征数量是否匹配
                        expected_features = self.scalers[period].n_features_in_ if hasattr(self.scalers[period], 'n_features_in_') else len(feature_cols)
                        # logger.info(f"📊 {period.upper()}标准化器期望特征数: {expected_features}, 实际可用特征数: {len(feature_cols)}")
                        #
                        if len(feature_cols) == expected_features:
                            try:
                                transformed_data = self.scalers[period].transform(data[period][feature_cols])
                                # 将转换后的数据赋回原DataFrame
                                data[period][feature_cols] = transformed_data
                                logger.debug(f"✅ {period.upper()}标准化完成")
                            except ValueError as e:
                                logger.warning(f"⚠️ {period}标准化器特征数量不匹配: {e}，跳过标准化")
                            except Exception as e:
                                logger.warning(f"⚠️ {period}标准化器应用失败: {e}，跳过标准化")
                        else:
                            logger.warning(f"⚠️ {period}特征数量不匹配: 期望{expected_features}，实际{len(feature_cols)}，跳过标准化")
                    else:
                        logger.warning(f"⚠️ {period}无匹配特征，跳过标准化")
            else:
                logger.warning(f"⚠️ {period}数据为None，跳过标准化")

        return data

    def calculate_signal(self, df, period_key):
        """通用信号计算方法"""
        try:
            min_bars = self.HISTORY_M1_BARS if period_key == 'm1' else self.HISTORY_M5_BARS if period_key == 'm5' else self.HISTORY_M15_BARS

            if len(df) < min_bars:
                logger.warning(f"⚠️ {period_key.upper()}数据不足，需要{min_bars}根K线，当前{len(df)}根")
                return 0.0, 0.0, 0.0

            # 获取特征列
            feature_columns = self.FEATURE_CONFIG[period_key]
            available_features = [col for col in feature_columns if col in df.columns]

            if not available_features:
                logger.error(f"❌ {period_key.upper()}无可用特征")
                return 0.0, 0.0, 0.0

            # 检查是否需要标准化以及特征数量是否匹配
            if self.scalers.get(period_key) is not None:
                expected_features = self.scalers[period_key].n_features_in_ if hasattr(self.scalers[period_key], 'n_features_in_') else len(available_features)
                if len(available_features) != expected_features:
                    logger.warning(f"⚠️ {period_key.upper()}特征数量不匹配: 期望{expected_features}，实际{len(available_features)}")
                    # 尝试找到共同特征
                    if hasattr(self.scalers[period_key], 'feature_names_in_'):
                        scaler_features = set(self.scalers[period_key].feature_names_in_)
                        available_features = [f for f in self.FEATURE_CONFIG[period_key] if f in df.columns and f in scaler_features]
                    if not available_features:
                        logger.error(f"❌ {period_key.upper()}无匹配特征")
                        return 0.0, 0.0, 0.0

            # 获取最新的特征数据
            latest_row = df.iloc[-1][available_features]
            latest_data = latest_row.values.reshape(1, -1)
            
            # 检查数据中是否包含NaN或无穷大值
            if np.isnan(latest_data).any() or np.isinf(latest_data).any():
                logger.warning(f"⚠️ {period_key.upper()}特征数据包含NaN或无穷大值，进行填充处理")
                # 使用前一个有效值填充NaN
                latest_data = pd.DataFrame(latest_data).fillna(method='ffill').fillna(method='bfill').values
                # 检查是否仍然包含NaN或无穷大值
                if np.isnan(latest_data).any() or np.isinf(latest_data).any():
                    logger.error(f"❌ {period_key.upper()}特征数据无法修复，跳过预测")
                    return 0.0, 0.0, 0.0

            # 如果有标准化器，应用标准化
            if self.scalers.get(period_key) is not None and len(available_features) > 0:
                try:
                    if hasattr(self.scalers[period_key], 'feature_names_in_'):
                        scaler_features = set(self.scalers[period_key].feature_names_in_)
                        if set(available_features) == scaler_features:
                            latest_data = self.scalers[period_key].transform(latest_data)
                        else:
                            logger.warning(f"⚠️ {period_key.upper()}特征名称不匹配，跳过标准化")
                    else:
                        # 检查特征数量是否匹配
                        expected_features = self.scalers[period_key].n_features_in_ if hasattr(self.scalers[period_key], 'n_features_in_') else len(available_features)
                        if len(available_features) == expected_features:
                            latest_data = self.scalers[period_key].transform(latest_data)
                        else:
                            logger.warning(f"⚠️ {period_key.upper()}特征数量不匹配: 期望{expected_features}，实际{len(available_features)}，跳过标准化")
                except Exception as e:
                    logger.warning(f"⚠️ {period_key.upper()}标准化失败: {e}，跳过标准化")

            # 创建DMatrix进行预测
            dtest = xgb.DMatrix(latest_data)

            # 预测概率
            try:
                pred_proba_raw = self.models[period_key].predict(dtest)
            except Exception as e:
                logger.error(f"❌ {period_key.upper()}模型预测失败: {e}")
                return 0.0, 0.0, 0.0
            
            # 确保pred_proba是numpy数组的一维数组
            if isinstance(pred_proba_raw, (list, np.ndarray)):
                pred_proba = pred_proba_raw[0] if len(pred_proba_raw) > 0 else pred_proba_raw
            else:
                pred_proba = pred_proba_raw
            
            # 检查预测结果是否为有效的数值
            if not isinstance(pred_proba, np.ndarray) and not isinstance(pred_proba, (list, tuple)):
                logger.error(f"❌ {period_key.upper()}预测结果格式不正确: {type(pred_proba)}")
                return 0.0, 0.0, 0.0
            
            # 转换为numpy数组以确保可以正确索引
            pred_proba = np.array(pred_proba)
            
            # 使用标签映射获取正确的概率分布
            label_mapping = self.label_mappings.get(period_key, {-1: 0, 0: 1, 1: 2})
            down_idx = label_mapping.get(-1, 0)
            hold_idx = label_mapping.get(0, 1)
            up_idx = label_mapping.get(1, 2)

            # 确保索引有效
            down_prob = pred_proba[down_idx] if down_idx < len(pred_proba) else 0.0
            hold_prob = pred_proba[hold_idx] if hold_idx < len(pred_proba) else 0.0
            up_prob = pred_proba[up_idx] if up_idx < len(pred_proba) else 0.0

            # 检查概率值是否为有效数值
            if np.isnan(up_prob) or np.isnan(down_prob) or np.isnan(hold_prob):
                logger.warning(f"⚠️ {period_key.upper()}预测概率包含NaN值，使用默认值")
                return 0.0, 0.0, 1.0  # 默认返回观望

            # 归一化概率
            total = up_prob + down_prob + hold_prob
            if total > 0:
                up_prob /= total
                down_prob /= total
                hold_prob /= total
            else:
                # 如果总和为0，设置为默认值
                up_prob, down_prob, hold_prob = 0.0, 0.0, 1.0

            logger.debug(
                f"📊 {period_key.upper()}周期预测概率 - 上涨: {up_prob:.4f}, 下跌: {down_prob:.4f}, 观望: {hold_prob:.4f}")

            return up_prob, down_prob, hold_prob

        except Exception as e:
            logger.error(f"❌ 计算{period_key.upper()}信号失败: {e}")
            return 0.0, 0.0, 0.0

    def calculate_fused_signal(self):
        """计算融合信号"""
        try:
            # 一次性获取所有周期数据
            data = self.get_all_period_data()

            if any(value is None for value in data.values()):
                logger.error("❌ 获取多周期数据失败")
                return "HOLD", 0.0

            # 计算各周期信号
            m1_up, m1_down, m1_hold = self.calculate_signal(data['m1'], 'm1')
            m5_up, m5_down, m5_hold = self.calculate_signal(data['m5'], 'm5')
            m15_up, m15_down, m15_hold = self.calculate_signal(data['m15'], 'm15')
            
            # 输出简化的多周期预测概率（一行显示）
            logger.info(f"📊 多周期预测 - M1(涨{m1_up:.4f}/跌{m1_down:.4f}/观{m1_hold:.4f}) | M5(涨{m5_up:.4f}/跌{m5_down:.4f}/观{m5_hold:.4f}) | M15(涨{m15_up:.4f}/跌{m15_down:.4f}/观{m15_hold:.4f})")

            # 应用权重融合信号
            fused_up = (m1_up * self.MODEL_WEIGHTS['m1'] +
                        m5_up * self.MODEL_WEIGHTS['m5'] +
                        m15_up * self.MODEL_WEIGHTS['m15'])

            fused_down = (m1_down * self.MODEL_WEIGHTS['m1'] +
                          m5_down * self.MODEL_WEIGHTS['m5'] +
                          m15_down * self.MODEL_WEIGHTS['m15'])

            fused_hold = (m1_hold * self.MODEL_WEIGHTS['m1'] +
                          m5_hold * self.MODEL_WEIGHTS['m5'] +
                          m15_hold * self.MODEL_WEIGHTS['m15'])

            # 动态调整阈值（基于近期准确率）
            current_accuracy = self.get_recent_accuracy()
            dynamic_threshold = max(self.MIN_THRESHOLD,
                                    min(self.MAX_THRESHOLD, self.BASE_THRESHOLD - (current_accuracy - 0.5) * 0.2))

            # 生成最终信号
            if fused_up > dynamic_threshold:
                signal = "BUY"
                confidence = fused_up
                reason = f"综合上涨概率 {fused_up:.4f} 超过动态阈值{dynamic_threshold:.2f}"
            elif fused_down > dynamic_threshold:
                signal = "SELL"
                confidence = fused_down
                reason = f"综合下跌概率 {fused_down:.4f} 超过动态阈值{dynamic_threshold:.2f}"
            else:
                signal = "HOLD"
                confidence = max(fused_up, fused_down)
                reason = f"无明确方向，动态阈值{dynamic_threshold:.2f}"

            logger.debug(f"🔍 融合信号 - 上涨: {fused_up:.4f}, 下跌: {fused_down:.4f}, 阈值: {dynamic_threshold:.2f}")
            logger.info(f"📢 交易信号: {signal} (置信度: {confidence:.4f}) - {reason}")

            return signal, confidence

        except Exception as e:
            logger.error(f"❌ 计算融合信号失败: {e}")
            return "HOLD", 0.0

    def calculate_dynamic_stop_take(self, entry_price, signal_type, m5_data):
        """基于ATR动态计算止盈止损"""
        try:
            # 获取M5周期的ATR
            atr = m5_data['atr_14'].iloc[-1] if 'atr_14' in m5_data.columns else 0.5

            # 波动率系数调整
            vol_pct = m5_data['volatility_pct'].iloc[-1] if 'volatility_pct' in m5_data.columns else 1.0
            vol_mean = m5_data['volatility_pct'].rolling(20).mean().iloc[
                -1] if 'volatility_pct' in m5_data.columns else 1.0

            if vol_pct > 1.5 * vol_mean:
                vol_coeff = self.VOL_HIGH_COEFF
            elif vol_pct < 0.5 * vol_mean:
                vol_coeff = self.VOL_LOW_COEFF
            else:
                vol_coeff = 1.0

            # 计算止损止盈点位（XAUUSD 1点=0.1美金）
            stop_loss_points = atr * self.ATR_STOP_LOSS * vol_coeff * 10
            take_profit_points = atr * self.ATR_TAKE_PROFIT * vol_coeff * 10

            # 转换为价格
            if signal_type == "BUY":
                sl = entry_price - stop_loss_points / 100
                tp = entry_price + take_profit_points / 100
            else:
                sl = entry_price + stop_loss_points / 100
                tp = entry_price - take_profit_points / 100

            # 价格合法性校验
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick:
                if signal_type == "BUY":
                    sl = max(sl, tick.bid * 0.99)  # 止损不低于当前价格的99%
                    tp = min(tp, tick.ask * 1.01)  # 止盈不高于当前价格的101%
                else:
                    sl = min(sl, tick.ask * 1.01)
                    tp = max(tp, tick.bid * 0.99)

            logger.info(f"🎯 动态止盈止损计算 - ATR: {atr:.4f}, 波动率系数: {vol_coeff:.2f}")
            logger.info(f"🎯 {signal_type} - 止损: {sl:.5f}, 止盈: {tp:.5f}")

            return sl, tp

        except Exception as e:
            logger.error(f"❌ 计算动态止盈止损失败: {e}")
            # 兜底方案
            sl = entry_price - 6 if signal_type == "BUY" else entry_price + 6
            tp = entry_price + 10 if signal_type == "BUY" else entry_price - 10
            return sl, tp

    def place_order(self, signal):
        """下单（带重试和成交确认）"""
        for retry in range(self.MAX_RETRIES):
            try:
                # 获取当前价格
                tick = mt5.symbol_info_tick(self.SYMBOL)
                if tick is None:
                    logger.error(f"❌ 无法获取当前价格（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                # 确定订单类型
                if signal == "BUY":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                elif signal == "SELL":
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    logger.warning("⚠️ 无效的交易信号")
                    return False

                # 获取M5数据用于动态止盈止损计算
                m5_data = self.get_current_market_data(self.M5_TIMEFRAME, self.HISTORY_M5_BARS)
                if m5_data is not None and len(m5_data) > 0:
                    sl, tp = self.calculate_dynamic_stop_take(price, signal, m5_data)
                else:
                    # 兜底方案
                    sl = price - 6 if signal == "BUY" else price + 6
                    tp = price + 10 if signal == "BUY" else price - 10

                # 准备订单请求
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.SYMBOL,
                    "volume": self.LOT_SIZE,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 20,
                    "magic": self.MAGIC_NUMBER,
                    "comment": f"多周期信号交易_{signal}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                # 执行订单
                result = mt5.order_send(request)
                if result is None:
                    logger.error(f"❌ 订单发送失败（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(
                        f"❌ 订单执行失败（重试{retry + 1}/{self.MAX_RETRIES}）: {result.retcode} - {result.comment}")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                # 确认订单成交（轮询检查持仓）
                time.sleep(1)
                self.check_existing_positions()
                if self.current_position:
                    logger.info(
                        f"✅ 开仓成功: {signal} | 手数: {self.LOT_SIZE} | 订单号: {result.order} | 入场价: {price:.5f}")
                    # 记录交易，使用XAUUSD市场数据时间
                    current_tick = mt5.symbol_info_tick(self.SYMBOL)
                    if current_tick:
                        trade_time = datetime.fromtimestamp(current_tick.time)
                        self.daily_trades.append({
                            'time': trade_time,
                            'type': signal,
                            'price': price,
                            'sl': sl,
                            'tp': tp,
                            'ticket': result.order
                        })
                    else:
                        logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                        raise Exception("无法获取XAUUSD市场数据时间")
                    return True
                else:
                    logger.warning(f"⚠️ 订单返回成功但未检测到持仓（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)

            except Exception as e:
                logger.error(f"❌ 下单失败（重试{retry + 1}/{self.MAX_RETRIES}）: {e}")
                if retry < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_INTERVAL)

        logger.error(f"❌ 下单失败，已重试{self.MAX_RETRIES}次")
        return False

    def close_position(self, reason=""):
        """平仓（带重试和成交确认）"""
        if self.current_position is None:
            logger.info("ℹ️ 当前无持仓")
            return True

        for retry in range(self.MAX_RETRIES):
            try:
                # 获取持仓信息
                ticket = self.current_position['ticket']
                pos_type = self.current_position['type']

                # 获取当前价格
                tick = mt5.symbol_info_tick(self.SYMBOL)
                if tick is None:
                    logger.error(f"❌ 无法获取当前价格（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                # 确定平仓价格和类型
                if pos_type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask

                # 准备平仓订单请求
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.SYMBOL,
                    "volume": self.LOT_SIZE,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "magic": self.MAGIC_NUMBER,
                    "comment": f"多周期平仓_{reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                # 执行平仓订单
                result = mt5.order_send(request)
                if result is None:
                    logger.error(f"❌ 平仓订单发送失败（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(
                        f"❌ 平仓订单执行失败（重试{retry + 1}/{self.MAX_RETRIES}）: {result.retcode} - {result.comment}")
                    time.sleep(self.RETRY_INTERVAL)
                    continue

                # 确认平仓
                time.sleep(1)
                self.check_existing_positions()
                if self.current_position is None:
                    logger.info(f"✅ 平仓成功: {reason} | 订单号: {ticket} | 平仓价: {price:.5f}")
                    # 更新预测准确率
                    self.update_prediction_accuracy(reason)
                    return True
                else:
                    logger.warning(f"⚠️ 平仓订单返回成功但仍有持仓（重试{retry + 1}/{self.MAX_RETRIES}）")
                    time.sleep(self.RETRY_INTERVAL)

            except Exception as e:
                logger.error(f"❌ 平仓失败（重试{retry + 1}/{self.MAX_RETRIES}）: {e}")
                if retry < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_INTERVAL)

        logger.error(f"❌ 平仓失败，已重试{self.MAX_RETRIES}次")
        return False

    def check_and_close_by_signal(self, current_signal):
        """根据信号检查是否需要平仓"""
        if self.current_position is None:
            return False

        try:
            current_direction = self.current_position['direction']

            # 信号反向时平仓
            if (current_direction == "做多" and current_signal == "SELL") or \
                    (current_direction == "做空" and current_signal == "BUY"):
                logger.info("📉 平仓: 信号反向出现")
                return self.close_position("信号反向")

            # 检查持仓盈利（动态阈值）
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions and len(positions) > 0:
                pos = positions[0]
                profit = pos.profit

                # 动态盈利阈值（基于ATR）
                m5_data = self.get_current_market_data(self.M5_TIMEFRAME, self.HISTORY_M5_BARS)
                if m5_data is not None and 'atr_14' in m5_data.columns:
                    atr = m5_data['atr_14'].iloc[-1]
                    dynamic_profit_threshold = atr * 10 * self.LOT_SIZE * 9  # ATR*手数相关
                else:
                    dynamic_profit_threshold = 90  # 兜底

                # 观望信号且盈利超过阈值时平仓
                if current_signal == "HOLD" and profit > dynamic_profit_threshold:
                    logger.info(f"💰 平仓: 观望信号且盈利超过{dynamic_profit_threshold:.2f}美金 ({profit:.2f}美金)")
                    return self.close_position(f"观望信号盈利{profit:.2f}")

        except Exception as e:
            logger.error(f"❌ 检查持仓盈利失败: {e}")

        return False

    def check_daily_close(self):
        """检查是否需要每日收盘前平仓"""
        if self.current_position is None:
            return False

        try:
            # 获取当前市场时间
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick is None:
                return False

            current_time = datetime.fromtimestamp(tick.time)

            # 每日20:00 UTC平仓
            if current_time.hour >= 20 and current_time.minute >= 0:
                logger.info("⏰ 平仓: 每日收盘前平仓")
                return self.close_position("每日收盘")

        except Exception as e:
            logger.error(f"❌ 检查每日平仓失败: {e}")

        return False

    def check_risk_management(self):
        """检查风控管理（精确计算）"""
        try:
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("❌ 无法获取账户信息")
                return False

            balance = account_info.balance
            equity = account_info.equity

            # 计算当日回撤
            if self.daily_start_balance:
                daily_drawdown = (self.daily_start_balance - equity) / self.daily_start_balance
            else:
                daily_drawdown = 0

            # 计算累计回撤
            total_drawdown = (self.INITIAL_BALANCE - equity) / self.INITIAL_BALANCE

            # 检查最大回撤限制
            if daily_drawdown > self.FTMO_MAX_DRAWDOWN or total_drawdown > self.FTMO_MAX_DRAWDOWN:
                logger.warning(f"⚠️ 超过最大回撤限制 - 当日回撤: {daily_drawdown:.4f}, 累计回撤: {total_drawdown:.4f}")
                if self.current_position is not None:
                    logger.info("🛡️ 执行风控平仓")
                    return self.close_position("风控平仓")

            # 检查账户余额
            if balance < self.FTMO_MIN_BALANCE:
                logger.warning(f"⚠️ 账户余额低于最低要求: {balance} < {self.FTMO_MIN_BALANCE}")
                if self.current_position is not None:
                    logger.info("🛡️ 执行风控平仓")
                    return self.close_position("余额不足")

            # 检查盈利目标
            if balance >= self.INITIAL_BALANCE * (1 + self.FTMO_PROFIT_TARGET):
                logger.info(f"🏆 达到盈利目标: {balance} >= {self.INITIAL_BALANCE * (1 + self.FTMO_PROFIT_TARGET)}")
                if self.current_position is not None:
                    logger.info("🛡️ 执行盈利目标平仓")
                    return self.close_position("盈利目标")

        except Exception as e:
            logger.error(f"❌ 风控检查失败: {e}")

        return False

    def update_prediction_accuracy(self, reason):
        """更新预测准确率（关联实际结果）"""
        if not self.prediction_history:
            return

        # 获取最新的预测记录
        latest_pred = self.prediction_history[-1]

        # 获取实际结果
        positions = mt5.history_deals_get(symbol=self.SYMBOL)
        if positions:
            latest_deal = max(positions, key=lambda x: x.time)
            profit = latest_deal.profit
            is_correct = (latest_pred['signal'] == "BUY" and profit > 0) or (
                        latest_pred['signal'] == "SELL" and profit > 0)

            latest_pred['actual_outcome'] = "盈利" if profit > 0 else "亏损"
            latest_pred['is_correct'] = is_correct
            latest_pred['profit'] = profit

            logger.info(
                f"📊 预测结果更新 - 信号: {latest_pred['signal']}, 实际: {latest_pred['actual_outcome']}, 准确率: {self.get_recent_accuracy():.4f}")

    def get_recent_accuracy(self):
        """获取最近的预测准确率"""
        if not self.prediction_history:
            return 0.0

        # 只计算有实际结果的预测
        valid_predictions = [record for record in self.prediction_history if record.get('actual_outcome') is not None]
        if not valid_predictions:
            return 0.0

        correct_predictions = sum(1 for record in valid_predictions if record['is_correct'])
        accuracy = correct_predictions / len(valid_predictions)

        return accuracy

    def incremental_training(self):
        """增量训练模型（带性能验证）"""
        try:
            logger.info("🔄 开始多周期模型增量训练...")

            # 对每个模型进行增量训练
            for period_key in ['m1', 'm5', 'm15']:
                # 获取最新数据
                timeframe = self.M1_TIMEFRAME if period_key == 'm1' else self.M5_TIMEFRAME if period_key == 'm5' else self.M15_TIMEFRAME
                data = self.get_current_market_data(timeframe, 500)

                if data is None or len(data) < 100:
                    logger.warning(f"⚠️ 获取{period_key}新数据不足，跳过增量训练")
                    continue

                # 准备特征和目标变量
                feature_columns = self.FEATURE_CONFIG[period_key]
                available_features = [col for col in feature_columns if col in data.columns]

                if not available_features:
                    logger.warning(f"⚠️ {period_key}无可用特征，跳过增量训练")
                    continue

                # 创建目标变量
                data['future_close'] = data['close'].shift(-1)
                data['price_change_pct'] = (data['future_close'] - data['close']) / data['close']
                data['target'] = np.where(data['price_change_pct'] > 0.001, 1,
                                          np.where(data['price_change_pct'] < -0.001, -1, 0))

                # 准备训练数据
                X = data[available_features].values
                y = data['target'].values

                # 过滤NaN值
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]

                if len(X) < 50:
                    logger.warning(f"⚠️ {period_key}有效训练数据不足，跳过增量训练")
                    continue

                # 使用最近的样本
                n_samples = min(200, len(X))
                X_recent = X[-n_samples:]
                y_recent = y[-n_samples:]

                logger.info(f"📈 使用 {len(X_recent)} 个新样本进行{period_key}模型增量训练")

                # 评估旧模型性能
                dtest = xgb.DMatrix(X_recent, label=y_recent)
                old_pred = self.models[period_key].predict(dtest)
                old_acc = np.mean((old_pred.argmax(axis=1) if len(old_pred.shape) > 1 else old_pred) == y_recent)

                # 增量训练
                dtrain = xgb.DMatrix(X_recent, label=y_recent)
                updated_model = xgb.train(
                    self.models[period_key].save_config(),
                    dtrain,
                    xgb_model=self.models[period_key],
                    num_boost_round=10
                )

                # 评估新模型性能
                new_pred = updated_model.predict(dtest)
                new_acc = np.mean((new_pred.argmax(axis=1) if len(new_pred.shape) > 1 else new_pred) == y_recent)

                # 仅当性能提升≥1%时更新
                if new_acc >= old_acc + 0.01:
                    self.models[period_key] = updated_model
                    # 保存新模型
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    new_model_path = f"xauusd_{period_key}_model_{timestamp}.json"
                    self.models[period_key].save_model(new_model_path)
                    logger.info(f"✅ {period_key}模型更新成功，准确率从{old_acc:.4f}提升至{new_acc:.4f}")
                else:
                    logger.info(f"ℹ️ {period_key}模型未更新，新模型准确率{new_acc:.4f}低于旧模型{old_acc:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ 增量训练失败: {e}")
            return False

    def run_trading_cycle(self):
        """执行单次交易循环"""
        try:
            # 检查是否有暂停交易的标记文件
            if os.path.exists("暂停交易.flag"):
                logger.info("📅 检测到暂停交易标记，暂停交易操作...")
                return False
            
            # 同步持仓状态
            self.check_existing_positions()

            # 计算融合信号
            signal, prob = self.calculate_fused_signal()

            # 获取当前准确率
            current_accuracy = self.get_recent_accuracy()
            logger.info(f"📊 模型最近预测准确率: {current_accuracy:.4f}")

            # 风控检查
            self.check_risk_management()

            # 每日收盘前平仓检查
            self.check_daily_close()

            # 检查是否需要根据信号平仓
            if self.check_and_close_by_signal(signal):
                logger.info("📉 已根据信号平仓")

            # 如果没有持仓且有明确信号，则开仓
            if self.current_position is None and signal in ["BUY", "SELL"]:
                # 动态开仓阈值
                min_confidence = max(0.6, 0.8 - current_accuracy * 0.3)
                if prob > min_confidence:
                    logger.info(f"📈 开仓: {signal} 信号，置信度 {prob:.3f} (阈值: {min_confidence:.3f})")
                    # 记录预测，使用XAUUSD市场数据时间
                    current_tick = mt5.symbol_info_tick(self.SYMBOL)
                    if current_tick:
                        timestamp = datetime.fromtimestamp(current_tick.time)
                        self.prediction_history.append({
                            'signal': signal,
                            'confidence': prob,
                            'timestamp': timestamp,
                            'actual_outcome': None,
                            'is_correct': None
                        })
                    else:
                        logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                        raise Exception("无法获取XAUUSD市场数据时间")
                    # 限制历史长度
                    if len(self.prediction_history) > self.max_history_length:
                        self.prediction_history.pop(0)

                    # 执行开仓
                    self.place_order(signal)
                else:
                    logger.info(f"⚠️ 信号置信度 {prob:.3f} 低于动态阈值 {min_confidence:.3f}，暂不交易")

            # 打印持仓状态
            if self.current_position is not None:
                # 从MT5获取当前持仓的实际盈亏信息
                positions = mt5.positions_get(symbol=self.SYMBOL)
                if positions is not None:
                    # 筛选出属于当前交易器的持仓（通过magic number）
                    filtered_positions = [pos for pos in positions if pos.magic == self.MAGIC_NUMBER]
                    if len(filtered_positions) > 0:
                        current_position_info = filtered_positions[0]
                        profit = current_position_info.profit  # 使用MT5提供的实际盈亏
                        logger.info(
                            f"📌 当前持仓: {self.current_position['direction']}, 盈亏: {profit:.2f}美金")
                    else:
                        # 如果无法从MT5获取持仓信息，使用计算方式作为备选
                        data = self.get_current_market_data(self.M5_TIMEFRAME, 1)
                        if data is not None and len(data) > 0:
                            current_price = data['close'].iloc[-1]  # 获取当前价格
                            profit = 0
                            if self.current_position['direction'] == "做多":  # 做多
                                profit = (current_price - self.current_position['entry_price']) * 100  # XAUUSD标准合约乘数
                            else:  # 做空
                                profit = (self.current_position['entry_price'] - current_price) * 100  # XAUUSD标准合约乘数
                            logger.info(f"📌 当前持仓: {self.current_position['direction']}, 盈亏: {profit:.2f}美金")
                else:
                    # 如果无法获取持仓信息，使用计算方式作为备选
                    data = self.get_current_market_data(self.M5_TIMEFRAME, 1)
                    if data is not None and len(data) > 0:
                        current_price = data['close'].iloc[-1]  # 获取当前价格
                        profit = 0
                        if self.current_position['direction'] == "做多":  # 做多
                            profit = (current_price - self.current_position['entry_price']) * 100  # XAUUSD标准合约乘数
                        else:  # 做空
                            profit = (self.current_position['entry_price'] - current_price) * 100  # XAUUSD标准合约乘数
                        logger.info(f"📌 当前持仓: {self.current_position['direction']}, 盈亏: {profit:.2f}美金")
            else:
                logger.info("📌 当前无持仓")

            return True

        except Exception as e:
            logger.error(f"❌ 交易循环执行失败: {e}", exc_info=True)
            return False

    def get_latest_data(self, timeframe, count=50):

        try:

            # 从MT5获取实时数据，获取额外一根K线以确保我们有足够数据
            rates = mt5.copy_rates_from_pos(self.SYMBOL, timeframe, 0, count + 10)  # 增加获取的数据量以确保有足够的历史数据

            if rates is None or len(rates) == 0:
                logger.warning("获取MT5数据失败或数据为空")
                return None

            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # 根据时间框架过滤已完成的K线
            # 对于M1，确保获取到已完成的分钟K线
            # 对于M5，确保获取到已完成的5分钟K线
            # 对于M15，确保获取到已完成的15分钟K线
            current_tick = mt5.symbol_info_tick(self.SYMBOL)
            if current_tick:
                current_time = datetime.fromtimestamp(current_tick.time)
                
                # 根据不同时间框架确定已完成K线
                if timeframe == self.M1_TIMEFRAME:
                    # M1 K线在当前时间的前1分钟及更早的K线是完成的
                    completed_time = current_time - timedelta(minutes=1)
                    df = df[df['time'] <= completed_time]
                elif timeframe == self.M5_TIMEFRAME:
                    # M5 K线在当前时间的前5分钟及更早的K线是完成的
                    completed_time = current_time - timedelta(minutes=5)
                    df = df[df['time'] <= completed_time]
                elif timeframe == self.M15_TIMEFRAME:
                    # M15 K线在当前时间的前15分钟及更早的K线是完成的
                    completed_time = current_time - timedelta(minutes=15)
                    df = df[df['time'] <= completed_time]
            
            # 确保只返回请求的数量
            if len(df) > count:
                df = df.iloc[-count:]

            return df

        except Exception as e:
            logger.error(f"获取最新数据异常: {str(e)}")
            return None

    def check_kline_update(self):
        """检查K线是否更新"""
        df1 = self.get_latest_data(self.M1_TIMEFRAME, 1)
        df5 = self.get_latest_data(self.M5_TIMEFRAME, 1)
        df15 = self.get_latest_data(self.M15_TIMEFRAME, 1)

        current_kline_time_1 = df1.iloc[-1]['time']
        current_kline_time_5 = df5.iloc[-1]['time']
        current_kline_time_15 = df15.iloc[-1]['time']
        # 打印并验证M1、M5、M15各周期最新K线的时间戳
        logging.info(
            f"📅 最新M1 K线时间: {current_kline_time_1} | "
            f"📅 最新M5 K线时间: {current_kline_time_5} | "
            f"📅 最新M15 K线时间: {current_kline_time_15}"
        )
        return True


    def run_trading_loop(self):
        """运行交易循环（优化版）"""
        self.is_running = True
        self.last_m5_time = None
        logger.info("🚀 开始多周期实时交易循环")
        
        # 首次运行数据新鲜度保障 - 等待最新的已完成K线
        first_run = True
        while first_run:
            m5_rates = mt5.copy_rates_from_pos(self.SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
            if len(m5_rates) > 0:
                current_m5_time = datetime.fromtimestamp(m5_rates[0]['time'])
                # 获取XAUUSD市场数据时间，严格遵守时间源使用规范
                current_tick = mt5.symbol_info_tick(self.SYMBOL)
                if current_tick:
                    current_time = datetime.fromtimestamp(current_tick.time)
                else:
                    # 严格禁止使用本地时间，抛出异常
                    logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                    raise Exception("无法获取XAUUSD市场数据时间")
                
                time_diff = abs((current_time - current_m5_time).total_seconds())
                
                # 如果最新K线时间与当前时间相差超过15分钟，等待并重新获取
                if time_diff > 900:  # 15分钟 = 900秒
                    logger.info(f"📅 首次运行：最新K线时间({current_m5_time})与服务器时间({current_time})相差{time_diff/60:.1f}分钟，等待数据更新...")
                    time.sleep(30)  # 等待30秒后重新检查
                    continue
                else:
                    logger.info(f"📅 首次运行：K线数据新鲜度正常，开始交易")
                    self.last_m5_time = current_m5_time
                    break
            else:
                logger.error("❌ 首次运行：无法获取最新K线数据，等待...")
                time.sleep(30)
                continue
            
            first_run = False

        # 记录上次增量训练时间，使用XAUUSD市场数据时间
        current_tick = mt5.symbol_info_tick(self.SYMBOL)
        if current_tick:
            last_training_time = datetime.fromtimestamp(current_tick.time)
        else:
            logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
            raise Exception("无法获取XAUUSD市场数据时间")

        while self.is_running and not self.stop_event.is_set():
            try:
                # 检查K线更新
                self.check_kline_update()

                # 计算交易信号并执行交易（如果需要）
                signal, confidence = self.calculate_fused_signal()
                if signal != "HOLD":
                    logger.info(f"💡 决策建议: {signal} | 置信度: {confidence:.4f}")
                    # 如果没有持仓，则执行交易
                    if self.current_position is None:
                        self.place_order(signal)
                    else:
                        # 如果有持仓，检查是否需要平仓
                        self.check_and_close_by_signal(signal)
                else:
                    logger.info(f"📊 当前无交易信号，保持观望")

                # 每小时执行一次增量训练，使用XAUUSD市场数据时间
                current_tick = mt5.symbol_info_tick(self.SYMBOL)
                if current_tick:
                    current_time = datetime.fromtimestamp(current_tick.time)
                else:
                    logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                    raise Exception("无法获取XAUUSD市场数据时间")
                if (current_time - last_training_time).total_seconds() >= 3600:
                    self.incremental_training()
                    last_training_time = current_time

                # M1周期检查（60秒）
                time.sleep(CONFIG["TRADING_CYCLE"]["m1"])

            except Exception as e:
                logger.error(f"❌ 交易循环异常: {e}", exc_info=True)
                time.sleep(5)

        logger.info("🛑 多周期实时交易循环结束")

    def stop_trading(self):
        """停止交易"""
        logger.info("🛑 正在停止交易...")
        self.is_running = False
        self.stop_event.set()

        # 如果有持仓，执行平仓
        if self.current_position is not None:
            logger.info("📉 检测到持仓，执行平仓")
            self.close_position("停止交易")

        # 保存当日交易记录，使用XAUUSD市场数据时间
        if self.daily_trades:
            current_tick = mt5.symbol_info_tick(self.SYMBOL)
            if current_tick:
                current_date = datetime.fromtimestamp(current_tick.time).strftime('%Y%m%d')
                with open(f"daily_trades_{current_date}.log", 'w', encoding='utf-8') as f:
                    for trade in self.daily_trades:
                        f.write(f"{trade}\n")
            else:
                logger.error("❌ 无法获取XAUUSD市场时间，严格禁止使用本地时间")
                raise Exception("无法获取XAUUSD市场数据时间")

        # 关闭MT5连接
        mt5.shutdown()
        logger.info("✅ MT5连接已关闭")


def main():
    """主函数"""
    trader = None
    try:
        # 创建多周期交易实例
        trader = MultiPeriodRealTimeTrader(
            m1_model_path="xauusd_m1_model.json",
            m5_model_path="xauusd_m5_model.json",
            m15_model_path="xauusd_m15_model.json"
        )

        # 运行交易循环
        trader.run_trading_loop()

    except KeyboardInterrupt:
        logger.info("🛑 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 交易程序异常: {e}", exc_info=True)
    finally:
        if trader:
            trader.stop_trading()


if __name__ == "__main__":
    main()
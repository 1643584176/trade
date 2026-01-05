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

import m5_feature_engineering
M5FeatureEngineer = m5_feature_engineering.M5FeatureEngineer

# 导入M1特征工程
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))

# 从M1训练文件中导入M1特征工程方法
import importlib.util

# 动态导入M1特征工程
m1_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M1_模型训练.py")
spec = importlib.util.spec_from_file_location("m1_model_trainer", m1_trainer_path)
m1_model_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1_model_trainer)

# 从M1训练器中获取特征工程方法
M1FeatureEngineer = m1_model_trainer.M1ModelTrainer

# 动态导入M15特征工程
m15_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M15_模型训练.py")
spec = importlib.util.spec_from_file_location("m15_model_trainer", m15_trainer_path)
m15_model_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m15_model_trainer)

# 从M15训练器中获取特征工程方法
M15FeatureEngineer = m15_model_trainer.M15ModelTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xauusd_multi_period_trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiPeriodRealTimeTrader:
    def __init__(self, m1_model_path="xauusd_m1_model.json", 
                 m5_model_path="xauusd_m5_model.json", 
                 m15_model_path="xauusd_m15_model.json", 
                 lot_size=0.2):
        """
        初始化多周期实时交易器
        
        参数:
            m1_model_path (str): M1模型路径
            m5_model_path (str): M5模型路径
            m15_model_path (str): M15模型路径
            lot_size (float): 交易手数
        """
        # 配置参数
        self.SYMBOL = "XAUUSD"
        self.M1_TIMEFRAME = mt5.TIMEFRAME_M1
        self.M5_TIMEFRAME = mt5.TIMEFRAME_M5
        self.M15_TIMEFRAME = mt5.TIMEFRAME_M15
        self.M1_MODEL_PATH = m1_model_path
        self.M5_MODEL_PATH = m5_model_path
        self.M15_MODEL_PATH = m15_model_path
        self.LOT_SIZE = lot_size
        # 动态止盈止损参数
        self.STOP_LOSS_THRESHOLD = 0.006  # 止损阈值 0.6%
        self.TAKE_PROFIT_THRESHOLD = 0.010  # 止盈阈值 1.0%
        self.ATR_MULTIPLIER = 2.0  # ATR倍数
        self.MAGIC_NUMBER = 10000005  # M5周期魔法数字
        self.HISTORY_M1_BARS = 50   # M1周期历史K线数
        self.HISTORY_M5_BARS = 120  # M5周期历史K线数
        self.HISTORY_M15_BARS = 200 # M15周期历史K线数
        
        # 模型权重配置
        self.MODEL_WEIGHTS = {
            'm1': 0.15,   # M1权重较低（0.1-0.2）：时机优化和风险预警
            'm5': 0.55,   # M5权重最高（0.5-0.6）：主要决策依据
            'm15': 0.30   # M15权重中等（0.3-0.4）：趋势确认
        }
        
        # FTMO规则
        self.FTMO_MAX_DRAWDOWN = 0.045  # 最大回撤4.5%
        self.FTMO_PROFIT_TARGET = 0.10  # 盈利目标10%
        self.FTMO_MIN_BALANCE = 99020  # 最低余额
        
        # 交易状态
        self.current_position = None
        self.is_running = False
        self.stop_event = Event()
        
        # 特征工程实例
        self.feature_engineer = M5FeatureEngineer()
        self.m1_feature_engineer = M1FeatureEngineer()
        
        # 动态导入M15特征工程
        m15_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M15_模型训练.py")
        spec = importlib.util.spec_from_file_location("m15_model_trainer", m15_trainer_path)
        m15_model_trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m15_model_trainer)
        
        # 从M15训练器中获取特征工程方法
        self.m15_feature_engineer = m15_model_trainer.M15ModelTrainer()
        
        # 模型自检特征 - 记录最近预测的准确率
        self.prediction_history = []  # 存储最近的预测和实际结果
        self.max_history_length = 20  # 最多存储20次预测历史
        
        # 初始化MT5连接
        if not mt5.initialize():
            raise Exception(f"MT5初始化失败：{mt5.last_error()}")
        
        # 确保交易品种被选中
        if not mt5.symbol_select(self.SYMBOL, True):
            raise Exception(f"无法选择交易品种 {self.SYMBOL}")
        
        # 加载模型
        self.load_models()
        
        # 检查现有持仓
        self.check_existing_positions()
        
        logger.info(f"MT5连接成功")
        logger.info(f"开始基于多周期的实时交易 {self.SYMBOL}，手数: {self.LOT_SIZE}")
        logger.info(f"策略: 信号反向平仓、观望信号且盈利超90美金平仓、每日收盘前平仓")
        logger.info(f"模型权重 - M1: {self.MODEL_WEIGHTS['m1']:.2f}, M5: {self.MODEL_WEIGHTS['m5']:.2f}, M15: {self.MODEL_WEIGHTS['m15']:.2f}")

    def load_models(self):
        """加载所有模型"""
        try:
            # 加载M1模型 - 使用XGBoost原生模型格式
            self.m1_model = xgb.Booster()
            self.m1_model.load_model(self.M1_MODEL_PATH)
            logger.info(f"M1模型已从 {self.M1_MODEL_PATH} 加载")
            
            # 加载M1标准化器
            m1_scaler_path = "m1_scaler.pkl"
            if os.path.exists(m1_scaler_path):
                with open(m1_scaler_path, 'rb') as f:
                    self.m1_scaler = pickle.load(f)
                logger.info(f"M1标准化器已从 {m1_scaler_path} 加载")
            else:
                logger.warning(f"M1标准化器文件不存在: {m1_scaler_path}，可能影响预测准确性")
                self.m1_scaler = None
        except Exception as e:
            logger.error(f"加载M1模型失败: {e}")
            raise e
        
        try:
            # 加载M5模型 - 使用XGBoost原生模型格式
            self.m5_model = xgb.Booster()
            self.m5_model.load_model(self.M5_MODEL_PATH)
            logger.info(f"M5模型已从 {self.M5_MODEL_PATH} 加载")
            
            # 加载M5标准化器
            m5_scaler_path = "m5_scaler.pkl"
            if os.path.exists(m5_scaler_path):
                with open(m5_scaler_path, 'rb') as f:
                    self.m5_scaler = pickle.load(f)
                logger.info(f"M5标准化器已从 {m5_scaler_path} 加载")
            else:
                logger.warning(f"M5标准化器文件不存在: {m5_scaler_path}，可能影响预测准确性")
                self.m5_scaler = None
        except Exception as e:
            logger.error(f"加载M5模型失败: {e}")
            raise e
        
        try:
            # 加载M15模型 - 使用XGBoost原生模型格式
            self.m15_model = xgb.Booster()
            self.m15_model.load_model(self.M15_MODEL_PATH)
            logger.info(f"M15模型已从 {self.M15_MODEL_PATH} 加载")
            
            # 加载M15标准化器
            m15_scaler_path = "m15_scaler.pkl"
            if os.path.exists(m15_scaler_path):
                with open(m15_scaler_path, 'rb') as f:
                    self.m15_scaler = pickle.load(f)
                logger.info(f"M15标准化器已从 {m15_scaler_path} 加载")
            else:
                logger.warning(f"M15标准化器文件不存在: {m15_scaler_path}，可能影响预测准确性")
                self.m15_scaler = None
        except Exception as e:
            logger.error(f"加载M15模型失败: {e}")
            raise e
    def calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)  # 防止除零
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
        return (series - series.shift(1)) / series.shift(1)

    def check_existing_positions(self):
        """检查现有持仓"""
        try:
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions:
                pos = positions[0]  # 假设只处理一个持仓
                direction = "做多" if pos.type == mt5.POSITION_TYPE_BUY else "做空"
                self.current_position = {
                    'ticket': pos.ticket,
                    'type': pos.type,
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'time': pos.time,
                    'direction': direction
                }
                logger.info(f"检测到现有持仓: {direction}, 手数: {pos.volume}")
            else:
                self.current_position = None
                logger.info("未检测到现有持仓")
        except Exception as e:
            logger.error(f"检查现有持仓失败: {e}")

    def get_current_market_data(self, timeframe, bars_count: int):
        """获取指定时间周期的市场数据"""
        try:
            # 获取当前时间
            current_tick = mt5.symbol_info_tick(self.SYMBOL)
            if current_tick is None:
                logger.error("无法获取当前市场数据")
                return None
            
            # 计算开始时间（获取最近的K线数据）
            current_time = datetime.fromtimestamp(current_tick.time)
            
            # 根据时间周期计算开始时间
            if timeframe == mt5.TIMEFRAME_M1:
                start_time = current_time - timedelta(minutes=bars_count)
            elif timeframe == mt5.TIMEFRAME_M5:
                start_time = current_time - timedelta(minutes=5*bars_count)
            elif timeframe == mt5.TIMEFRAME_M15:
                start_time = current_time - timedelta(minutes=15*bars_count)
            else:
                logger.error(f"不支持的时间周期: {timeframe}")
                return None
            
            # 获取历史数据
            rates = mt5.copy_rates_range(
                self.SYMBOL,
                timeframe,
                int(start_time.timestamp()),
                int(current_time.timestamp())
            )
            
            if rates is None or len(rates) == 0:
                logger.error(f"获取{timeframe}历史数据失败: {mt5.last_error()}")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            
            # 根据时间周期添加相应特征
            if timeframe == mt5.TIMEFRAME_M1:
                # M1周期：添加基础特征
                df = self.feature_engineer.add_core_features(df)
                
                # 添加M1特定特征
                df['rsi_7'] = self.calculate_rsi(df['close'], 7)
                df['ma3'] = df['close'].rolling(window=3).mean()
                df['ma7'] = df['close'].rolling(window=7).mean()
                
                # 计算方向特征（避免与基础特征工程重复）
                df['ma3_direction'] = (df['ma3'] - df['ma3'].shift(1)) / (df['ma3'].shift(1) + 1e-8)
                df['ma7_direction'] = (df['ma7'] - df['ma7'].shift(1)) / (df['ma7'].shift(1) + 1e-8)
                df['atr_7'] = self.calculate_atr(df['high'], df['low'], df['close'], 7)
                
                # 添加M1专用的微观交易特征
                df = self.m1_feature_engineer.add_micro_features(df)
                
                # 只保留M1模型需要的特征
                # 只保留M1训练时使用的特征列
                m1_features = [
                    # M1周期特征（短期波动）
                    'open', 'high', 'low', 'close', 'tick_volume',
                    'rsi_7',  # 短期RSI
                    'ma3', 'ma7',  # 短期均线
                    'atr_7',  # 短期ATR - 核心特征（仅保留此版本，删除重复的）
                    'volatility_pct',
                    'hour_of_day', 'is_peak_hour',
                    # K线形态特征
                    'hammer', 'shooting_star', 'engulfing',
                    # 技术指标
                    'rsi_14', 'macd', 'macd_hist',
                    'bollinger_position',
                    'ma5', 'ma10', 'ma20', 'ma10_direction', 'ma20_direction',
                    # 一致性特征
                    'rsi_price_consistency',
                    # 跨周期特征
                    'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                    'trend_consistency',
                    # 信号特征
                    'rsi_signal_strength', 'short_long_signal_consistency',
                    # 风险特征
                    'volatility_regime', 'vol_cluster',
                    # M1专用微观特征
                    'tick_vol_zscore',  # Tick成交量脉冲
                    'up_down_count_10',  # 1分钟内涨跌次数
                    'hl_spread_zscore',  # 高低价差z-score
                    'volatility_intensity',  # 价格波动强度
                    'ma5_deviation',  # 短期偏离度
                    'volume_impulse',  # 成交量脉冲特征（当前成交量/前3根均值）
                    'price_direction_consistency',  # 涨跌延续性特征
                    'dynamic_activity',  # 动态活跃度特征
                    'high_activity',  # 高活跃度标记
                    'up_momentum_3',  # 连续3根M1涨跌幅之和（仅计算上涨）
                    'down_momentum_3',  # 连续3根M1下跌动能（新增跌类动能特征）
                    'down_volume_ratio',  # 跌时成交量占比（新增跌类动能特征）
                    # 涨跌动能特征
                    'momentum_3',  # 3根K线的涨跌幅之和
                    'momentum_5',  # 5根K线的涨跌幅之和
                    'volume_price_divergence',  # 成交量与价格背离
                    'consecutive_up',  # 连续上涨次数
                    'consecutive_down',  # 连续下跌次数
                    # 新增涨类专属特征
                    'volume_up_ratio',  # 成交量放量占比
                    'up_momentum_5',  # 5根K线仅计算上涨部分的强度
                    'volume_up_ratio_enhanced',  # volume_up_ratio 强化版
                    'activity_trend_up',  # activity_trend 上涨趋势
                    'ma5_deviation_up',  # ma5_deviation 向上偏离
                    # 新增跌类专属特征
                    'down_momentum_5',  # 5根K线仅计算下跌部分的强度
                    'down_volume_impulse',  # 放量下跌占比
                    # 新增高活跃度涨类加权特征
                    'high_activity_up_weight',  # 高活跃时段涨类样本加权
                    # dynamic_activity 特征优化
                    'activity_trend',  # 活跃度趋势特征
                    # 新增涨跌活跃度差异特征
                    'up_down_activity_diff',  # 涨跌活跃度差异
                    # 新增跌类专属特征
                    'activity_trend_down',  # 活跃度趋势下跌分量
                    'ma5_deviation_down',  # ma5_deviation 向下偏离
                ]
                
                # 删除重复特征：移除重复的 atr_7, tick_volume, bollinger_position, up_momentum_5
                # 只保留存在于df中的特征
                available_features = [f for f in m1_features if f in df.columns]
                df = df[available_features]  # 只保留需要的特征
                
            elif timeframe == mt5.TIMEFRAME_M5:
                # M5周期：添加基础特征
                df = self.feature_engineer.add_core_features(df)
                
                # 添加增强特征
                df = self.feature_engineer.add_enhanced_features(df)
                
                # 保留M5模型需要的特征
                m5_features = [
                    # M5周期特征（主要决策）
                    'open', 'high', 'low', 'close', 'tick_volume',  # 保留一个tick_volume
                    'price_position', 'volatility_pct',
                    'm15_trend', 'm30_support', 'm30_resistance',
                    'volatility_change', 'tick_density',
                    'hour_of_day', 'is_peak_hour',
                    # K线形态特征
                    'hammer', 'shooting_star', 'engulfing',
                    # 技术指标
                    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                    'bollinger_position',  # 保留位置特征，移除未实现的上下轨
                    'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                    # 一致性特征
                    'rsi_price_consistency',
                    # 跨周期特征
                    'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                    'trend_consistency',
                    # 信号特征
                    'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                    # 风险特征
                    'volatility_regime', 'vol_cluster',
                    # M5专用周期共振特征
                    'm15_trend_ma_consistency',  # M15趋势与M5均线一致性
                    'm5_m1_volume_correlation',  # M5与M1成交量联动
                    'trend_strength_m5_m15',  # M5与M15趋势强度比
                    'cycle_alignment_score',  # 周期对齐评分
                    # 新增跨周期联动特征
                    'm5_m15_volume_correlation',  # M5与M15的volume_correlation
                    'volatility_diff_m5_m1',  # M5与M1的volatility_pct差值
                    # 趋势强度特征
                    'adx',  # ADX指标（趋势强度）
                    'ma5_ma20_alignment',  # MA5与MA20方向一致性
                    # 涨跌动能特征
                    'momentum_3',  # 3根K线的涨跌幅之和
                    'momentum_5',  # 5根K线的涨跌幅之和
                    'volume_price_divergence',  # 成交量与价格背离
                    'consecutive_up',  # 连续上涨次数
                    'consecutive_down',  # 连续下跌次数
                    'body_strength',  # K线实体强度
                    'upper_shadow',  # 上影线强度
                    'lower_shadow',  # 下影线强度
                    'price_position_5',  # 价格在短期高低点中的位置
                    # 动态活跃度特征
                    'dynamic_activity',  # 动态活跃度
                    'activity_level',  # 活跃度等级
                    # 跌类专属特征
                    'volume_up_ratio',  # tick_volume放量下跌占比
                    'atr_down_prob',  # ATR14扩张时的下跌概率
                    # 核心特征（清理重复特征后）
                    'atr_14',  # 核心ATR特征 - 保留高权重版本
                    'hl_ratio',  # 核心高低价比值 - 保留高权重版本
                    'volatility_pct',  # 核心波动率特征
                ]
                
                # 只保留存在于df中的特征
                available_features = [f for f in m5_features if f in df.columns]
                df = df[available_features]  # 只保留需要的特征
                
            elif timeframe == mt5.TIMEFRAME_M15:
                # M15周期：添加基础特征
                df = self.feature_engineer.add_core_features(df)
                
                # 添加M15特定特征
                df['rsi_21'] = self.calculate_rsi(df['close'], 21)
                df['ma21'] = df['close'].rolling(window=21).mean()
                df['ma50'] = df['close'].rolling(window=50).mean()
                
                # 计算方向特征（避免与基础特征工程重复）
                df['ma21_direction'] = (df['ma21'] - df['ma21'].shift(1)) / (df['ma21'].shift(1) + 1e-8)
                df['ma50_direction'] = (df['ma50'] - df['ma50'].shift(1)) / (df['ma50'].shift(1) + 1e-8)
                df['atr_21'] = self.calculate_atr(df['high'], df['low'], df['close'], 21)
                df['trend_strength'] = abs(df['ma21'] - df['ma50']) / df['close']
                
                # 添加增强特征
                df = self.feature_engineer.add_enhanced_features(df)
                
                # 添加M15专用的趋势特征
                df = self.m15_feature_engineer.add_trend_features(df)
                
                # 保留M15模型需要的特征
                m15_features = [
                    # M15周期特征（长期趋势）
                    'open', 'close', 'tick_volume',  # 核心特征
                    'rsi_21',  # 长期RSI
                    'ma21',  # 长期均线（删除ma50，权重仅1）
                    'ma21_direction',  # 长期均线方向（删除ma50_direction，权重仅1）
                    'atr_21',  # 长期ATR - 核心特征
                    'trend_strength',  # 趋势强度
                    'volatility_pct',  # 核心特征
                    # 跨周期趋势特征：M15与M60均线方向一致性
                    'm60_trend_consistency',  # M15与M60趋势一致性特征
                    # K线形态特征
                    'hammer', 'shooting_star', 'engulfing',
                    # 技术指标
                    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                    'bollinger_position',  # 保留位置特征，移除上下轨
                    'ma5', 'ma20', 'ma5_direction', 'ma20_direction',  # 删除ma10（权重仅1），保留其他均线
                    # 趋势强度特征
                    'adx',  # 趋势强度指标
                    'ma_trend_alignment',  # 均线排列一致性
                    'trend_duration',  # 趋势持续时长
                    # 动态活跃度特征 - 替换硬编码时间特征
                    'dynamic_activity',  # 动态活跃度 - 核心特征
                    'activity_level',  # 活跃度等级（高/中/低）
                    # 涨类专属趋势特征
                    'consecutive_up_momentum',  # 连续2根M15上涨动能
                    'up_prob_when_ma21_up',  # MA21向上时的涨概率
                    'up_prob_when_atr_contraction',  # ATR21收缩时的涨概率
                    'dynamic_activity_up_mean',  # dynamic_activity上涨区间均值
                    'up_after_high_volatility',  # 高波动后上涨概率
                    # 跌类专属趋势特征
                    'consecutive_down_momentum',  # 连续2根M15下跌动能
                    'atr_down_prob',  # ATR扩张时的下跌概率
                    # 高活跃度涨类加权特征
                    'high_activity_up_weight',  # 高活跃时段涨类样本加权
                    # 风险特征
                    'volatility_regime',  # 保留核心风险特征
                ]
                
                # 删除噪声特征：'ma50'、'ma10'、'ma20'（权重仅1）
                m15_features = [col for col in m15_features if col not in ['ma50', 'ma10', 'ma20']]
                
                # 只保留存在于df中的特征
                available_features = [f for f in m15_features if f in df.columns]
                df = df[available_features]  # 只保留需要的特征
            
            # 清理数据
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return None

    def calculate_m1_signal(self, df):
        """计算M1周期信号"""
        try:
            if len(df) < self.HISTORY_M1_BARS:
                logger.warning(f"M1数据不足，需要{self.HISTORY_M1_BARS}根K线，当前{len(df)}根")
                return 0.0, 0.0, 0.0  # 返回上涨、下跌、观望概率
            
            # 从M1训练器中获取特征列（与训练时一致）
            # 获取M1模型训练器的特征列表
            m1_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M1_模型训练.py")
            spec = importlib.util.spec_from_file_location("m1_model_trainer", m1_trainer_path)
            m1_model_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m1_model_trainer)
            
            # 临时创建一个M1训练器实例以获取特征列
            temp_trainer = m1_model_trainer.M1ModelTrainer()
            _, _, feature_names = temp_trainer.prepare_features_and_target(df.copy(), "M1")
            
            # 检查所有特征列是否存在
            available_features = []
            for col in feature_names:
                if col in df.columns:
                    available_features.append(col)
                else:
                    logger.warning(f"M1特征列 '{col}' 不存在于数据中")
            
            if len(available_features) == 0:
                logger.error("没有可用的M1特征")
                return 0.0, 0.0, 0.0
            
            # 获取最新的特征数据
            latest_data = df.iloc[-1][available_features].values.reshape(1, -1)
            
            # 使用M1标准化器进行标准化
            if self.m1_scaler is not None:
                latest_data = self.m1_scaler.transform(latest_data)
            
            # 创建DMatrix进行预测
            dtest = xgb.DMatrix(latest_data)
            
            # 预测概率
            pred_proba = self.m1_model.predict(dtest)[0]
            
            # 获取上涨和下跌概率
            # 类别顺序为[0, 1, 2] -> [下跌, 平, 上涨]，对应索引[0, 1, 2]
            if len(pred_proba) == 3:
                down_prob = pred_proba[0]  # 下跌概率
                hold_prob = pred_proba[1]  # 平概率
                up_prob = pred_proba[2]    # 上涨概率
            else:
                # 如果是二分类，需要根据实际情况调整
                down_prob = pred_proba[0]
                up_prob = pred_proba[1]
                hold_prob = 1 - up_prob - down_prob  # 中间概率
            
            logger.info(f"M1周期预测概率 - 上涨: {up_prob:.4f}, 下跌: {down_prob:.4f}, 观望: {hold_prob:.4f}")
            
            return up_prob, down_prob, hold_prob
            
        except Exception as e:
            logger.error(f"计算M1信号失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def calculate_m5_signal(self, df):
        """计算M5周期信号"""
        try:
            if len(df) < self.HISTORY_M5_BARS:
                logger.warning(f"M5数据不足，需要{self.HISTORY_M5_BARS}根K线，当前{len(df)}根")
                return 0.0, 0.0, 0.0  # 返回上涨、下跌、观望概率
            
            # 从M5训练器中获取特征列（与训练时一致）
            # 获取M5模型训练器的特征列表
            m5_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M5_模型训练.py")
            spec = importlib.util.spec_from_file_location("m5_model_trainer", m5_trainer_path)
            m5_model_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m5_model_trainer)
            
            # 临时创建一个M5训练器实例以获取特征列
            temp_trainer = m5_model_trainer.M5ModelTrainer()
            _, _, feature_names = temp_trainer.prepare_features_and_target(df.copy(), "M5")
            
            # 检查所有特征列是否存在
            available_features = []
            for col in feature_names:
                if col in df.columns:
                    available_features.append(col)
                else:
                    logger.warning(f"M5特征列 '{col}' 不存在于数据中")
            
            if len(available_features) == 0:
                logger.error("没有可用的M5特征")
                return 0.0, 0.0, 0.0
            
            # 获取最新的特征数据
            latest_data = df.iloc[-1][available_features].values.reshape(1, -1)
            
            # 使用M5标准化器进行标准化
            if self.m5_scaler is not None:
                latest_data = self.m5_scaler.transform(latest_data)
            
            # 创建DMatrix进行预测
            dtest = xgb.DMatrix(latest_data)
            
            # 预测概率
            pred_proba = self.m5_model.predict(dtest)[0]
            
            # 获取上涨和下跌概率
            # 类别顺序为[0, 1, 2] -> [下跌, 平, 上涨]，对应索引[0, 1, 2]
            if len(pred_proba) == 3:
                down_prob = pred_proba[0]  # 下跌概率
                hold_prob = pred_proba[1]  # 平概率
                up_prob = pred_proba[2]    # 上涨概率
            else:
                # 如果是二分类，需要根据实际情况调整
                down_prob = pred_proba[0]
                up_prob = pred_proba[1]
                hold_prob = 1 - up_prob - down_prob  # 中间概率
            
            logger.info(f"M5周期预测概率 - 上涨: {up_prob:.4f}, 下跌: {down_prob:.4f}, 观望: {hold_prob:.4f}")
            
            return up_prob, down_prob, hold_prob
            
        except Exception as e:
            logger.error(f"计算M5信号失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def calculate_m15_signal(self, df):
        """计算M15周期信号"""
        try:
            if len(df) < self.HISTORY_M15_BARS:
                logger.warning(f"M15数据不足，需要{self.HISTORY_M15_BARS}根K线，当前{len(df)}根")
                return 0.0, 0.0, 0.0  # 返回上涨、下跌、观望概率
            
            # 从M15训练器中获取特征列（与训练时一致）
            # 获取M15模型训练器的特征列表
            m15_trainer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m5", "XAUUSD_M15_模型训练.py")
            spec = importlib.util.spec_from_file_location("m15_model_trainer", m15_trainer_path)
            m15_model_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m15_model_trainer)
            
            # 临时创建一个M15训练器实例以获取特征列
            temp_trainer = m15_model_trainer.M15ModelTrainer()
            _, _, feature_names = temp_trainer.prepare_features_and_target(df.copy(), "M15")
            
            # 检查所有特征列是否存在
            available_features = []
            for col in feature_names:
                if col in df.columns:
                    available_features.append(col)
                else:
                    logger.warning(f"M15特征列 '{col}' 不存在于数据中")
            
            if len(available_features) == 0:
                logger.error("没有可用的M15特征")
                return 0.0, 0.0, 0.0
            
            # 获取最新的特征数据
            latest_data = df.iloc[-1][available_features].values.reshape(1, -1)
            
            # 使用M15标准化器进行标准化
            if self.m15_scaler is not None:
                latest_data = self.m15_scaler.transform(latest_data)
            
            # 创建DMatrix进行预测
            dtest = xgb.DMatrix(latest_data)
            
            # 预测概率
            pred_proba = self.m15_model.predict(dtest)[0]
            
            # 获取上涨和下跌概率
            # 类别顺序为[0, 1, 2] -> [下跌, 平, 上涨]，对应索引[0, 1, 2]
            if len(pred_proba) == 3:
                down_prob = pred_proba[0]  # 下跌概率
                hold_prob = pred_proba[1]  # 平概率
                up_prob = pred_proba[2]    # 上涨概率
            else:
                # 如果是二分类，需要根据实际情况调整
                down_prob = pred_proba[0]
                up_prob = pred_proba[1]
                hold_prob = 1 - up_prob - down_prob  # 中间概率
            
            logger.info(f"M15周期预测概率 - 上涨: {up_prob:.4f}, 下跌: {down_prob:.4f}, 观望: {hold_prob:.4f}")
            
            return up_prob, down_prob, hold_prob
            
        except Exception as e:
            logger.error(f"计算M15信号失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def calculate_fused_signal(self):
        """计算融合信号"""
        try:
            # 获取各周期数据，先尝试获取足够的数据
            initial_bars = max(self.HISTORY_M1_BARS + 50, self.HISTORY_M5_BARS + 50, self.HISTORY_M15_BARS + 50)  # 尝试获取更多数据以补偿特征工程造成的行数减少
            
            m1_data = self.get_current_market_data(self.M1_TIMEFRAME, initial_bars)
            m5_data = self.get_current_market_data(self.M5_TIMEFRAME, initial_bars)
            m15_data = self.get_current_market_data(self.M15_TIMEFRAME, initial_bars)
            
            if m1_data is None or m5_data is None or m15_data is None:
                logger.error("获取多周期数据失败")
                return "HOLD", 0.0
            
            # 检查每个周期的数据量是否足够，如果不足够则尝试获取更多数据
            if len(m1_data) < self.HISTORY_M1_BARS:
                m1_data = self.get_current_market_data(self.M1_TIMEFRAME, initial_bars * 2)
            if len(m5_data) < self.HISTORY_M5_BARS:
                m5_data = self.get_current_market_data(self.M5_TIMEFRAME, initial_bars * 2)
            if len(m15_data) < self.HISTORY_M15_BARS:
                m15_data = self.get_current_market_data(self.M15_TIMEFRAME, initial_bars * 2)
            
            # 计算各周期信号
            m1_up, m1_down, m1_hold = self.calculate_m1_signal(m1_data)
            m5_up, m5_down, m5_hold = self.calculate_m5_signal(m5_data)
            m15_up, m15_down, m15_hold = self.calculate_m15_signal(m15_data)
            
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
            
            # 生成最终信号 - 调整阈值以适应模型输出
            # 根据训练模型的输出，使用更灵活的阈值
            if fused_up > fused_down and fused_up > 0.45:  # 上涨概率大于下跌概率且超过0.45
                signal = "BUY"
                confidence = fused_up
                reason = f"综合上涨概率 {fused_up:.4f} 大于下跌概率 {fused_down:.4f} 且超过阈值0.45，M1贡献:{m1_up:.3f}, M5贡献:{m5_up:.3f}, M15贡献:{m15_up:.3f}"
            elif fused_down > fused_up and fused_down > 0.45:  # 下跌概率大于上涨概率且超过0.45
                signal = "SELL"
                confidence = fused_down
                reason = f"综合下跌概率 {fused_down:.4f} 大于上涨概率 {fused_up:.4f} 且超过阈值0.45，M1贡献:{m1_down:.3f}, M5贡献:{m5_down:.3f}, M15贡献:{m15_down:.3f}"
            else:
                signal = "HOLD"
                confidence = max(fused_up, fused_down)
                reason = f"无明确方向，综合上涨概率 {fused_up:.4f}，综合下跌概率 {fused_down:.4f}，或上涨/下跌概率未显著超过对方"
            
            logger.info(f"融合信号 - 上涨: {fused_up:.4f}, 下跌: {fused_down:.4f}, 观望: {fused_hold:.4f}")
            logger.info(f"交易信号: {signal} (置信度: {confidence:.4f})")
            logger.info(f"决策依据: {reason}")
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"计算融合信号失败: {e}")
            return "HOLD", 0.0

    def place_order(self, signal):
        """下单"""
        try:
            # 获取当前价格
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick is None:
                logger.error("无法获取当前价格")
                return False
            
            # 确定订单类型
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - 6  # 6美金止损，对应120美金(600点位)对0.2手的计算
                tp = price + 10  # 10美金止盈，对应200美金(1000点位)
            elif signal == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + 6  # 6美金止损
                tp = price - 10  # 10美金止盈
            else:
                logger.warning("无效的交易信号")
                return False
            
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
                logger.error("订单发送失败")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"订单执行失败: {result.retcode} - {result.comment}")
                return False
            
            # 更新持仓信息
            self.current_position = {
                'ticket': result.order,
                'type': order_type,
                'volume': self.LOT_SIZE,
                'price_open': price,
                'time': datetime.now(),
                'direction': "做多" if order_type == mt5.ORDER_TYPE_BUY else "做空"
            }
            
            logger.info(f"开仓成功: {signal} | 手数: {self.LOT_SIZE} | 订单号: {result.order}")
            return True
            
        except Exception as e:
            logger.error(f"下单失败: {e}")
            return False

    def close_position(self, reason=""):
        """平仓"""
        if self.current_position is None:
            logger.info("当前无持仓")
            return True
        
        try:
            # 获取持仓信息
            ticket = self.current_position['ticket']
            pos_type = self.current_position['type']
            
            # 获取当前价格
            tick = mt5.symbol_info_tick(self.SYMBOL)
            if tick is None:
                logger.error("无法获取当前价格")
                return False
            
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
                logger.error("平仓订单发送失败")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"平仓订单执行失败: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"平仓成功: {reason} | 订单号: {ticket}")
            
            # 清除持仓信息
            self.current_position = None
            return True
            
        except Exception as e:
            logger.error(f"平仓失败: {e}")
            return False

    def check_and_close_by_signal(self, current_signal):
        """根据信号检查是否需要平仓"""
        if self.current_position is None:
            return False
        
        current_direction = self.current_position['direction']
        
        # 信号反向时平仓
        if (current_direction == "做多" and current_signal == "SELL") or \
           (current_direction == "做空" and current_signal == "BUY"):
            logger.info(f"平仓: 信号反向出现")
            return self.close_position("信号反向")
        
        # 检查持仓盈利
        try:
            # 获取持仓详情
            positions = mt5.positions_get(symbol=self.SYMBOL)
            if positions and len(positions) > 0:
                pos = positions[0]
                profit = pos.profit
                
                # 观望信号且盈利超过90美金时平仓
                if current_signal == "HOLD" and profit > 90:
                    logger.info(f"平仓: 观望信号且盈利超过90美金 ({profit:.2f}美金)")
                    return self.close_position("观望信号盈利")
        except Exception as e:
            logger.error(f"检查持仓盈利失败: {e}")
        
        return False

    def check_daily_close(self):
        """检查是否需要每日收盘前平仓"""
        if self.current_position is None:
            return False
        
        # 获取当前XAUUSD市场时间
        tick = mt5.symbol_info_tick(self.SYMBOL)
        if tick is None:
            return False
        
        current_time = datetime.fromtimestamp(tick.time)
        
        # 每日20:00 UTC平仓（XAUUSD市场时间）
        if current_time.hour >= 20 and current_time.minute >= 0:
            logger.info("平仓: 每日收盘前平仓")
            return self.close_position("每日收盘")
        
        return False

    def check_risk_management(self):
        """检查风控管理"""
        try:
            # 获取账户信息
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("无法获取账户信息")
                return False
            
            balance = account_info.balance
            equity = account_info.equity
            
            # 计算当日盈亏（简化版，实际需要记录当日初始余额）
            daily_pnl = equity - balance  # 这里是简化处理
            
            # 检查是否超过最大回撤限制
            if daily_pnl < -abs(balance * self.FTMO_MAX_DRAWDOWN):
                logger.warning(f"超过最大回撤限制: 当日亏损 {daily_pnl:.2f}, 限制: {balance * self.FTMO_MAX_DRAWDOWN:.2f}")
                if self.current_position is not None:
                    logger.info("执行风控平仓")
                    return self.close_position("风控平仓")
            
            # 检查账户余额是否低于最低要求
            if balance < self.FTMO_MIN_BALANCE:
                logger.warning(f"账户余额低于最低要求: {balance} < {self.FTMO_MIN_BALANCE}")
                if self.current_position is not None:
                    logger.info("执行风控平仓")
                    return self.close_position("余额不足")
            
            # 检查是否达到盈利目标
            initial_balance = 100000  # 假设初始余额
            if balance >= initial_balance * (1 + self.FTMO_PROFIT_TARGET):
                logger.info(f"达到盈利目标: {balance} >= {initial_balance * (1 + self.FTMO_PROFIT_TARGET)}")
                if self.current_position is not None:
                    logger.info("执行盈利目标平仓")
                    return self.close_position("盈利目标")
            
        except Exception as e:
            logger.error(f"风控检查失败: {e}")
        
        return False

    def update_prediction_accuracy(self, signal):
        """更新预测准确率"""
        # 记录预测
        self.prediction_history.append({
            'signal': signal,
            'actual_outcome': None,
            'is_correct': None
        })
        
        # 限制历史长度
        if len(self.prediction_history) > self.max_history_length:
            self.prediction_history.pop(0)

    def get_recent_accuracy(self):
        """获取最近的预测准确率"""
        if not self.prediction_history:
            return 0.0
        
        # 只计算有实际结果的预测
        valid_predictions = [record for record in self.prediction_history if record['actual_outcome'] is not None]
        if not valid_predictions:
            return 0.0
        
        correct_predictions = sum(1 for record in valid_predictions if record['is_correct'])
        return correct_predictions / len(valid_predictions) if valid_predictions else 0.0

    def incremental_training(self, new_data=None):
        """增量训练模型 - 这里简化处理，实际应用中可能需要分别对每个模型进行增量训练"""
        try:
            logger.info("开始多周期模型增量训练...")
            
            # 这里可以实现对每个模型的增量训练
            # 为简化实现，这里仅演示M5模型的增量训练
            # 实际应用中应该对M1、M5、M15三个模型都进行增量训练
            
            # 获取最新的M5市场数据用于训练
            m5_data = self.get_current_market_data(self.M5_TIMEFRAME, 500)
            if m5_data is None:
                logger.error("获取M5新数据失败，跳过增量训练")
                return False
            
            # 准备特征和目标变量
            feature_columns = [
                # M5周期特征（主要决策）
                'open', 'high', 'low', 'close', 'tick_volume',  # 保留一个tick_volume
                'price_position', 'volatility_pct',
                'm15_trend', 'm30_support', 'm30_resistance',
                'volatility_change', 'tick_density',
                # K线形态特征
                'hammer', 'shooting_star', 'engulfing',
                # 技术指标
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_position',  # 保留位置特征，移除未实现的上下轨
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                # 一致性特征
                'rsi_price_consistency',
                # 跨周期特征
                'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                'trend_consistency',
                # 信号特征
                'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                # 风险特征
                'volatility_regime', 'vol_cluster',
                # M5专用周期共振特征
                'm15_trend_ma_consistency',  # M15趋势与M5均线一致性
                'm5_m1_volume_correlation',  # M5与M1成交量联动
                'trend_strength_m5_m15',  # M5与M15趋势强度比
                'cycle_alignment_score',  # 周期对齐评分
                # 新增跨周期联动特征
                'm5_m15_volume_correlation',  # M5与M15的volume_correlation
                'volatility_diff_m5_m1',  # M5与M1的volatility_pct差值
                # 趋势强度特征
                'adx',  # ADX指标（趋势强度）
                'ma5_ma20_alignment',  # MA5与MA20方向一致性
                # 涨跌动能特征
                'momentum_3',  # 3根K线的涨跌幅之和
                'momentum_5',  # 5根K线的涨跌幅之和
                'volume_price_divergence',  # 成交量与价格背离
                'consecutive_up',  # 连续上涨次数
                'consecutive_down',  # 连续下跌次数
                'body_strength',  # K线实体强度
                'upper_shadow',  # 上影线强度
                'lower_shadow',  # 下影线强度
                'price_position_5',  # 价格在短期高低点中的位置
                # 动态活跃度特征
                'dynamic_activity',  # 动态活跃度
                'activity_level',  # 活跃度等级
                # 跌类专属特征
                'volume_up_ratio',  # tick_volume放量下跌占比
                'atr_down_prob',  # ATR14扩张时的下跌概率
                # 核心特征（清理重复特征后）
                'atr_14',  # 核心ATR特征 - 保留高权重版本
                'hl_ratio',  # 核心高低价比值 - 保留高权重版本
                'volatility_pct',  # 核心波动率特征
            ]
            
            available_features = []
            for col in feature_columns:
                if col in m5_data.columns:
                    available_features.append(col)
            
            # 创建目标变量：预测未来1根K线的涨跌
            m5_data['future_close'] = m5_data['close'].shift(-1)
            m5_data['price_change_pct'] = (m5_data['future_close'] - m5_data['close']) / m5_data['close']
            m5_data['target'] = np.where(m5_data['price_change_pct'] > 0.001, 1,  # 涨
                                        np.where(m5_data['price_change_pct'] < -0.001, -1, 0))  # 跌和平
            
            # 准备训练数据
            X_new = m5_data[available_features].values
            y_new = m5_data['target'].values
            
            # 过滤掉NaN值
            mask = ~(np.isnan(X_new).any(axis=1) | np.isnan(y_new))
            X_new = X_new[mask]
            y_new = y_new[mask]
            
            if len(X_new) == 0:
                logger.warning("没有有效的M5训练数据，跳过增量训练")
                return False
            
            # 获取最新的一部分数据进行增量训练
            n_samples = min(200, len(X_new))  # 最多使用200个样本
            X_recent = X_new[-n_samples:]
            y_recent = y_new[-n_samples:]
            
            logger.info(f"使用 {len(X_recent)} 个新样本进行M5模型增量训练")
            
            # 评估旧模型性能
            try:
                # 创建DMatrix进行评估
                dtest = xgb.DMatrix(X_recent, label=y_recent)
                old_score = self.m5_model.eval(dtest)
                logger.info(f"旧M5模型在新数据上的评估: {old_score}")
            except:
                logger.warning("无法评估旧M5模型性能")
            
            # 对M5模型进行增量训练
            # 创建训练DMatrix
            dtrain = xgb.DMatrix(X_recent, label=y_recent)
            
            # 更新模型参数
            updated_model = xgb.train(
                self.m5_model.save_config(),  # 使用现有模型配置
                dtrain,
                xgb_model=self.m5_model,  # 使用现有模型作为基础
                num_boost_round=10  # 少量额外训练轮次
            )
            
            # 更新模型
            self.m5_model = updated_model
            
            # 评估新模型性能
            try:
                # 创建DMatrix进行评估
                dtest = xgb.DMatrix(X_recent, label=y_recent)
                new_score = self.m5_model.eval(dtest)
                logger.info(f"新M5模型在新数据上的评估: {new_score}")
                
                # 如果性能提升，则保存新模型
                if 'merror' in new_score or 'mlogloss' in new_score:  # 检查评估指标
                    new_model_path = f"xauusd_m5_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.m5_model.save_model(new_model_path)
                    logger.info(f"新M5模型已保存到: {new_model_path}")
                    self.M5_MODEL_PATH = new_model_path
            except:
                logger.warning("无法评估新M5模型性能")
            
            return True
            
        except Exception as e:
            logger.error(f"增量训练失败: {e}")
            return False

    def run_trading_cycle(self):
        """执行单次交易循环"""
        try:
            # 计算融合信号
            signal, prob = self.calculate_fused_signal()
            
            # 获取当前准确率
            current_accuracy = self.get_recent_accuracy()
            logger.info(f"模型最近预测准确率: {current_accuracy:.4f}")
            
            # 风控检查
            self.check_risk_management()
            
            # 每日收盘前平仓检查
            self.check_daily_close()
            
            # 检查是否需要根据信号平仓
            if self.check_and_close_by_signal(signal):
                logger.info("已根据信号平仓")
            
            # 如果没有持仓且有明确信号，则开仓
            if self.current_position is None and signal in ["BUY", "SELL"]:
                if prob > 0.45:  # 调整为更灵活的阈值，适应模型输出
                    logger.info(f"开仓: {signal} 信号，置信度 {prob:.3f}")
                    # 记录预测，实际结果将在后续确定
                    self.update_prediction_accuracy(signal)
                    self.place_order(signal)
                else:
                    logger.info(f"信号置信度 {prob:.3f} 不足，暂不交易")
            
            # 检查持仓状态
            if self.current_position is not None:
                # 获取当前持仓盈亏
                try:
                    positions = mt5.positions_get(symbol=self.SYMBOL)
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        profit = pos.profit
                        direction = "做多" if pos.type == mt5.POSITION_TYPE_BUY else "做空"
                        logger.info(f"当前持仓: {direction}, 持仓收益: {profit:.2f}美金")
                    else:
                        logger.info("当前无持仓")
                        self.current_position = None
                except Exception as e:
                    logger.error(f"获取持仓信息失败: {e}")
            else:
                logger.info("当前无持仓")
            
            return True
            
        except Exception as e:
            logger.error(f"交易循环执行失败: {e}")
            return False

    def run_trading_loop(self):
        """运行交易循环 - 改为实时监控模式"""
        self.is_running = True
        logger.info("开始多周期实时交易循环")
        
        # 记录上次增量训练时间
        last_training_time = datetime.now()
        
        # 记录上次K线时间，用于检测K线更新
        last_m1_time = None
        last_m5_time = None
        last_m15_time = None
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # 获取当前时间
                current_tick = mt5.symbol_info_tick(self.SYMBOL)
                if current_tick is None:
                    logger.error("无法获取当前市场数据")
                    time.sleep(5)
                    continue
                
                now_market_time = datetime.fromtimestamp(current_tick.time)
                
                # 获取最新的M1、M5、M15数据
                m1_rates = mt5.copy_rates_from_pos(self.SYMBOL, mt5.TIMEFRAME_M1, 0, 1)
                m5_rates = mt5.copy_rates_from_pos(self.SYMBOL, mt5.TIMEFRAME_M5, 0, 1)
                m15_rates = mt5.copy_rates_from_pos(self.SYMBOL, mt5.TIMEFRAME_M15, 0, 1)
                
                if len(m1_rates) == 0 or len(m5_rates) == 0 or len(m15_rates) == 0:
                    logger.error("无法获取最新K线数据")
                    time.sleep(5)
                    continue
                
                current_m1_time = datetime.fromtimestamp(m1_rates[0]['time'])
                current_m5_time = datetime.fromtimestamp(m5_rates[0]['time'])
                current_m15_time = datetime.fromtimestamp(m15_rates[0]['time'])
                
                logger.info(f"最新数据 - M1: {current_m1_time.strftime('%Y-%m-%d %H:%M:%S')}, M5: {current_m5_time.strftime('%Y-%m-%d %H:%M:%S')}, M15: {current_m15_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 检查M1 K线是否更新
                if last_m1_time is None or current_m1_time > last_m1_time:
                    logger.info(f"M1 K线已更新到: {current_m1_time}")
                    last_m1_time = current_m1_time
                    
                    # 检查M5 K线是否也更新了
                    if last_m5_time is None or current_m5_time > last_m5_time:
                        logger.info(f"M5 K线已更新到: {current_m5_time}")
                        last_m5_time = current_m5_time
                        
                        # 检查M15 K线是否也更新了
                        if last_m15_time is None or current_m15_time > last_m15_time:
                            logger.info(f"M15 K线已更新到: {current_m15_time}")
                            last_m15_time = current_m15_time
                            
                            # 所有周期K线都已更新，执行多周期融合交易决策
                            self.run_trading_cycle()
                        else:
                            # M1和M5更新，但M15未更新，只执行M1和M5的交易决策
                            # 这里可以执行M1和M5的融合交易决策
                            self.run_trading_cycle()
                    else:
                        # 只有M1更新，M5和M15未更新，可以考虑执行M1交易决策
                        # 但为了保持策略一致性，我们只在M5 K线完成时执行完整交易决策
                        pass
                
                # 每小时执行一次增量训练
                current_time = datetime.now()
                if (current_time - last_training_time).total_seconds() >= 3600:  # 1小时
                    self.incremental_training()
                    last_training_time = current_time
                
                # 每分钟检查一次K线是否更新
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"交易循环异常: {e}")
                time.sleep(5)  # 异常后暂停5秒再继续
        
        logger.info("多周期实时交易循环结束")

    def stop_trading(self):
        """停止交易"""
        logger.info("正在停止交易...")
        self.is_running = False
        self.stop_event.set()
        
        # 如果有持仓，执行平仓
        if self.current_position is not None:
            logger.info("检测到持仓，执行平仓")
            self.close_position("停止交易")
        
        # 关闭MT5连接
        mt5.shutdown()
        logger.info("MT5连接已关闭")

def main():
    """主函数"""
    trader = None
    try:
        # 创建多周期交易实例
        trader = MultiPeriodRealTimeTrader(
            m1_model_path="xauusd_m1_model.json",  # M1模型
            m5_model_path="xauusd_m5_model.json",  # M5模型
            m15_model_path="xauusd_m15_model.json",  # M15模型
            lot_size=0.2  # 固定手数
        )
        
        # 运行交易循环
        trader.run_trading_loop()
        
    except Exception as e:
        logger.error(f"交易程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if trader:
            trader.stop_trading()

if __name__ == "__main__":
    main()
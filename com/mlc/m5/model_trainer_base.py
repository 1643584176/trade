"""
多周期模型训练基类
提供通用的模型训练功能，减少代码重复
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys
import os
from datetime import datetime, timedelta, timezone
import warnings
import importlib.util
warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

# 动态导入特征工程模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
feature_engineering_path = os.path.join(project_root, 'mlc', 'common', 'm5_feature_engineering.py')
spec = importlib.util.spec_from_file_location("m5_feature_engineering", feature_engineering_path)
m5_feature_engineering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5_feature_engineering)
M5FeatureEngineer = m5_feature_engineering.M5FeatureEngineer


class BaseModelTrainer:
    """模型训练基类"""
    
    def __init__(self, symbol="XAUUSD", utc_tz=timezone.utc):
        self.SYMBOL = symbol
        self.UTC_TZ = utc_tz
        self.feature_engineer = M5FeatureEngineer()
    
    def initialize_mt5(self):
        """初始化MT5连接"""
        if not mt5.initialize():
            raise Exception(f"MT5初始化失败：{mt5.last_error()}")
        
        # 确保交易品种被选中
        if not mt5.symbol_select(self.SYMBOL, True):
            raise Exception(f"无法选择交易品种 {self.SYMBOL}")
    
    def calculate_rsi(self, prices, window=14):
        """计算RSI"""
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
    
    def prepare_data_with_features(self, rates, timeframe_type="M5"):
        """准备数据并添加特征"""
        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # 使用特征工程类添加特征
        df = self.feature_engineer.add_core_features(df)
        
        # 根据时间周期添加特定特征
        if timeframe_type == "M1":
            # M1特有特征
            df['rsi_7'] = self.calculate_rsi(df['close'], 7)  # 更短期的RSI
            df['ma3'] = df['close'].rolling(window=3).mean()
            df['ma7'] = df['close'].rolling(window=7).mean()
            df['ma3_direction'] = self.calculate_direction(df['ma3'])
            df['ma7_direction'] = self.calculate_direction(df['ma7'])
            df['atr_7'] = self.calculate_atr(df['high'], df['low'], df['close'], 7)
        elif timeframe_type == "M5":
            # M5周期已通过feature_engineer处理
            pass
        elif timeframe_type == "M15":
            # M15特有特征
            df['rsi_21'] = self.calculate_rsi(df['close'], 21)  # 长期RSI
            df['ma21'] = df['close'].rolling(window=21).mean()  # 长期均线
            df['ma50'] = df['close'].rolling(window=50).mean()  # 更长期均线
            df['ma21_direction'] = self.calculate_direction(df['ma21'])
            df['ma50_direction'] = self.calculate_direction(df['ma50'])
            df['atr_21'] = self.calculate_atr(df['high'], df['low'], df['close'], 21)
            df['trend_strength'] = abs(df['ma21'] - df['ma50']) / df['close']
        
        # 清理数据
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def prepare_features_and_target(self, df, timeframe_type="M5"):
        """准备特征和目标变量"""
        # 根据时间周期选择特征列
        if timeframe_type == "M1":
            feature_columns = [
                # M1周期特征（短期波动）
                'open', 'high', 'low', 'close', 'tick_volume',
                'rsi_7',  # 短期RSI
                'ma3', 'ma7',  # 短期均线
                'ma3_direction', 'ma7_direction',  # 短期均线方向
                'atr_7',  # 短期ATR
                'volatility_pct',
                'hour_of_day', 'is_peak_hour',
                # K线形态特征
                'hammer', 'shooting_star', 'engulfing',
                # 技术指标
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_upper', 'bollinger_lower', 'bollinger_position',
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                # 一致性特征
                'ma_direction_consistency', 'rsi_price_consistency',
                # 跨周期特征
                'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                'trend_consistency',
                # 信号特征
                'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                # 风险特征
                'volatility_regime', 'vol_cluster'
            ]
        elif timeframe_type == "M5":
            feature_columns = [
                # M5周期特征（主要决策）
                'open', 'high', 'low', 'close', 'tick_volume',
                'price_position', 'atr_14', 'volatility_pct', 'hl_ratio',
                'm15_trend', 'm30_support', 'm30_resistance',
                'spread_change', 'volatility_change', 'tick_density',
                'hour_of_day', 'is_peak_hour',
                # K线形态特征
                'hammer', 'shooting_star', 'engulfing',
                # 技术指标
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_upper', 'bollinger_lower', 'bollinger_position',
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                # 一致性特征
                'ma_direction_consistency', 'rsi_price_consistency',
                # 跨周期特征
                'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                'trend_consistency',
                # 信号特征
                'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                # 风险特征
                'volatility_regime', 'vol_cluster'
            ]
        elif timeframe_type == "M15":
            feature_columns = [
                # M15周期特征（长期趋势）
                'open', 'high', 'low', 'close', 'tick_volume',
                'rsi_21',  # 长期RSI
                'ma21', 'ma50',  # 长期均线
                'ma21_direction', 'ma50_direction',  # 长期均线方向
                'atr_21',  # 长期ATR
                'trend_strength',  # 趋势强度
                'volatility_pct',
                'hour_of_day', 'is_peak_hour',
                # K线形态特征
                'hammer', 'shooting_star', 'engulfing',
                # 技术指标
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_upper', 'bollinger_lower', 'bollinger_position',
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                # 一致性特征
                'ma_direction_consistency', 'rsi_price_consistency',
                # 跨周期特征
                'rsi_divergence', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long',
                'trend_consistency',
                # 信号特征
                'rsi_signal_strength', 'macd_signal_strength', 'short_long_signal_consistency',
                # 风险特征
                'volatility_regime', 'vol_cluster'
            ]
        
        # 检查所有特征列是否存在
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                available_features.append(col)
            else:
                print(f"警告: 特征列 '{col}' 不存在")
        
        X = df[available_features].values
        y = df['target'].values
        
        return X, y, available_features

    def encode_target_labels(self, y):
        """将目标变量编码为从0开始的连续整数"""
        # 将[-1, 0, 1]转换为[0, 1, 2]
        y_encoded = y.copy()
        y_encoded[y == -1] = 0  # 将-1转换为0
        y_encoded[y == 0] = 1   # 将0转换为1
        y_encoded[y == 1] = 2   # 将1转换为2
        
        return y_encoded

    def decode_target_labels(self, y_encoded):
        """将编码的目标变量转换回原始标签"""
        # 将[0, 1, 2]转换为[-1, 0, 1]
        y = y_encoded.copy()
        y[y_encoded == 0] = -1  # 将0转换为-1
        y[y_encoded == 1] = 0   # 将1转换为0
        y[y_encoded == 2] = 1   # 将2转换为1
        
        return y

    def train_xgboost_model(self, X_train, X_test, y_train, y_test, model_params=None):
        """训练XGBoost模型"""
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'learning_rate': 0.1,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
        
        # 对目标变量进行编码
        y_train_encoded = self.encode_target_labels(y_train)
        y_test_encoded = self.encode_target_labels(y_test)
        
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train_encoded)
        
        # 评估模型
        train_score = model.score(X_train, y_train_encoded)
        test_score = model.score(X_test, y_test_encoded)
        
        print(f"训练集准确率: {train_score:.4f}")
        print(f"测试集准确率: {test_score:.4f}")
        
        return model, train_score, test_score
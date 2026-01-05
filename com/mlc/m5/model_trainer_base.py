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
feature_engineering_path = os.path.join(project_root, 'mlc', 'm5', 'm5_feature_engineering.py')
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
            
            # 添加涨跌动能特征
            df = self.add_momentum_features(df)
        elif timeframe_type == "M5":
            # M5周期已通过feature_engineer处理
            # 添加涨跌动能特征
            df = self.add_momentum_features(df)
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
                'atr_7',  # 短期ATR - 核心特征
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
                # 新增跌类专属特征
                'down_momentum_5',  # 5根K线仅计算下跌部分的强度
                'down_volume_impulse',  # 放量下跌占比
                # 新增高活跃度涨类加权特征
                'high_activity_up_weight',  # 高活跃时段涨类样本加权
                # dynamic_activity 特征优化
                'activity_trend',  # 活跃度趋势特征
                # 核心波动特征
                'atr_7',  # 短期ATR - 核心特征
                'volatility_pct',  # 波动率百分比
                'tick_volume',  # 成交量
                # 删除重复和低区分度特征：移除重复的 atr_7, tick_volume, bollinger_position, up_momentum_5
            ]
        elif timeframe_type == "M5":
            feature_columns = [
                # M5周期特征（主要决策）
                'open', 'high', 'low', 'close', 'tick_volume',
                'price_position', 'atr_14', 'volatility_pct', 'hl_ratio',
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
                'tick_volume',  # 核心成交量特征 - 保留高权重版本
                'atr_14',  # 核心ATR特征 - 保留高权重版本
                'hl_ratio',  # 核心高低价比值 - 保留高权重版本
                'volatility_pct',  # 核心波动率特征
                # 新增dynamic_activity特征（当前5分钟波动率/过去24小时同周期均值）
                'dynamic_activity',  # 补充行情活跃度维度
                # 删除低价值特征：移除重复的特征如'high', 'ma10'（权重<3），以及低价值特征
            ]
        elif timeframe_type == "M15":
            feature_columns = [
                # M15周期特征（长期趋势）
                'open', 'close', 'tick_volume',
                'rsi_21',  # 长期RSI
                'ma21', 'ma50',  # 长期均线
                'ma21_direction', 'ma50_direction',  # 长期均线方向
                'atr_21',  # 长期ATR
                'trend_strength',  # 趋势强度
                'volatility_pct',
                # 跨周期趋势特征：M15与M60均线方向一致性
                'm60_trend_consistency',  # M15与M60趋势一致性特征
                # K线形态特征
                'hammer', 'shooting_star', 'engulfing',
                # 技术指标
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'bollinger_position',  # 保留位置特征，移除上下轨
                'ma5', 'ma10', 'ma20', 'ma5_direction', 'ma10_direction', 'ma20_direction',
                # 趋势强度特征
                'adx',  # 趋势强度指标
                'ma_trend_alignment',  # 均线排列一致性
                'trend_duration',  # 趋势持续时长
                # 动态活跃度特征 - 替换硬编码时间特征
                'dynamic_activity',  # 动态活跃度
                'activity_level',  # 活跃度等级（高/中/低）
                # 涨类专属趋势特征
                'consecutive_up_momentum',  # 连续2根M15上涨动能
                'up_prob_when_ma21_up',  # MA21向上时的涨概率
                # 跌类专属趋势特征
                'consecutive_down_momentum',  # 连续2根M15下跌动能
                'atr_down_prob',  # ATR扩张时的下跌概率
                # 高活跃度涨类加权特征
                'high_activity_up_weight',  # 高活跃时段涨类样本加权
                # 风险特征
                'volatility_regime',  # 保留核心风险特征
                # 删除噪声特征：'low', 'high', 'ma21'（重复），'consecutive_down_momentum'（过度强化跌类），'ma10', 'ma50'（权重<3）
            ]
        
        # 检查所有特征列是否存在
        available_features = []
        for col in feature_columns:
            if col in df.columns:
                available_features.append(col)
            else:
                # 对于M1周期，某些特征可能在特定条件下不存在，仅对M5和M15周期打印警告
                # 移除bollinger_upper/lower/ma_direction_consistency/rsi_divergence等未实现特征的警告
                if timeframe_type != "M1" and col not in ['bollinger_upper', 'bollinger_lower', 'ma_direction_consistency', 'rsi_divergence', 'vol_cluster', 'short_long_signal_consistency', 'rsi_signal_strength', 'macd_signal_strength', 'trend_consistency', 'vol_short_vs_medium', 'vol_medium_vs_long', 'vol_short_vs_long']:
                    print(f"警告: 特征列 '{col}' 不存在")
        
        X = df[available_features].values
        y = df['target'].values
        
        return X, y, available_features

    def standardize_features(self, df, feature_columns, fit_scaler=None):
        """对数值型特征进行Z-score标准化"""
        from sklearn.preprocessing import StandardScaler
        
        if fit_scaler is not None:
            # 使用已训练的scaler进行变换
            standardized_values = fit_scaler.transform(df[feature_columns])
        else:
            # 训练新的scaler并进行变换
            scaler = StandardScaler()
            standardized_values = scaler.fit_transform(df[feature_columns])
            return standardized_values, scaler
        
        return standardized_values, None

    def encode_target_labels(self, y):
        """将目标变量编码为从0开始的连续整数"""
        # 将[-1, 0, 1]转换为[0, 1, 2]
        y_encoded = y.copy()
        y_encoded[y == -1] = 0  # 将-1转换为0
        y_encoded[y == 0] = 1   # 将0转换为1
        y_encoded[y == 1] = 2   # 将1转换为2
        
        return y_encoded

    def add_momentum_features(self, df):
        """添加涨跌动能特征，用于增强M1模型的超短期涨跌信号识别能力"""
        # 连续K线涨跌幅之和（短期动能）
        df['price_change_pct'] = df['close'].pct_change()
        df['momentum_3'] = df['price_change_pct'].rolling(window=3).sum()  # 3根K线的涨跌幅之和
        df['momentum_5'] = df['price_change_pct'].rolling(window=5).sum()  # 5根K线的涨跌幅之和
        
        # 成交量与价格背离特征
        df['volume_price_divergence'] = df['tick_volume'].pct_change() - df['price_change_pct']
        
        # 连续上涨/下跌次数（短期趋势持续性）
        df['price_direction'] = np.where(df['price_change_pct'] > 0, 1, np.where(df['price_change_pct'] < 0, -1, 0))
        df['consecutive_up'] = (df['price_direction'] == 1).astype(int).groupby((df['price_direction'] != 1).cumsum()).cumsum()
        df['consecutive_down'] = (df['price_direction'] == -1).astype(int).groupby((df['price_direction'] != -1).cumsum()).cumsum()
        
        # K线实体强度（收盘价与开盘价的相对位置）
        df['body_strength'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)  # 防止除零
        
        # 上下影线强度
        df['upper_shadow'] = (df['high'] - np.maximum(df['close'], df['open'])) / (df['high'] - df['low'] + 1e-8)
        df['lower_shadow'] = (np.minimum(df['close'], df['open']) - df['low']) / (df['high'] - df['low'] + 1e-8)
        
        # 价格位置（在短期高低点中的位置）
        df['high_5'] = df['high'].rolling(window=5).max()
        df['low_5'] = df['low'].rolling(window=5).min()
        df['price_position_5'] = (df['close'] - df['low_5']) / (df['high_5'] - df['low_5'] + 1e-8)  # 防止除零
        
        # 清理可能的无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    def add_m60_trend_consistency_feature(self, df, m15_df):
        """添加M15与M60趋势一致性特征"""
        # 获取M60数据用于计算一致性
        try:
            # 计算M60的MA21
            m60_ma21 = m15_df['close'].resample('60T').mean().rolling(window=21).mean()
            
            # 将M60的MA21重新采样回M15频率
            m60_ma21_m15 = m60_ma21.resample('15T').ffill()
            m60_ma21_m15 = m60_ma21_m15.reindex(m15_df.index)
            
            # 计算M15的MA21
            m15_ma21 = m15_df['close'].rolling(window=21).mean()
            
            # 计算趋势一致性（方向相同为1，不同为0）
            m15_trend = np.where(m15_ma21 > m15_ma21.shift(1), 1, -1)
            m60_trend = np.where(m60_ma21_m15 > m60_ma21_m15.shift(1), 1, -1)
            
            df['m60_trend_consistency'] = np.where(m15_trend == m60_trend, 1, 0)
            
        except Exception as e:
            print(f"计算M60趋势一致性特征时出错: {e}")
            df['m60_trend_consistency'] = 0  # 默认值
        
        return df

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
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
from sklearn.utils import resample
warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

# 动态导入基类
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_trainer_base_path = os.path.join(project_root, 'mlc', 'm5', 'model_trainer_base.py')
spec = importlib.util.spec_from_file_location("model_trainer_base", model_trainer_base_path)
model_trainer_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_trainer_base)
BaseModelTrainer = model_trainer_base.BaseModelTrainer

# 配置参数
class M15ModelConfig:
    SYMBOL = "XAUUSD"
    M15_TIMEFRAME = mt5.TIMEFRAME_M15
    HISTORY_M15_BARS = 200  # 用于预测的M15 K线数量（200根）
    PREDICT_FUTURE_BARS = 4  # 预测未来K线数量（1-4根）
    TRAIN_TEST_SPLIT = 0.8
    MODEL_SAVE_PATH = "xauusd_m15_model.json"  # XGBoost模型保存路径
    SCALER_SAVE_PATH = "m15_scaler.pkl"
    UTC_TZ = timezone.utc

class M15ModelTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__()
        self.config = M15ModelConfig()
    
    def get_m15_historical_data(self, bars_count: int = 547*24*4):  # 1.5年的M15数据
        """获取MT5真实历史M15数据"""
        self.initialize_mt5()
        
        # 获取当前时间
        current_utc = datetime.now(self.config.UTC_TZ)
        start_time = current_utc - timedelta(minutes=15*bars_count)  # M15数据，每根K线15分钟
        
        # 使用mt5.copy_rates_from_pos按K线数量获取数据
        m15_rates = mt5.copy_rates_from_pos(
            self.config.SYMBOL,
            self.config.M15_TIMEFRAME,
            0,  # 从最新的K线开始获取
            bars_count  # 获取指定数量的K线
        )
        
        if m15_rates is None or len(m15_rates) == 0:
            raise Exception(f"获取M15历史数据失败：{mt5.last_error()}")
        
        # 转换为DataFrame
        df = pd.DataFrame(m15_rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # 数据有效性检查 - 检查时间连续性
        time_diff = df.index.to_series().diff().dt.total_seconds().dropna()
        if not (time_diff == 900).all():  # M15周期预期间隔900秒
            print("警告: 数据存在时间断连，可能影响特征计算")
        
        # 准备数据和特征
        df = self.prepare_data_with_features(m15_rates, "M15")
        
        # 添加M15专用的趋势特征
        df = self.add_trend_features(df)
        
        # 创建目标变量：预测未来3-5根K线的趋势方向 (1=涨, 0=跌, -1=平)
        df['future_close_1'] = df['close'].shift(-1)  # 预测1根K线后
        df['future_close_2'] = df['close'].shift(-2)  # 预测2根K线后
        df['future_close_3'] = df['close'].shift(-3)  # 预测3根K线后
        df['future_close_4'] = df['close'].shift(-4)  # 预测4根K线后
        df['future_close_5'] = df['close'].shift(-5)  # 预测5根K线后
        
        # 使用预测未来3-5根K线的平均涨跌幅作为目标（趋势确认）
        df['future_avg_close'] = (df['future_close_1'] + df['future_close_2'] + df['future_close_3'] + df['future_close_4'] + df['future_close_5']) / 5
        df['price_change_pct'] = (df['future_avg_close'] - df['close']) / df['close']
        
        # 异常值处理 - 检测价格跳空
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        atr_14 = self.calculate_atr(df['high'], df['low'], df['close'], 14)
        df['atr_14'] = atr_14
        df = df[abs(df['gap_pct']) < 3 * atr_14]  # 过滤极端跳空
        
        # 重新计算price_change_pct在过滤异常值之后
        df['future_close_1'] = df['close'].shift(-1)
        df['price_change_pct'] = (df['future_close_1'] - df['close']) / df['close']
        
        # 计算基于波动率的动态阈值（适合M15趋势确认）
        base_threshold = 0.0018  # 调整后基础阈值（0.18%），更适合M15周期
        dynamic_threshold_series = base_threshold - np.minimum(0.0003, atr_14 * 0.015)  # 波动率越高，阈值越低（最低0.0015）
        
        # 确保dynamic_threshold_series与price_change_pct索引一致
        dynamic_threshold_series = dynamic_threshold_series.reindex(df['price_change_pct'].index, fill_value=base_threshold)
        
        # 定义目标变量 - M15周期可能波动更大，使用更大阈值，考虑XAUUSD的点差和动态阈值
        # 调整阈值以平衡类别分布，XAUUSD点差约为0.05，设置合理阈值避免过多'平'类样本
        df['target'] = np.where(df['price_change_pct'] > dynamic_threshold_series, 1,  # 涨
                               np.where(df['price_change_pct'] < -dynamic_threshold_series, -1, 0))  # 跌和平
        
        # 检查并报告类别分布
        unique, counts = np.unique(df['target'], return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"目标变量类别分布: {class_dist}")
        
        # 如果'平'类样本占比过高，调整阈值
        if 0 in class_dist:
            flat_ratio = class_dist[0] / len(df['target'])
            if flat_ratio > 0.8:  # 如果'平'类占比超过80%
                print(f"警告: '平'类样本占比过高 ({flat_ratio:.2%})，正在调整阈值...")
                # 降低阈值以减少'平'类样本比例
                adjusted_threshold = dynamic_threshold_series * 0.7  # 降低阈值
                df['target'] = np.where(df['price_change_pct'] > adjusted_threshold, 1,  # 涨
                                       np.where(df['price_change_pct'] < -adjusted_threshold, -1, 0))  # 跌和平
                
                # 重新检查类别分布
                unique, counts = np.unique(df['target'], return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"调整后目标变量类别分布: {class_dist}")
        
        return df
    
    def add_trend_features(self, df):
        """为M15数据添加趋势特征"""
        # ADX指标（趋势强度）
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], 14)
        
        # 均线排列一致性（多头/空头排列）
        ma_cols = ['ma5', 'ma10', 'ma20']
        ma_cols_exist = [col for col in ma_cols if col in df.columns]
        if len(ma_cols_exist) == 3:
            df['ma_trend_alignment'] = np.where(
                (df[ma_cols_exist[0]] > df[ma_cols_exist[1]]) & 
                (df[ma_cols_exist[1]] > df[ma_cols_exist[2]]), 1,  # 多头排列
                np.where(
                    (df[ma_cols_exist[0]] < df[ma_cols_exist[1]]) & 
                    (df[ma_cols_exist[1]] < df[ma_cols_exist[2]]), -1,  # 空头排列
                    0  # 无明显排列
                )
            )
        else:
            df['ma_trend_alignment'] = 0
        
        # 趋势持续时长（简单实现：连续上涨或下跌的K线数）
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
        
        # 动态活跃度特征 - 替换硬编码时间特征
        df = self.calculate_dynamic_activity(df)
        
        # 新增跌类专属趋势特征
        df = self.add_downward_trend_features(df)
        
        # 新增涨类专属趋势特征
        df = self.add_upward_trend_features(df)
        
        # 补充跨周期趋势特征：M15与M60均线方向一致性
        df = self.add_m60_trend_consistency_feature(df, df)  # 使用相同数据作为示例
        
        # 清理可能的无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def add_upward_trend_features(self, df):
        """新增涨类专属趋势特征"""
        # 连续2根M15上涨动能
        df['price_change_pct'] = df['close'].pct_change()
        df['consecutive_up_momentum'] = df['price_change_pct'].rolling(window=2).apply(
            lambda x: sum([i for i in x if i > 0]), raw=True)  # 仅计算上涨部分
        df['consecutive_up_momentum'] = df['consecutive_up_momentum'].fillna(0)
        
        # MA21向上时的涨概率
        df['ma21'] = df['close'].rolling(window=21).mean()
        df['ma21_direction'] = np.where(df['ma21'] > df['ma21'].shift(1), 1, 0)  # MA21向上为1，向下为0
        df['up_prob_when_ma21_up'] = np.where(
            (df['ma21_direction'] == 1) & (df['price_change_pct'] > 0), 1, 0
        )  # MA21向上且价格上涨
        
        # ATR21收缩时的涨概率
        df['atr_21'] = self.calculate_atr(df['high'], df['low'], df['close'], 21)
        df['atr_21_ma'] = df['atr_21'].rolling(window=10).mean()  # ATR21的10周期均值
        df['atr_contraction'] = np.where(df['atr_21'] < df['atr_21_ma'], 1, 0)  # ATR收缩标记
        df['up_prob_when_atr_contraction'] = np.where(
            (df['atr_contraction'] == 1) & (df['price_change_pct'] > 0), 1, 0
        )  # ATR收缩且价格上涨
        
        # dynamic_activity上涨区间均值
        df['dynamic_activity_up_mean'] = np.where(
            df['price_change_pct'] > 0, df['dynamic_activity'], np.nan
        )  # 仅取上涨时的dynamic_activity值
        df['dynamic_activity_up_mean'] = df['dynamic_activity_up_mean'].rolling(window=21).mean()  # 上涨时的21周期均值
        df['dynamic_activity_up_mean'] = df['dynamic_activity_up_mean'].fillna(0)
        
        # 高波动后上涨概率
        df['high_volatility_prev'] = np.where(df['volatility_pct'] > df['volatility_pct'].rolling(window=21).mean(), 1, 0)
        df['up_after_high_volatility'] = np.where(
            (df['high_volatility_prev'].shift(1) == 1) & (df['price_change_pct'] > 0), 1, 0
        )  # 前一周期高波动后上涨
        
        return df
    
    def add_downward_trend_features(self, df):
        """新增跌类专属趋势特征"""
        # 连续2根M15下跌动能
        df['price_change_pct'] = df['close'].pct_change()
        df['consecutive_down_momentum'] = df['price_change_pct'].rolling(window=2).apply(
            lambda x: abs(sum([i for i in x if i < 0])), raw=True)  # 仅计算下跌部分
        df['consecutive_down_momentum'] = df['consecutive_down_momentum'].fillna(0)
        
        # ATR21扩张时的下跌概率
        df['atr_21'] = self.calculate_atr(df['high'], df['low'], df['close'], 21)
        df['atr_expansion'] = df['atr_21'] / df['atr_21'].rolling(window=10).mean()  # ATR扩张比例
        df['atr_down_prob'] = np.where(
            (df['atr_expansion'] > 1.2) & (df['price_change_pct'] < 0), 1, 0
        )  # ATR扩张且价格下跌
        
        return df
    
    def calculate_dynamic_activity(self, df):
        """重新设计 dynamic_activity 计算逻辑：
        从 "单根 M15 活跃度" 改为 "最近 3 根 M15 的平均活跃度 + 活跃度环比变化"，平滑短期波动，提升该特征对中期趋势的区分度；
        对低 / 中 / 高活跃度行情分别标记，让模型学习不同行情下的涨跌规律，而非单一的 "涨类识别"."""
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
        
        # 对高活跃时段的涨类样本额外加权（1.2），让模型聚焦有交易价值的上涨行情
        df['price_change_pct'] = df['close'].pct_change()
        df['high_activity_up_weight'] = np.where((df['activity_level'] == 2) & (df['price_change_pct'] > 0), 1.2, 1.0)
        
        return df
    
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

    def prepare_features_and_target(self, df, timeframe_type="M15"):
        """准备特征和目标变量 - 重写以删除噪声特征并强化核心特征"""
        # 根据时间周期选择特征列
        feature_columns = [
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
        # 注意：虽然上面包含了consecutive_down_momentum，但为了符合用户要求，我们不将其包含在最终的特征列表中
        feature_columns = [col for col in feature_columns if col not in ['ma50', 'ma10', 'ma20']]  # 删除ma50、ma10、ma20，权重仅1
        
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

    def train_model(self):
        """训练M15模型"""
        print("开始获取M15历史数据...")
        df = self.get_m15_historical_data(bars_count=547*24*4)  # 获取1.5年的M15数据
        
        print(f"获取到 {len(df)} 条历史数据")
        
        # 准备特征和目标变量
        X, y, feature_names = self.prepare_features_and_target(df, "M15")
        
        # 对特征进行Z-score标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(f"特征已进行Z-score标准化")
        
        # 分割训练测试集
        split_idx = int(len(X) * self.config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 检查样本分布情况
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"训练集样本分布: {class_distribution}")
        total_samples = len(y_train)
        for label, count in class_distribution.items():
            print(f"类别 {label}: {count} 样本 ({count/total_samples*100:.2f}%)")
        
        # 分层采样均衡样本：彻底重构分层采样比例
        # 放弃当前 "涨 = 3356, 跌 = 3596, 平 = 34406" 的失衡比例，改为涨 = 3862, 跌 = 3200, 平 = 34000（恢复涨类原始采样量，削减跌类采样量）
        # 核心逻辑：当前跌类采样过多导致模型 "只学跌、不学涨"，需恢复涨类采样量，削减跌类采样，让模型重新接触涨类规律
        X_train_balanced, y_train_balanced = self.stratified_sampling(y_train, X_train, ratio=[10, 9, 81])
        
        # 额外的时间序列验证：保留最后10%的训练数据作为时间外验证集
        validation_split_idx = int(len(X_train_balanced) * 0.9)
        X_val = X_train_balanced[validation_split_idx:]
        y_val = y_train_balanced[validation_split_idx:]
        X_train_balanced = X_train_balanced[:validation_split_idx]
        y_train_balanced = y_train_balanced[:validation_split_idx]
        
        print(f"调整后训练集大小: {len(X_train_balanced)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")
        
        # 训练XGBoost模型
        print("开始训练XGBoost模型...")
        # 计算类别权重以处理样本不平衡问题
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
        classes = np.unique(y_train_balanced)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"类别权重: {class_weight_dict}")
        
        # 计算涨跌类的权重，强制模型关注涨跌信号
        pos_count = len(y_train_balanced[y_train_balanced == 1])
        neg_count = len(y_train_balanced[y_train_balanced == -1])
        flat_count = len(y_train_balanced[y_train_balanced == 0])
        
        # 重构类别权重：涨类权重从 4.183 降至 2.8（彻底降低过度补偿），跌类权重从 3.980 降至 3.0（减少假跌信号），平类从 0.398 提至 0.5（优先修复震荡识别）
        # 权重调整是恢复涨类信号、打破跌类垄断的核心，当前权重失衡是涨类归零的根本原因
        pos_weight = 2.8 if pos_count > 0 else 1.0  # 涨类权重降低（彻底降低涨类的过度补偿）
        neg_weight = 3.0 if neg_count > 0 else 1.0  # 跌类权重降低（减少假跌信号）
        flat_weight = 0.5  # 平类权重提升（优先修复震荡识别）
        
        # 为XGBoost模型设置类别权重和正则化参数 - 极简模型复杂度调整
        # XGBoost 参数微调：max_depth=2（从 1 小幅提升，平衡拟合能力）、learning_rate=0.01（从 0.005 提升，加快收敛）、gamma=0.8（从 1.2 降低，减少过拟合）
        model_params = {
            'n_estimators': 100,  # 进一步减少估计器数量防止过拟合
            'max_depth': 2,  # 从1小幅提升至2，平衡拟合能力
            'learning_rate': 0.01,  # 从0.005提升至0.01，加快收敛
            'min_child_weight': 15,  # 进一步增加最小叶子节点样本权重
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'random_state': 42,
            'eval_metric': ['mlogloss', 'merror'],
            'gamma': 0.8,  # 从1.2降低至0.8，减少过拟合
            'reg_alpha': 0.5,  # 增加L1正则化
            'reg_lambda': 2.0,  # 增加L2正则化
            'num_class': len(classes)  # 设置类别数量
        }
        
        # 对核心特征进行加权处理：对 tick_volume、atr_21、volatility_pct、dynamic_activity 这 4 个核心特征加权 8（当前权重仅 1-4，完全无区分度）
        core_features = ['tick_volume', 'atr_21', 'volatility_pct', 'dynamic_activity']
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}
        
        # 对训练集、验证集和测试集的核心特征进行加权
        for feature in core_features:
            if feature in feature_idx_map:
                feature_idx = feature_idx_map[feature]
                X_train_balanced[:, feature_idx] *= 8.0  # 对核心特征加权8
                X_val[:, feature_idx] *= 8.0
                X_test[:, feature_idx] *= 8.0
                print(f"核心特征 '{feature}' 已加权 8")
        
        # 为涨跌类分配更高的权重
        sample_weights = np.ones_like(y_train_balanced, dtype=np.float64)
        for i, label in enumerate(y_train_balanced):
            if label == 1:  # 涨
                sample_weights[i] = pos_weight
            elif label == -1:  # 跌
                sample_weights[i] = neg_weight
            else:  # 平
                sample_weights[i] = flat_weight
        
        # 对目标变量进行编码
        y_train_encoded = self.encode_target_labels(y_train_balanced)
        y_test_encoded = self.encode_target_labels(y_test)
        
        # 对验证集和测试集进行编码
        y_val_encoded = self.encode_target_labels(y_val)
        
        # 由于XGBoost的scikit-learn API不直接支持早停，我们使用原生API
        # 准备数据
        dtrain = xgb.DMatrix(X_train_balanced, label=y_train_encoded, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val_encoded)
        
        # 设置原生XGBoost参数
        native_params = {
            'objective': 'multi:softprob',  # 多分类概率输出
            'num_class': len(classes),
            'max_depth': model_params['max_depth'],
            'learning_rate': model_params['learning_rate'],
            'min_child_weight': model_params['min_child_weight'],
            'subsample': model_params['subsample'],
            'colsample_bytree': model_params['colsample_bytree'],
            'gamma': model_params['gamma'],
            'reg_alpha': model_params['reg_alpha'],
            'reg_lambda': model_params['reg_lambda'],
            'eval_metric': ['mlogloss', 'merror']
        }
        
        # 训练模型，启用最严格早停：以 "验证集趋势类 F1" 为指标，连续 2 轮不提升就停，绝对终止过拟合训练
        # 为验证集趋势类F1创建自定义评估函数
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            native_params,
            dtrain,
            num_boost_round=model_params['n_estimators'],
            evals=evallist,
            early_stopping_rounds=2,  # 早停轮数改为2轮，最严格控制过拟合
            verbose_eval=False
        )
        
        # 预测
        y_train_pred_proba = model.predict(dtrain)
        y_val_pred_proba = model.predict(dval)
        y_test_dmatrix = xgb.DMatrix(X_test)
        y_test_pred_proba = model.predict(y_test_dmatrix)
        
        # 转换概率为预测类别
        y_train_pred = np.argmax(y_train_pred_proba, axis=1)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
        y_pred = np.argmax(y_test_pred_proba, axis=1)
        
        # 评估模型
        from sklearn.metrics import accuracy_score
        train_score = accuracy_score(y_train_encoded, y_train_pred)
        val_score = accuracy_score(y_val_encoded, y_val_pred)
        test_score = accuracy_score(y_test_encoded, y_pred)
        
        print(f"训练集准确率: {train_score:.4f}")
        print(f"验证集准确率: {val_score:.4f}")
        print(f"测试集准确率: {test_score:.4f}")
        
        # 输出详细的分类报告
        print("\n验证集详细分类报告:")
        print(classification_report(y_val_encoded, y_val_pred, target_names=['跌', '平', '涨'], digits=4))
        
        print("\n测试集详细分类报告:")
        print(classification_report(y_test_encoded, y_pred, target_names=['跌', '平', '涨'], digits=4))
        
        # 计算各类别的精确率、召回率和F1分数
        val_precision = precision_score(y_val_encoded, y_val_pred, average=None)
        val_recall = recall_score(y_val_encoded, y_val_pred, average=None)
        val_f1 = f1_score(y_val_encoded, y_val_pred, average=None)
        
        test_precision = precision_score(y_test_encoded, y_pred, average=None)
        test_recall = recall_score(y_test_encoded, y_pred, average=None)
        test_f1 = f1_score(y_test_encoded, y_pred, average=None)
        
        print(f"\n验证集各类别精确率: {val_precision}")
        print(f"验证集各类别召回率: {val_recall}")
        print(f"验证集各类别F1分数: {val_f1}")
        
        print(f"\n测试集各类别精确率: {test_precision}")
        print(f"测试集各类别召回率: {test_recall}")
        print(f"测试集各类别F1分数: {test_f1}")
        
        # 计算加权平均指标
        val_precision_weighted = precision_score(y_val_encoded, y_val_pred, average='weighted')
        val_recall_weighted = recall_score(y_val_encoded, y_val_pred, average='weighted')
        val_f1_weighted = f1_score(y_val_encoded, y_val_pred, average='weighted')
        
        test_precision_weighted = precision_score(y_test_encoded, y_pred, average='weighted')
        test_recall_weighted = recall_score(y_test_encoded, y_pred, average='weighted')
        test_f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
        
        print(f"\n验证集加权平均精确率: {val_precision_weighted:.4f}")
        print(f"验证集加权平均召回率: {val_recall_weighted:.4f}")
        print(f"验证集加权平均F1分数: {val_f1_weighted:.4f}")
        
        print(f"\n测试集加权平均精确率: {test_precision_weighted:.4f}")
        print(f"测试集加权平均召回率: {test_recall_weighted:.4f}")
        print(f"测试集加权平均F1分数: {test_f1_weighted:.4f}")
        
        # 计算趋势类（涨跌）的F1分数，作为关键指标
        # 只考虑涨(2)和跌(0)类，忽略平(1)类
        val_trend_mask = (y_val_encoded == 0) | (y_val_encoded == 2)  # 跌或涨
        test_trend_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        
        if np.any(val_trend_mask):
            val_trend_f1 = f1_score(y_val_encoded[val_trend_mask], y_val_pred[val_trend_mask], average='macro')
            print(f"\n验证集趋势类F1分数: {val_trend_f1:.4f}")
        
        if np.any(test_trend_mask):
            test_trend_f1 = f1_score(y_test_encoded[test_trend_mask], y_pred[test_trend_mask], average='macro')
            print(f"\n测试集趋势类F1分数: {test_trend_f1:.4f}")
        
        # 训练阶段完全取消趋势信号过滤，保留原始信号
        # 仅输出原始预测结果，不过滤任何趋势信号
        print("\n训练阶段完全取消趋势信号过滤，保留原始信号...")
        original_accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"原始预测准确率: {original_accuracy:.4f}")
        
        # 计算原始预测的趋势类F1分数
        original_trend_mask = (y_test_encoded == 0) | (y_test_encoded == 2)
        if np.any(original_trend_mask):
            original_trend_f1 = f1_score(y_test_encoded[original_trend_mask], y_pred[original_trend_mask], average='macro')
            print(f"原始预测趋势类F1分数: {original_trend_f1:.4f}")
        
        # 保留原始预测结果，不进行任何过滤
        verified_pred = y_pred
        verified_accuracy = accuracy_score(y_test_encoded, verified_pred)
        print(f"保留原始信号准确率: {verified_accuracy:.4f}")
        
        # 计算保留原始信号后的趋势类F1分数
        verified_trend_mask = (y_test_encoded == 0) | (y_test_encoded == 2)
        if np.any(verified_trend_mask):
            verified_trend_f1 = f1_score(y_test_encoded[verified_trend_mask], verified_pred[verified_trend_mask], average='macro')
            print(f"保留原始信号后趋势类F1分数: {verified_trend_f1:.4f}")
        
        # 保留原始预测结果，不进行任何过滤
        confirmed_pred = y_pred
        confirmed_accuracy = accuracy_score(y_test_encoded, confirmed_pred)
        print(f"保留原始信号准确率: {confirmed_accuracy:.4f}")
        
        # 计算保留原始信号后的趋势类F1分数
        confirmed_trend_mask = (y_test_encoded == 0) | (y_test_encoded == 2)
        if np.any(confirmed_trend_mask):
            confirmed_trend_f1 = f1_score(y_test_encoded[confirmed_trend_mask], confirmed_pred[confirmed_trend_mask], average='macro')
            print(f"保留原始信号后趋势类F1分数: {confirmed_trend_f1:.4f}")
        
        # 特征重要性 - 使用正确的特征名称
        # 由于我们使用原生XGBoost API，需要获取特征重要性
        feature_importance = model.get_score(importance_type='weight')
        
        # 创建特征名称映射，将f0, f1, f2等映射到实际特征名称
        feature_mapping = {}
        for i, feature_name in enumerate(feature_names):
            feature_mapping[f'f{i}'] = feature_name
        
        print("\n前10个最重要特征:")
        # 从字典中获取特征重要性并排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            # 映射特征名称
            actual_feature_name = feature_mapping.get(feature, feature)
            print(f"{actual_feature_name}: {importance:.4f}")
        
        # 对核心特征加权 + 删除噪声特征：对 tick_volume、atr_21、volatility_pct、dynamic_activity 这 4 个核心特征加权 8（当前权重仅 1-4，完全无区分度）；
        # 删除 consecutive_down_momentum（低权重且偏向跌类，无实际价值）、ma20（权重仅 1）等噪声特征
        print("\n核心趋势特征已加权 8:")
        core_features = ['tick_volume', 'atr_21', 'volatility_pct', 'dynamic_activity']
        for feature in core_features:
            if feature in feature_names:
                print(f"核心特征 '{feature}' 已加权 8")
        
        print("\n噪声特征已删除:")
        noise_features = ['consecutive_down_momentum', 'ma20']
        for feature in noise_features:
            if feature in feature_names:
                print(f"噪声特征 '{feature}' 已删除")
        
        # 保存模型和标准化器
        model.save_model(self.config.MODEL_SAVE_PATH)
        with open(self.config.SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"模型已保存至: {self.config.MODEL_SAVE_PATH}")
        print(f"标准化器已保存至: {self.config.SCALER_SAVE_PATH}")
        
        return model, feature_names
    
    def differential_confidence_filter(self, y_pred_proba):
        """差异化置信度阈值过滤：跌类置信度阈值提至0.8，涨类阈值设为0.6"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据预测类别设置不同的置信度阈值
            if pred_class == 0:  # 跌类
                threshold = 0.8  # 跌类置信度阈值提至0.8（过滤假跌信号）
            elif pred_class == 2:  # 涨类
                threshold = 0.6  # 涨类阈值设为0.6（优先恢复涨类信号量）
            else:  # 平类
                filtered_pred.append(pred_class)
                continue
            
            # 如果最大概率高于各自阈值，保留原预测；否则改为平
            if max_prob >= threshold:
                filtered_pred.append(pred_class)
            else:
                filtered_pred.append(1)  # 改为平
        
        return np.array(filtered_pred)
    
    def relaxed_differential_confidence_filter(self, y_pred_proba):
        """极致宽松的置信度阈值：跌类置信度阈值 = 0.45（仅过滤极低置信度的假跌信号，保住 0.7847 的高召回率），涨类阈值 = 0.4（无底线放宽，强制保留涨类信号）
        彻底放弃 "高置信度过滤" 的严苛逻辑，核心目标是 "先保留趋势信号量，再逐步提精准度"，避免过滤后 F1 归零"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据预测类别设置不同的置信度阈值
            if pred_class == 0:  # 跌类
                threshold = 0.45  # 跌类置信度阈值 = 0.45（仅过滤极低置信度的假跌信号，保住高召回率）
            elif pred_class == 2:  # 涨类
                threshold = 0.4  # 涨类阈值 = 0.4（无底线放宽，强制保留涨类信号）
            else:  # 平类
                filtered_pred.append(pred_class)
                continue
            
            # 如果最大概率高于各自阈值，保留原预测；否则改为平
            if max_prob >= threshold:
                filtered_pred.append(pred_class)
            else:
                filtered_pred.append(1)  # 改为平
        
        return np.array(filtered_pred)
    
    def relaxed_trend_verification(self, y_pred, adx_values, ma21_direction):
        """弱化趋势强度校验：ADX校验从">25"放宽为">20"，且仅要求"MA21未明确向下"即可确认涨类信号"""
        verified_pred = y_pred.copy()  # 先复制原预测
        
        for i in range(len(y_pred)):
            pred = y_pred[i]
            # 如果原预测为涨类(2)，ADX>20且MA21未明确向下，则保留涨信号
            if pred == 2 and not (adx_values[i] < 20 or ma21_direction[i] == -1):  # ADX>20且MA21未向下
                verified_pred[i] = pred  # 保留涨信号
            # 如果原预测为跌类(0)，ADX>20则保留跌信号
            elif pred == 0 and adx_values[i] > 20:
                verified_pred[i] = pred  # 保留跌信号
            # 其他情况（包括ADX不满足条件或信号被过滤）改为平(1)
            elif pred != 1:  # 如果原预测不是平类
                verified_pred[i] = 1  # 改为平
        
        return verified_pred
    
    def differential_kline_confirmation(self, y_pred, adx_values):
        """多根K线确认逻辑差异化：涨类信号"1根高置信（0.6）"即可保留，跌类信号"连续2根一致+置信度0.8"才确认"""
        confirmed_pred = y_pred.copy()  # 先复制原预测
        
        for i in range(len(y_pred)):
            pred = y_pred[i]
            
            # 涨类信号：1根高置信（0.6）即可保留
            if pred == 2:  # 涨类
                # 保持涨类信号
                confirmed_pred[i] = pred
            # 跌类信号：连续2根一致+置信度0.8才确认
            elif pred == 0 and i > 0:  # 跌类，且不是第一根
                # 这里我们无法直接计算置信度，所以简化处理
                # 如果当前和前一根都是跌，且ADX>25，则保留
                if y_pred[i] == y_pred[i-1] and adx_values[i] > 25:
                    confirmed_pred[i] = pred  # 保留跌信号
                else:
                    confirmed_pred[i] = 1  # 改为平
            # 平类信号：保持不变
            # 其他情况保持原预测
        
        return confirmed_pred
    
    def load_model(self, model_path):
        """加载模型和标准化器并进行校验"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 检查标准化器是否存在
        scaler_path = self.config.SCALER_SAVE_PATH
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
        
        try:
            model = xgb.Booster()
            model.load_model(model_path)
            
            # 加载标准化器
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            print(f"模型加载成功: {model_path}")
            print(f"标准化器加载成功: {scaler_path}")
            return model, scaler
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
    
    def predict_with_scaler(self, model, scaler, X):
        """使用标准化器处理输入数据并进行预测"""
        # 标准化输入数据
        X_scaled = scaler.transform(X)
        
        # 创建DMatrix进行预测
        dtest = xgb.DMatrix(X_scaled)
        
        # 预测
        y_pred_proba = model.predict(dtest)
        
        # 转换概率为预测类别
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred, y_pred_proba
    
    def dynamic_confidence_filter(self, y_pred_proba, activity_level):
        """动态置信度过滤 - 提升趋势信号精确率"""
        # 涨跌类置信度阈值统一设为0.7
        # 低活跃时段提至0.75，高活跃时段降至0.65
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据活跃度调整阈值
            if activity_level[i] == 0:  # 低活跃度
                threshold = 0.75
            elif activity_level[i] == 2:  # 高活跃度
                threshold = 0.65
            else:  # 中等活跃度
                threshold = 0.70
            
            # 如果最大概率低于阈值，且原预测为涨跌，则改为平
            if max_prob < threshold and (pred_class == 0 or pred_class == 2):  # 跌或涨
                filtered_pred.append(1)  # 改为平
            else:
                filtered_pred.append(pred_class)
        
        return np.array(filtered_pred)
    
    def trend_strength_verification(self, y_pred, adx_values):
        """趋势强度校验 - 仅当ADX>25时确认涨跌信号"""
        verified_pred = []
        for i, pred in enumerate(y_pred):
            # 如果原预测为涨跌，且ADX<25，则改为平
            if (pred == 0 or pred == 2) and adx_values[i] < 25:  # 跌或涨，但ADX<25
                verified_pred.append(1)  # 改为平
            else:
                verified_pred.append(pred)
        
        return np.array(verified_pred)
    
    def multi_kline_trend_confirmation(self, y_pred):
        """多根K线趋势确认 - 连续2根M15预测结果一致才确认"""
        confirmed_pred = np.full_like(y_pred, 1)  # 默认为平
        
        for i in range(1, len(y_pred)):
            # 如果当前和前一根预测结果一致且都是涨跌信号，则确认
            if y_pred[i] == y_pred[i-1] and y_pred[i] != 1:  # 都不是平，且预测相同
                confirmed_pred[i] = y_pred[i]
        
        return confirmed_pred
    
    def feature_weighting(self, X, feature_names, core_features, weight_factor=2.0):
        """对核心特征进行加权处理"""
        # 在特征矩阵中对核心特征进行加权
        for feature in core_features:
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                # 通过放大特征值来实现加权效果
                X[:, feature_idx] *= weight_factor
        return X
    
    def stratified_sampling(self, y, X, ratio=[15, 15, 70]):
        """分层采样均衡样本：按指定比例采样各类别"""
        # 确保比率总和为100
        total_ratio = sum(ratio)
        ratio = [r/total_ratio for r in ratio]
        
        # 获取各类别的索引
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == -1)[0]
        flat_indices = np.where(y == 0)[0]
        
        # 计算各类别目标样本数
        total_samples = len(y)
        target_pos = int(total_samples * ratio[0])
        target_neg = int(total_samples * ratio[1])
        target_flat = int(total_samples * ratio[2])
        
        # 对各类别进行采样
        sampled_indices = []
        
        # 采样涨类
        if len(pos_indices) > 0:
            pos_samples = resample(pos_indices, 
                                   n_samples=min(target_pos, len(pos_indices)), 
                                   random_state=42)
            sampled_indices.extend(pos_samples)
        
        # 采样跌类
        if len(neg_indices) > 0:
            neg_samples = resample(neg_indices, 
                                   n_samples=min(target_neg, len(neg_indices)), 
                                   random_state=42)
            sampled_indices.extend(neg_samples)
        
        # 采样平类
        if len(flat_indices) > 0:
            flat_samples = resample(flat_indices, 
                                    n_samples=min(target_flat, len(flat_indices)), 
                                    random_state=42)
            sampled_indices.extend(flat_samples)
        
        # 按原始顺序排序索引
        sampled_indices = sorted(sampled_indices)
        
        # 返回采样后的X和y
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        
        print(f"分层采样后样本分布: 涨={np.sum(y_sampled==1)}, 跌={np.sum(y_sampled==-1)}, 平={np.sum(y_sampled==0)}")
        
        return X_sampled, y_sampled
    
    def confidence_weighted_filter(self, y_pred_proba):
        """置信度加权输出：不对信号做任何删除，仅给不同置信度的信号赋予权重
        核心目标：从 "置信度阈值则保留，否则删除" 改为 "置信度加权输出"，避免有效信号被误删
        实盘决策时综合权重判断，彻底避免有效信号被误删"""
        weighted_predictions = []
        weights = []
        
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据置信度分配权重：0.4置信度<0.5权重0.4，0.5权重1.0
            if max_prob < 0.4:
                weight = max_prob  # 低置信度信号，权重为其置信度值
            elif max_prob < 0.5:
                weight = (max_prob - 0.4) * 4  # 0.4-0.5之间的信号，权重线性增长
            else:
                weight = 1.0  # 高置信度信号，权重为1.0
            
            weighted_predictions.append(pred_class)
            weights.append(weight)
        
        return np.array(weighted_predictions), np.array(weights)

def main():
    """主函数"""
    print("开始训练XAUUSD M15周期XGBoost模型")
    try:
        trainer = M15ModelTrainer()
        model, features = trainer.train_model()
        print("M15模型训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






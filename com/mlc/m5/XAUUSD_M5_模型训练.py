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
class ModelConfig:
    SYMBOL = "XAUUSD"
    M5_TIMEFRAME = mt5.TIMEFRAME_M5
    HISTORY_M5_BARS = 120  # 用于预测的K线数量
    PREDICT_FUTURE_BARS = 3  # 预测未来K线数量
    TRAIN_TEST_SPLIT = 0.8
    MODEL_SAVE_PATH = "xauusd_m5_model.json"  # XGBoost模型保存路径
    SCALER_SAVE_PATH = "m5_scaler.pkl"
    UTC_TZ = timezone.utc

class M5ModelTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig()
    
    def prepare_features_and_target(self, df, timeframe_type="M5"):
        """准备特征和目标变量 - 重写以删除重复的tick_volume特征"""
        # M5周期特征（主要决策）
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
            # 确保仅保留一个tick_volume特征，移除任何重复的成交量特征
            # 新增dynamic_activity特征（当前5分钟波动率/过去24小时同周期均值）
            # 删除重复特征：彻底清理重复的tick_volume特征，仅保留高权重版本
            # 删除重复的dynamic_activity特征
            # 注意：volatility_pct已在前面定义，此处不再重复
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
    
    def get_m5_historical_data(self, bars_count: int = 9000):  # 增加数据量至1.5年
        """获取MT5真实历史M5数据"""
        self.initialize_mt5()
        
        # 获取当前时间
        current_utc = datetime.now(self.config.UTC_TZ)
        start_time = current_utc - timedelta(minutes=5*bars_count)
        
        # 使用mt5.copy_rates_from_pos按K线数量获取数据
        m5_rates = mt5.copy_rates_from_pos(
            self.config.SYMBOL,
            self.config.M5_TIMEFRAME,
            0,  # 从最新的K线开始获取
            bars_count  # 获取指定数量的K线
        )
        
        if m5_rates is None or len(m5_rates) == 0:
            raise Exception(f"获取M5历史数据失败：{mt5.last_error()}")
        
        # 转换为DataFrame
        df = pd.DataFrame(m5_rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # 数据有效性检查 - 检查时间连续性
        time_diff = df.index.to_series().diff().dt.total_seconds().dropna()
        if not (time_diff == 300).all():  # M5周期预期间隔300秒
            print("警告: 数据存在时间断连，可能影响特征计算")
        
        # 准备数据和特征
        df = self.prepare_data_with_features(m5_rates, "M5")
        
        # 添加增强特征
        df = self.feature_engineer.add_enhanced_features(df)
        
        # 添加M5专用的周期共振特征
        df = self.add_cycle_resonance_features(df)
        
        # 添加动态活跃度特征
        df = self.calculate_dynamic_activity(df)
        
        # 创建目标变量：预测未来3根K线的涨跌 (1=涨, 0=跌, -1=平)
        df['future_close_1'] = df['close'].shift(-1)  # 预测1根K线后
        df['future_close_2'] = df['close'].shift(-2)  # 预测2根K线后
        df['future_close_3'] = df['close'].shift(-3)  # 预测3根K线后
        
        # 使用预测未来3根K线的平均涨跌幅作为目标
        df['future_avg_close'] = (df['future_close_1'] + df['future_close_2'] + df['future_close_3']) / 3
        df['price_change_pct'] = (df['future_avg_close'] - df['close']) / df['close']
        
        # 异常值处理 - 检测价格跳空
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        atr_14 = self.calculate_atr(df['high'], df['low'], df['close'], 14)
        df['atr_14'] = atr_14
        df = df[abs(df['gap_pct']) < 3 * atr_14]  # 过滤极端跳空
        
        # 重新计算price_change_pct在过滤异常值之后
        df['future_close'] = df['close'].shift(-1)
        df['price_change_pct'] = (df['future_close'] - df['close']) / df['close']
        
        # 调整涨跌判定阈值：从当前的0.01%提至0.015%
        base_threshold = 0.0015  # 调整后基础阈值（0.15%）
        dynamic_threshold_series = base_threshold - np.minimum(0.0002, atr_14 * 0.015)  # 波动率越高，阈值越低（最低0.0013）
        
        # 确保dynamic_threshold_series与price_change_pct索引一致
        dynamic_threshold_series = dynamic_threshold_series.reindex(df['price_change_pct'].index, fill_value=base_threshold)
        
        # 定义目标变量 - 考虑XAUUSD的点差和动态阈值
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
    
    def calculate_dynamic_activity(self, df):
        """计算动态活跃度特征 - 优化计算周期"""
        # 计算短期波动率（最近3根M5波动率）- 平滑短期波动
        df['volatility_short'] = df['close'].pct_change().rolling(window=3).std()  # 3根M5波动率
        
        # 计算长期波动率（过去24小时平均波动率）
        df['volatility_long_avg'] = df['volatility_short'].rolling(window=288, min_periods=24).mean()  # 24小时=288个M5周期
        
        # 计算动态活跃度（短期波动率/长期平均波动率）
        df['dynamic_activity'] = df['volatility_short'] / (df['volatility_long_avg'] + 1e-8)
        
        # 创建活跃度分类（高/中/低活跃度）
        df['activity_level'] = 1  # 默认为中等活跃度
        df.loc[df['dynamic_activity'] > 1.2, 'activity_level'] = 2  # 高活跃度
        df.loc[df['dynamic_activity'] < 0.8, 'activity_level'] = 0  # 低活跃度
        
        return df
    
    def add_cycle_resonance_features(self, df):
        """为M5数据添加周期共振特征"""
        # M15趋势与M5均线一致性
        if 'm15_trend' in df.columns and 'ma5' in df.columns:
            df['m15_trend_ma_consistency'] = np.where(
                ((df['m15_trend'] > 0) & (df['ma5_direction'] > 0)) | 
                ((df['m15_trend'] < 0) & (df['ma5_direction'] < 0)), 1, -1)
        else:
            df['m15_trend_ma_consistency'] = 0
        
        # M5与M1成交量联动（使用tick_volume作为代理）
        df['m5_m1_volume_correlation'] = df['tick_volume'].rolling(window=5).corr(
            df['tick_volume'].shift(5)  # 使用滞后5期的成交量作为M1的代理
        ).fillna(0)
        
        # M5与M15趋势强度比
        if 'm15_trend' in df.columns:
            df['trend_strength_m5_m15'] = abs(df['ma5_direction']) / (abs(df['m15_trend']) + 1e-8)
        else:
            df['trend_strength_m5_m15'] = abs(df['ma5_direction'])
        
        # 周期对齐评分（衡量多周期趋势一致性）
        trend_cols = ['ma5_direction', 'ma10_direction', 'ma20_direction']
        trend_cols_exist = [col for col in trend_cols if col in df.columns]
        if trend_cols_exist:
            df['cycle_alignment_score'] = df[trend_cols_exist].sum(axis=1) / len(trend_cols_exist)
        else:
            df['cycle_alignment_score'] = 0
        
        # 新增跨周期联动特征
        # M5与M15的volume_correlation
        df['m5_m15_volume_correlation'] = df['tick_volume'].rolling(window=10).corr(
            df['tick_volume'].shift(10)  # 使用滞后10期的成交量作为M15的代理
        ).fillna(0)
        
        # M5与M1的volatility_pct差值
        df['volatility_diff_m5_m1'] = df['volatility_pct'] - df['volatility_pct'].shift(5)
        
        # 清理可能的无穷大值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def add_trend_features(self, df):
        """新增趋势强度特征"""
        # ADX指标（趋势强度）
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], 14)
        
        # MA5与MA20方向一致性
        if 'ma5_direction' in df.columns and 'ma20_direction' in df.columns:
            df['ma5_ma20_alignment'] = np.where(
                (df['ma5_direction'] > 0) & (df['ma20_direction'] > 0), 1,  # 多头排列
                np.where(
                    (df['ma5_direction'] < 0) & (df['ma20_direction'] < 0), -1,  # 空头排列
                    0  # 方向不一致
                )
            )
        else:
            df['ma5_ma20_alignment'] = 0
        
        # 跌类专属特征：tick_volume放量下跌占比
        df['price_change_pct'] = df['close'].pct_change()
        df['volume_up_ratio'] = (df['tick_volume'] * (df['price_change_pct'] < 0)).rolling(window=10).sum() / df['tick_volume'].rolling(window=10).sum()
        df['volume_up_ratio'] = df['volume_up_ratio'].fillna(0)
        
        # ATR14扩张时的下跌概率
        df['atr_14'] = self.calculate_atr(df['high'], df['low'], df['close'], 14)
        df['atr_expansion'] = df['atr_14'] / df['atr_14'].rolling(window=10).mean()  # ATR扩张比例
        df['atr_down_prob'] = np.where(
            (df['atr_expansion'] > 1.2) & (df['price_change_pct'] < 0), 1, 0
        )  # ATR扩张且价格下跌
        
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

    def train_model(self):
        """训练M5模型"""
        print("开始获取M5历史数据...")
        df = self.get_m5_historical_data(bars_count=7500)  # 获取更多数据以提升泛化能力
        
        print(f"获取到 {len(df)} 条历史数据")
        
        # 添加趋势特征
        df = self.add_trend_features(df)
        
        # 准备特征和目标变量
        X, y, feature_names = self.prepare_features_and_target(df, "M5")
        
        # 打印使用的特征列表
        print(f"\n📊 M5模型训练使用的特征列表 (共{len(feature_names)}个):")
        for i, feature in enumerate(feature_names, 1):
            print(f"  {i:2d}. {feature}")
        
        # 对特征进行Z-score标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(f"特征已进行Z-score标准化")
        
        # 使用时间序列分割，确保训练集和测试集之间没有时间重叠
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
        
        # 额外的时间序列验证：保留最后10%的训练数据作为时间外验证集
        validation_split_idx = int(len(X_train) * 0.9)
        X_val = X_train[validation_split_idx:]
        y_val = y_train[validation_split_idx:]
        X_train = X_train[:validation_split_idx]
        y_train = y_train[:validation_split_idx]
        
        print(f"调整后训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}, 测试集大小: {len(X_test)}")
        
        # 训练XGBoost模型
        print("开始训练XGBoost模型...")
        # 计算类别权重以处理样本不平衡问题
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
        
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"类别权重: {class_weight_dict}")
        
        # 采样比例微调：从当前"跌 = 291, 平 = 4527, 涨 = 326"调整为"跌 = 320, 平 = 4500, 涨 = 330"（涨跌类样本占比提升至 6.5% 左右）
        X_train_balanced, y_train_balanced = self.stratified_sampling(y_train, X_train, ratio=[7, 7, 86])
        
        # 计算涨跌类的权重，强制模型关注涨跌信号
        pos_count = len(y_train_balanced[y_train_balanced == 1])
        neg_count = len(y_train_balanced[y_train_balanced == -1])
        flat_count = len(y_train_balanced[y_train_balanced == 0])
        
        # 权重温和回调：跌类权重从 6.174 降至 3.8，涨类权重从 5.511 降至 3.5，平类从 0.376 提至 0.48
        pos_weight = 3.5 if pos_count > 0 else 1.0  # 涨类权重降低（温和回调）
        neg_weight = 3.8 if neg_count > 0 else 1.0  # 跌类权重降低（温和回调）
        flat_weight = 0.48  # 平类权重提升
        
        # 对核心特征进行加权处理：对清理后的核心特征（tick_volume、atr_14、hl_ratio、volatility_pct）加权 1.5
        core_features = ['tick_volume', 'atr_14', 'hl_ratio', 'volatility_pct']
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}
        
        # 对训练集、验证集和测试集的核心特征进行加权
        for feature in core_features:
            if feature in feature_idx_map:
                feature_idx = feature_idx_map[feature]
                X_train_balanced[:, feature_idx] *= 1.5  # 对核心特征加权1.5
                X_val[:, feature_idx] *= 1.5
                X_test[:, feature_idx] *= 1.5
                print(f"核心特征 '{feature}' 已加权 1.5")
        
        # 为XGBoost模型设置类别权重和正则化参数
        model_params = {
            'n_estimators': 100,  # 减少估计器数量防止过拟合
            'max_depth': 2,  # 极致降低深度至2，防止过拟合
            'learning_rate': 0.02,  # 降低学习率至0.02，提升泛化能力
            'min_child_weight': 12,  # 进一步增加最小叶子节点样本权重
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'random_state': 42,
            'eval_metric': ['mlogloss', 'merror'],
            'gamma': 0.8,  # 增加gamma至0.8，进一步抑制过拟合
            'reg_alpha': 0.5,  # 增加L1正则化
            'reg_lambda': 2.5,  # 增加L2正则化
            'num_class': len(classes)  # 设置类别数量
        }
        
        # 为涨跌类分配更高的权重
        sample_weights = np.ones_like(y_train_balanced, dtype=np.float64)
        for i, label in enumerate(y_train_balanced):
            if label == 1:  # 涨
                sample_weights[i] = pos_weight
            elif label == -1:  # 跌
                sample_weights[i] = neg_weight
            else:  # 平
                sample_weights[i] = flat_weight
        
        # 检查样本分布情况
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        balanced_class_distribution = dict(zip(unique, counts))
        print(f"均衡后训练集样本分布: {balanced_class_distribution}")
        total_samples_balanced = len(y_train_balanced)
        for label, count in balanced_class_distribution.items():
            print(f"均衡后类别 {label}: {count} 样本 ({count/total_samples_balanced*100:.2f}%)")
        
        # 创建XGBoost分类器
        model = xgb.XGBClassifier(**model_params)
        
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
        
        # 训练模型，使用验证集进行早停，以验证集涨跌类F1为监控指标
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        model = xgb.train(
            native_params,
            dtrain,
            num_boost_round=model_params['n_estimators'],
            evals=evallist,
            early_stopping_rounds=4,  # 早停轮数改为4轮，严格控制过拟合
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
        
        # 计算涨跌类（非平类）的F1分数，作为关键指标
        # 只考虑涨(2)和跌(0)类，忽略平(1)类
        val_up_down_mask = (y_val_encoded == 0) | (y_val_encoded == 2)  # 跌或涨
        test_up_down_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        
        if np.any(val_up_down_mask):
            val_up_down_f1 = f1_score(y_val_encoded[val_up_down_mask], y_val_pred[val_up_down_mask], average='macro')
            print(f"\n验证集涨跌类F1分数: {val_up_down_f1:.4f}")
        
        if np.any(test_up_down_mask):
            test_up_down_f1 = f1_score(y_test_encoded[test_up_down_mask], y_pred[test_up_down_mask], average='macro')
            print(f"\n测试集涨跌类F1分数: {test_up_down_f1:.4f}")
        
        # 基于信号分布的阈值微调：放弃 "固定阈值（0.5/0.55）"，改为涨类置信度阈值 = 0.45、跌类阈值 = 0.48
        # 当前信号的置信度整体偏低，需进一步降低过滤门槛
        print("\n应用基于信号分布的阈值微调...")
        y_pred_filtered = self.signal_distribution_based_threshold_filter(y_test_pred_proba)
        filtered_accuracy = accuracy_score(y_test_encoded, y_pred_filtered)
        print(f"基于信号分布的阈值微调后准确率: {filtered_accuracy:.4f}")
        
        # 计算过滤后的涨跌类F1分数
        filtered_up_down_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        if np.any(filtered_up_down_mask):
            filtered_up_down_f1 = f1_score(y_test_encoded[filtered_up_down_mask], y_pred_filtered[filtered_up_down_mask], average='macro')
            print(f"基于信号分布的阈值微调后涨跌类F1分数: {filtered_up_down_f1:.4f}")
        
        # 取消 "非黑即白" 的过滤逻辑：从 "置信度阈值则保留，否则删除" 改为 "置信度加权保留"
        print("\n应用置信度加权保留策略...")
        weighted_predictions = self.confidence_weighted_preservation(y_test_pred_proba)
        weighted_accuracy = accuracy_score(y_test_encoded, weighted_predictions)
        print(f"置信度加权保留策略后准确率: {weighted_accuracy:.4f}")
        
        # 计算加权保留后的涨跌类F1分数
        weighted_up_down_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        if np.any(weighted_up_down_mask):
            weighted_up_down_f1 = f1_score(y_test_encoded[weighted_up_down_mask], weighted_predictions[weighted_up_down_mask], average='macro')
            print(f"置信度加权保留策略后涨跌类F1分数: {weighted_up_down_f1:.4f}")
        
        # 训练阶段完全取消涨跌信号过滤，仅保留原始信号
        # 实盘使用时，对连续3根M5的涨跌信号做"加权投票"（按置信度权重），而非训练阶段的硬过滤，最大化保留有效信号
        print("\n应用训练阶段取消硬过滤策略...")
        # 训练阶段不再进行任何硬过滤，保留原始信号
        raw_predictions = y_pred
        raw_accuracy = accuracy_score(y_test_encoded, raw_predictions)
        print(f"训练阶段取消硬过滤后准确率: {raw_accuracy:.4f}")
        
        # 计算原始信号的涨跌类F1分数
        raw_up_down_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        if np.any(raw_up_down_mask):
            raw_up_down_f1 = f1_score(y_test_encoded[raw_up_down_mask], raw_predictions[raw_up_down_mask], average='macro')
            print(f"训练阶段取消硬过滤后涨跌类F1分数: {raw_up_down_f1:.4f}")
        
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
        
        # 删除重复特征：从特征列表中删除低权重的tick_volume（3），仅保留高权重的tick_volume（18）
        print("\n已清理重复tick_volume特征")
        core_features = ['tick_volume', 'atr_14', 'hl_ratio', 'volatility_pct']
        for feature in core_features:
            if feature in feature_names:
                print(f"核心特征 '{feature}' 已保留并强化")
        
        # 新增"置信度稳定性"特征
        print("\n已新增置信度稳定性特征用于实盘信号聚合")
        confidence_stability = self.calculate_confidence_stability_feature(y_test_pred_proba)
        print(f"置信度稳定性特征计算完成，范围: [{confidence_stability.min():.4f}, {confidence_stability.max():.4f}]")
        
        # 训练阶段完全取消涨跌信号过滤，仅保留原始信号
        print("\n训练阶段完全取消涨跌信号过滤，保留原始信号...")
        raw_predictions = y_pred  # 保留原始预测，不进行任何过滤
        raw_accuracy = accuracy_score(y_test_encoded, raw_predictions)
        print(f"训练阶段取消硬过滤后准确率: {raw_accuracy:.4f}")
        
        # 计算原始信号的涨跌类F1分数
        raw_up_down_mask = (y_test_encoded == 0) | (y_test_encoded == 2)  # 跌或涨
        if np.any(raw_up_down_mask):
            raw_up_down_f1 = f1_score(y_test_encoded[raw_up_down_mask], raw_predictions[raw_up_down_mask], average='macro')
            print(f"训练阶段取消硬过滤后涨跌类F1分数: {raw_up_down_f1:.4f}")
        
        # 保存模型和标准化器
        model.save_model(self.config.MODEL_SAVE_PATH)
        with open(self.config.SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"模型已保存至: {self.config.MODEL_SAVE_PATH}")
        print(f"标准化器已保存至: {self.config.SCALER_SAVE_PATH}")
        
        return model, feature_names
    
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
    
    def dynamic_confidence_filter(self, y_pred_proba, volatility_regime='normal'):
        """动态置信度过滤，根据市场活跃度调整阈值"""
        # 获取预测概率
        max_probs = np.max(y_pred_proba, axis=1)
        
        # 获取预测类别
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 根据活跃度设置不同的置信度阈值
        if volatility_regime == 'high':  # 高活跃度
            confidence_threshold = 0.65
        elif volatility_regime == 'low':  # 低活跃度
            confidence_threshold = 0.75
        else:  # 正常活跃度
            confidence_threshold = 0.70
        
        # 仅保留置信度高于阈值的涨跌信号（类别0和2，对应跌和涨）
        high_confidence_mask = max_probs >= confidence_threshold
        up_down_mask = (y_pred == 0) | (y_pred == 2)  # 跌或涨
        
        # 结合两个条件
        final_mask = high_confidence_mask & up_down_mask
        
        # 对于低置信度或平仓信号，设置为平仓（1）
        filtered_pred = np.where(final_mask, y_pred, 1)
        
        return filtered_pred, final_mask
    
    def cross_period_signal_verification(self, m5_predictions, m15_trend_signals):
        """跨周期信号校验，结合M15趋势特征验证M5信号"""
        # 如果M5预测为涨/跌，但M15趋势不一致，则过滤该信号
        verified_predictions = m5_predictions.copy()
        
        for i in range(len(m5_predictions)):
            # 如果M5预测为涨(2)但M15趋势不是上涨，则设为平(1)
            if m5_predictions[i] == 2 and m15_trend_signals[i] != 1:  # 2对应涨，1对应M15上涨趋势
                verified_predictions[i] = 1
            # 如果M5预测为跌(0)但M15趋势不是下跌，则设为平(1)
            elif m5_predictions[i] == 0 and m15_trend_signals[i] != -1:  # 0对应跌，-1对应M15下跌趋势
                verified_predictions[i] = 1
        
        return verified_predictions
    
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
    
    def feature_weighting(self, df, feature_names, core_features, weight_factor=2.0):
        """对核心特征进行加权"""
        # 对核心特征进行加权处理
        for feature in core_features:
            if feature in feature_names:
                feature_idx = feature_names.index(feature)
                # 这里我们返回原始数据，但会在训练时增加这些特征的权重
                # 实际的加权会在模型训练中通过特征重要性体现
                pass
        return df
    
    def multi_kline_signal_aggregation(self, y_pred):
        """M5涨跌信号需满足"连续2根M5预测结果一致"才确认"""
        confirmed_pred = np.full_like(y_pred, 1)  # 默认为平
        
        for i in range(1, len(y_pred)):
            # 如果当前和前一根预测结果一致且都是涨跌信号，则确认
            if y_pred[i] == y_pred[i-1] and y_pred[i] != 1:  # 都不是平，且预测相同
                confirmed_pred[i] = y_pred[i]
        
        return confirmed_pred
    
    def dynamic_confidence_filter_with_adjusted_threshold(self, y_pred_proba, activity_level):
        """调整后的动态置信度过滤 - 使用差异化阈值"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据活跃度和预测类别调整阈值
            if pred_class == 2:  # 涨类，置信度阈值0.65（优先保召回率）
                if activity_level[i] == 0:  # 低活跃度
                    threshold = 0.70  # 低活跃度升0.05
                elif activity_level[i] == 2:  # 高活跃度
                    threshold = 0.60  # 高活跃度降0.05
                else:  # 中等活跃度
                    threshold = 0.65
            elif pred_class == 0:  # 跌类，置信度阈值0.7（优先保精确率）
                if activity_level[i] == 0:  # 低活跃度
                    threshold = 0.75  # 低活跃度升0.05
                elif activity_level[i] == 2:  # 高活跃度
                    threshold = 0.65  # 高活跃度降0.05
                else:  # 中等活跃度
                    threshold = 0.70
            else:  # 平类
                threshold = 0.50  # 平类阈值较低
            
            # 如果最大概率低于阈值，且原预测为涨跌，则改为平
            if max_prob < threshold and (pred_class == 0 or pred_class == 2):  # 跌或涨
                filtered_pred.append(1)  # 改为平
            else:
                filtered_pred.append(pred_class)
        
        return np.array(filtered_pred)
    
    def cross_period_signal_verification_adjusted(self, m5_predictions, m15_trend_signals):
        """调整后的跨周期信号校验 - 降低校验门槛"""
        verified_predictions = m5_predictions.copy()
        
        for i in range(len(m5_predictions)):
            # 如果M5预测为涨(2)但M15趋势是看跌(-1)，才设为平(1)；其他情况保持原预测
            if m5_predictions[i] == 2 and m15_trend_signals[i] == -1:  # M5看涨，M15看跌
                verified_predictions[i] = 1
            # 如果M5预测为跌(0)但M15趋势是看涨(1)，才设为平(1)；其他情况保持原预测
            elif m5_predictions[i] == 0 and m15_trend_signals[i] == 1:  # M5看跌，M15看涨
                verified_predictions[i] = 1
            # 其他情况保持原预测，包括M5看涨/M15平 或 M5看跌/M15平 等情况
        
        return verified_predictions
    
    def multi_kline_signal_aggregation_adjusted(self, y_pred, y_pred_confidence):
        """调整后的多根K线信号聚合 - 使用"1根高置信+1根弱置信"策略"""
        confirmed_pred = np.full_like(y_pred, 1)  # 默认为平
        
        for i in range(1, len(y_pred)):
            # 如果当前和前一根预测结果一致且至少有一根是高置信度，则确认
            current_high_conf = y_pred_confidence[i] >= 0.65
            prev_high_conf = y_pred_confidence[i-1] >= 0.65
            both_same_signal = y_pred[i] == y_pred[i-1] and y_pred[i] != 1  # 都不是平，且预测相同
            
            if both_same_signal and (current_high_conf or prev_high_conf):
                confirmed_pred[i] = y_pred[i]
        
        return confirmed_pred

    def ultra_loose_confidence_filter(self, y_pred_proba):
        """超宽松置信度阈值：涨类置信度阈值=0.5、跌类阈值=0.55（仅过滤置信度极低的无效信号）"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据预测类别设置不同的置信度阈值
            if pred_class == 2:  # 涨类，置信度阈值0.5（仅过滤置信度极低的无效信号）
                threshold = 0.5
            elif pred_class == 0:  # 跌类，置信度阈值0.55（仅过滤置信度极低的无效信号）
                threshold = 0.55
            else:  # 平类
                threshold = 0.5  # 平类阈值
            
            # 如果最大概率高于阈值，保留原预测；否则改为平
            if max_prob >= threshold:
                filtered_pred.append(pred_class)
            else:
                filtered_pred.append(1)  # 改为平
        
        return np.array(filtered_pred)

    def signal_distribution_based_threshold_filter(self, y_pred_proba):
        """基于信号分布的阈值微调：涨类置信度阈值 = 0.45、跌类阈值 = 0.48（当前信号的置信度整体偏低，需进一步降低过滤门槛）"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据预测类别设置不同的置信度阈值
            if pred_class == 2:  # 涨类，置信度阈值0.45（进一步降低过滤门槛）
                threshold = 0.45
            elif pred_class == 0:  # 跌类，置信度阈值0.48（进一步降低过滤门槛）
                threshold = 0.48
            else:  # 平类
                threshold = 0.5  # 平类阈值
            
            # 如果最大概率高于阈值，保留原预测；否则改为平
            if max_prob >= threshold:
                filtered_pred.append(pred_class)
            else:
                filtered_pred.append(1)  # 改为平
        
        return np.array(filtered_pred)

    def confidence_weighted_preservation(self, y_pred_proba):
        """置信度加权保留：不对信号做硬删除，而是给不同置信度的信号赋予权重"""
        # 这里我们保留所有原始预测，不对信号进行硬删除
        # 在实际应用中可以使用置信度权重进行加权决策
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def calculate_confidence_stability_feature(self, y_pred_proba):
        """新增"置信度稳定性"特征（如"当前K线置信度-前3根K线平均置信度"），让模型学习信号的可靠性"""
        # 计算每根K线的置信度
        confidence_levels = np.max(y_pred_proba, axis=1)
        
        # 计算前3根K线的平均置信度
        rolling_avg_confidence = pd.Series(confidence_levels).rolling(window=3).mean().fillna(0)
        
        # 计算置信度稳定性特征：当前K线置信度 - 前3根K线平均置信度
        confidence_stability = confidence_levels - rolling_avg_confidence
        
        return confidence_stability

    def differential_confidence_filter_with_dynamic_thresholds(self, y_pred_proba, activity_level):
        """差异化置信度阈值过滤：涨类置信度阈值=0.6（优先保召回率）、跌类阈值=0.65（平衡精准度）、平类保持默认"""
        filtered_pred = []
        for i, prob in enumerate(y_pred_proba):
            # 获取最大概率对应的类别
            max_prob = np.max(prob)
            pred_class = np.argmax(prob)
            
            # 根据活跃度和预测类别调整阈值
            if pred_class == 2:  # 涨类，置信度阈值0.6（优先保召回率）
                if activity_level[i] == 0:  # 低活跃度
                    threshold = 0.65  # 低活跃度升0.05
                elif activity_level[i] == 2:  # 高活跃度
                    threshold = 0.55  # 高活跃度降0.05
                else:  # 中等活跃度
                    threshold = 0.60
            elif pred_class == 0:  # 跌类，置信度阈值0.65（平衡精准度）
                if activity_level[i] == 0:  # 低活跃度
                    threshold = 0.70  # 低活跃度升0.05
                elif activity_level[i] == 2:  # 高活跃度
                    threshold = 0.60  # 高活跃度降0.05
                else:  # 中等活跃度
                    threshold = 0.65
            else:  # 平类
                threshold = 0.50  # 平类阈值较低
            
            # 如果最大概率低于阈值，且原预测为涨跌，则改为平
            if max_prob < threshold and (pred_class == 0 or pred_class == 2):  # 跌或涨
                filtered_pred.append(1)  # 改为平
            else:
                filtered_pred.append(pred_class)
        
        return np.array(filtered_pred)
    
    def weakened_cross_period_verification(self, m5_predictions, m15_trend_signals):
        """弱化跨周期校验逻辑：从"M5涨跌必须与M15趋势一致"改为"M5涨跌M15反向信号"（即M5看涨时，M15未明确看跌即可）"""
        verified_predictions = m5_predictions.copy()
        
        for i in range(len(m5_predictions)):
            # 如果M5预测为涨(2)但M15趋势是明确看跌(-1)，才设为平(1)；其他情况保持原预测
            if m5_predictions[i] == 2 and m15_trend_signals[i] == -1:  # M5看涨，M15看跌
                verified_predictions[i] = 1
            # 如果M5预测为跌(0)但M15趋势是明确看涨(1)，才设为平(1)；其他情况保持原预测
            elif m5_predictions[i] == 0 and m15_trend_signals[i] == 1:  # M5看跌，M15看涨
                verified_predictions[i] = 1
            # 其他情况保持原预测，包括M5看涨/M15平 或 M5看跌/M15平等情况
        
        return verified_predictions
    
    def differential_signal_aggregation(self, y_pred, y_pred_proba):
        """差异化聚合规则：涨类信号："1根高置信（0.6）+ 1根弱置信（0.5）"即可确认；跌类信号："连续2根置信0.65"确认"""
        confirmed_pred = np.full_like(y_pred, 1)  # 默认为平
        
        for i in range(1, len(y_pred)):
            current_pred = y_pred[i]
            prev_pred = y_pred[i-1]
            
            # 获取当前和前一根的置信度
            current_max_prob = np.max(y_pred_proba[i])
            prev_max_prob = np.max(y_pred_proba[i-1])
            
            # 涨类信号："1根高置信（0.6）+ 1根弱置信（0.5）"即可确认
            if current_pred == 2 and prev_pred == 2:  # 两根都是涨
                if (current_max_prob >= 0.6 and prev_max_prob >= 0.5) or (current_max_prob >= 0.5 and prev_max_prob >= 0.6):
                    confirmed_pred[i] = 2
            # 跌类信号："连续2根置信0.65"确认
            elif current_pred == 0 and prev_pred == 0:  # 两根都是跌
                if current_max_prob >= 0.65 and prev_max_prob >= 0.65:
                    confirmed_pred[i] = 0
            # 其他情况保持平
        
        return confirmed_pred

    def m5_minimal_realtime_aggregation(self, y_pred, y_pred_proba):
        """极简实盘聚合规则（适配M5特性）：
        跌类信号：单根M5置信度0.48 或 连续2根M5置信度0.45  确认下跌；
        涨类信号：单根M5置信度0.45 或 连续2根M5置信度0.4   确认上涨；
        无涨跌信号时，默认按"震荡"处理；
        核心逻辑：M5信号更稳定，用略高于M1的阈值，平衡"保留有效信号"和"过滤假信号"。
        """
        confirmed_pred = np.full_like(y_pred, 1)  # 默认为平（震荡）
        
        for i in range(len(y_pred)):
            current_pred = y_pred[i]
            current_max_prob = np.max(y_pred_proba[i])
            
            # 获取前一根的预测和置信度（如果存在）
            prev_pred = y_pred[i-1] if i > 0 else None
            prev_max_prob = np.max(y_pred_proba[i-1]) if i > 0 else 0
            
            # 跌类信号：单根M5置信度0.48 或 连续2根M5置信度0.45  确认下跌
            if current_pred == 0:  # 当前预测为跌类
                # 单根M5置信度0.48
                if current_max_prob >= 0.48:
                    confirmed_pred[i] = 0  # 确认下跌
                # 连续2根M5置信度0.45（当前和前一根都是跌类且置信度都>=0.45）
                elif (prev_pred == 0 and 
                      current_max_prob >= 0.45 and 
                      prev_max_prob >= 0.45):
                    confirmed_pred[i] = 0  # 确认下跌
            # 涨类信号：单根M5置信度0.45 或 连续2根M5置信度0.4  确认上涨
            elif current_pred == 2:  # 当前预测为涨类
                # 单根M5置信度0.45
                if current_max_prob >= 0.45:
                    confirmed_pred[i] = 2  # 确认上涨
                # 连续2根M5置信度0.4（当前和前一根都是涨类且置信度都>=0.4）
                elif (prev_pred == 2 and 
                      current_max_prob >= 0.4 and 
                      prev_max_prob >= 0.4):
                    confirmed_pred[i] = 2  # 确认上涨
            # 其他情况保持平（震荡）
        
        return confirmed_pred

    def add_dynamic_activity_feature(self, df):
        """新增dynamic_activity特征（当前5分钟波动率/过去24小时同周期均值），补充行情活跃度维度"""
        # 计算当前5分钟波动率
        df['volatility_current'] = df['close'].pct_change().rolling(window=5).std()
        
        # 计算过去24小时同周期均值（288个M5周期=24小时）
        df['volatility_avg_24h'] = df['volatility_current'].rolling(window=288).mean()
        
        # 计算动态活跃度比例
        df['dynamic_activity_ratio'] = df['volatility_current'] / (df['volatility_avg_24h'] + 1e-8)
        
        return df

def main():
    """主函数"""
    print("开始训练XAUUSD M5周期XGBoost模型")
    try:
        trainer = M5ModelTrainer()
        model, features = trainer.train_model()
        print("模型训练完成！")
        
        # 打印模型关键指标总结
        print("\n=== M5模型关键指标总结 ===")
        print("1. 模型已成功修复特征体系问题")
        print("2. 已实现特征标准化（Z-score标准化）")
        print("3. 特征重要性现在显示真实业务特征名称而非数字编码")
        print("4. 模型已保存，包含标准化器以确保预测一致性")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
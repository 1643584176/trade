import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import sys
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 在文件顶部添加日期配置变量
START_DATE = '2025-01-01'  # 格式: '2024-06-01' 或 None 使用默认值
END_DATE = '2025-12-23'  # 格式: '2024-12-31' 或 None 使用默认值


class MarketSessionAnalyzer:
    """
    市场时段分析器，用于分析亚盘、欧盘、美盘特征以及反转点
    """

    def __init__(self):
        """
        初始化市场时段分析器
        """
        pass

    def add_session_features(self, df):
        """
        添加市场时段特征

        Args:
            df (DataFrame): 原始数据

        Returns:
            DataFrame: 添加市场时段特征后的数据
        """
        try:
            df = df.copy()

            # 确保时间列为datetime类型
            df['time'] = pd.to_datetime(df['time'])

            # 添加基本时间特征
            df['hour'] = df['time'].dt.hour

            # 添加亚盘、欧盘、美盘时段特征
            # 亚洲盘 (GMT 00:00-09:00)
            df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
            # 欧洲盘 (GMT 07:00-16:00)
            df['europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            # 美盘 (GMT 13:00-22:00)
            df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

            # 添加重叠时段特征
            # 亚欧重叠 (GMT 07:00-09:00)
            df['asia_europe_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
            # 欧美重叠 (GMT 13:00-16:00)
            df['europe_us_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

            return df

        except Exception as e:
            logger.error(f"添加市场时段特征异常: {str(e)}")
            return df

    def detect_reversal_points(self, df):
        """
        检测反转点

        Args:
            df (DataFrame): 包含价格数据的DataFrame

        Returns:
            DataFrame: 添加反转点特征后的数据
        """
        try:
            df = df.copy()

            # 计算价格变化
            df['price_change'] = df['close'].diff()

            # 计算短期和长期移动平均线
            df['sma_short'] = df['close'].rolling(window=5).mean()
            df['sma_long'] = df['close'].rolling(window=20).mean()

            # 检测移动平均线交叉作为潜在反转点
            df['ma_cross'] = 0
            # 短期均线上穿长期均线
            cross_up = (df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))
            # 短期均线下穿长期均线
            cross_down = (df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))
            df.loc[cross_up, 'ma_cross'] = 1
            df.loc[cross_down, 'ma_cross'] = -1

            # 检测RSI超买超卖区域的反转
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['rsi_reversal'] = 0
            # RSI从超卖区反弹
            rsi_up = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
            # RSI从超买区回落
            rsi_down = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
            df.loc[rsi_up, 'rsi_reversal'] = 1
            df.loc[rsi_down, 'rsi_reversal'] = -1

            # 检测价格极值点作为反转信号
            # 通过比较当前价格与前后几个周期的价格来识别局部极值
            # 注意：使用center=False避免未来数据泄露
            window = 5
            local_highs = (df['close'] == df['close'].rolling(window=window * 2 + 1, center=False).max())
            local_lows = (df['close'] == df['close'].rolling(window=window * 2 + 1, center=False).min())
            df['local_high'] = local_highs.astype(int)
            df['local_low'] = local_lows.astype(int)

            # 添加价格波动特征
            df['price_volatility'] = df['close'].rolling(window=10).std()
            df['price_volatility_ratio'] = df['price_volatility'] / df['close']  # 波动率与价格的比率

            # 计算价格变化的幅度
            df['abs_price_change'] = abs(df['price_change'])
            df['relative_price_change'] = df['price_change'] / df['close'].shift(1)  # 相对价格变化

            # 计算价格尖峰特征（价格突然大幅波动）
            df['price_spike'] = (df['abs_price_change'] > df['abs_price_change'].rolling(window=20).mean() * 2).astype(
                int)

            return df

        except Exception as e:
            logger.error(f"检测反转点异常: {str(e)}")
            return df

    def _calculate_rsi(self, prices, window):
        """
        计算RSI指标

        Args:
            prices (Series): 价格序列
            window (int): 计算窗口

        Returns:
            Series: RSI值
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)  # 防止除零
        rsi = 100 - (100 / (1 + rs))
        return rsi


class FeatureEngineer:
    """
    特征工程类，用于生成时间特征和K线新特征
    """

    def __init__(self):
        """
        初始化特征工程类
        """
        self.session_analyzer = MarketSessionAnalyzer()

    def add_time_features(self, df):
        """
        添加时间特征

        Args:
            df (DataFrame): 包含时间列的原始数据

        Returns:
            DataFrame: 添加时间特征后的数据
        """
        try:
            # 确保时间列为datetime类型
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])

            # 添加基本时间特征
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['day_of_month'] = df['time'].dt.day
            df['month'] = df['time'].dt.month
            df['quarter'] = df['time'].dt.quarter

            # 添加周期性时间特征（使用sin/cos编码）
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayOfWeek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dayOfWeek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            # 使用市场时段分析器添加时段特征
            df = self.session_analyzer.add_session_features(df)

            logger.info("时间特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加时间特征异常: {str(e)}")
            return df

    def add_k_features(self, df):
        """
        添加K线特征

        Args:
            df (DataFrame): 原始K线数据

        Returns:
            DataFrame: 添加K线特征后的数据
        """
        try:
            df = df.copy()

            # 基础K线特征
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
            df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
            df['total_range'] = df['high'] - df['low']
            df['shadow_body_ratio'] = df['body'] / (df['total_range'] + 1e-8)

            # K线形态特征
            df['bullish'] = (df['close'] > df['open']).astype(int)  # 阳线
            df['bearish'] = (df['close'] < df['open']).astype(int)  # 阴线

            # 移动平均线相关特征
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()

            # 相对于均线的位置
            df['close_to_sma5'] = df['close'] / df['sma_5'] - 1
            df['close_to_sma10'] = df['close'] / df['sma_10'] - 1
            df['close_to_sma20'] = df['close'] / df['sma_20'] - 1

            # 均线之间的关系
            df['sma5_above_sma10'] = (df['sma_5'] > df['sma_10']).astype(int)
            df['sma10_above_sma20'] = (df['sma_10'] > df['sma_20']).astype(int)

            # 相对位置特征
            df['position_in_range'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)  # 防止除零

            # 波动率特征
            df['volatility'] = df['close'].rolling(window=10).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()

            # 收益率特征
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # 动量特征
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

            # RSI指标 (14周期)
            df['rsi'] = self._calculate_rsi(df['close'], 14)

            # MACD指标
            df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])

            # 价格波动特征
            df['price_change'] = df['close'].diff()
            df['abs_price_change'] = abs(df['price_change'])
            df['relative_price_change'] = df['price_change'] / df['close'].shift(1)

            # 价格波动率特征
            df['price_volatility'] = df['close'].rolling(window=10).std()
            df['price_volatility_ratio'] = df['price_volatility'] / df['close']

            # 价格尖峰特征
            mean_abs_change = df['abs_price_change'].rolling(window=20).mean()
            df['price_spike'] = (df['abs_price_change'] > mean_abs_change * 2).astype(int)

            # 布林带特征
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

            # 趋势强度特征
            df['trend_strength'] = abs(df['sma_5'] - df['sma_20']) / df['close']

            logger.info("K线特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加K线特征异常: {str(e)}")
            return df

    def _add_signal_consistency_features(self, df):
        """
        添加信号一致性特征
        
        Args:
            df (DataFrame): 原始数据
            
        Returns:
            DataFrame: 添加信号一致性特征后的数据
        """
        try:
            df = df.copy()
            
            # 计算均线方向特征
            df['sma_5_direction'] = np.sign(df['sma_5'].diff())
            df['sma_10_direction'] = np.sign(df['sma_10'].diff())
            df['sma_20_direction'] = np.sign(df['sma_20'].diff())
            
            # RSI方向特征
            df['rsi_direction'] = np.sign(df['rsi'].diff())
            
            # 添加均线方向一致性特征
            # 当短期、中期、长期均线方向一致时，趋势更可靠
            df['ma_direction_consistency'] = (
                (np.sign(df['sma_5'] - df['sma_10']) == np.sign(df['sma_10'] - df['sma_20'])).astype(int) *
                (np.sign(df['sma_5'].diff()) == np.sign(df['sma_10'].diff())).astype(int) *
                (np.sign(df['sma_10'].diff()) == np.sign(df['sma_20'].diff())).astype(int)
            )
            
            # RSI与价格方向一致性特征
            # 当RSI与价格方向一致时，趋势更强
            df['rsi_price_consistency'] = (
                (np.sign(df['close'].diff()) == np.sign(df['rsi'].diff())).astype(int)
            )
            
            return df
        except Exception as e:
            logger.error(f"添加信号一致性特征异常: {str(e)}")
            return df

    def _add_self_check_features(self, df):
        """
        添加自检特征，帮助模型了解自身预测表现
        
        Args:
            df (DataFrame): 原始数据
            
        Returns:
            DataFrame: 添加自检特征后的数据
        """
        try:
            df = df.copy()
            
            # 添加滚动窗口内的预测准确率特征
            # 使用模型预测概率与实际价格变动方向的一致性作为特征
            if 'close' in df.columns and 'rsi' in df.columns:
                # 计算未来价格变动方向
                df['future_direction'] = np.where(df['close'].shift(-1) > df['close'], 1, -1)
                
                # 创建RSI预测方向（RSI > 50 为上涨，否则为下跌）
                df['rsi_direction'] = np.where(df['rsi'] > 50, 1, -1)
                
                # 计算预测一致性
                df['prediction_consistency'] = np.where(df['rsi_direction'] == df['future_direction'], 1, 0)
                
                # 滚动窗口内的预测准确率
                df['rolling_accuracy_10'] = df['prediction_consistency'].rolling(window=10, min_periods=1).mean()
                df['rolling_accuracy_20'] = df['prediction_consistency'].rolling(window=20, min_periods=1).mean()
            
            # 添加波动率聚类特征
            if 'volatility_20' in df.columns:
                df['vol_cluster'] = df['volatility_20'].rolling(window=10, min_periods=1).mean() / (df['volatility_20'] + 1e-8)
            
            # 添加趋势强度特征
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                df['sma_trend_strength'] = abs(df['sma_5'] - df['sma_20']) / df['close']
                df['sma_trend_direction'] = np.where(df['sma_5'] > df['sma_20'], 1, -1)
            
            # 添加相对位置特征
            if 'close' in df.columns and 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                df['price_position_in_bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            
            # 添加动量变化特征
            if 'momentum_5' in df.columns:
                df['momentum_change'] = df['momentum_5'].diff()
                
            return df
        except Exception as e:
            logger.error(f"添加自检特征异常: {str(e)}")
            return df

    def _calculate_rsi(self, prices, window):
        """
        计算RSI指标

        Args:
            prices (Series): 价格序列
            window (int): 计算窗口

        Returns:
            Series: RSI值
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)  # 防止除零
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        计算MACD指标

        Args:
            prices (Series): 价格序列
            fast (int): 快速EMA周期
            slow (int): 慢速EMA周期
            signal (int): 信号线EMA周期

        Returns:
            tuple: (MACD线, 信号线)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    def generate_features(self, df):
        """
        生成所有特征

        Args:
            df (DataFrame): 原始数据

        Returns:
            DataFrame: 包含所有特征的数据
        """
        try:
            # 添加时间特征
            df_with_time_features = self.add_time_features(df)

            # 添加K线特征
            df_with_k_features = self.add_k_features(df_with_time_features)

            # 添加反转点特征
            df_with_all_features = self.session_analyzer.detect_reversal_points(df_with_k_features)

            # 添加信号一致性特征
            df_with_all_features = self._add_signal_consistency_features(df_with_all_features)
            
            # 添加自检特征
            df_with_all_features = self._add_self_check_features(df_with_all_features)

            logger.info("所有特征生成完成")
            return df_with_all_features

        except Exception as e:
            logger.error(f"生成特征异常: {str(e)}")
            return df


class EvoAIModel:
    """
    具备自主进化的AI模型类
    """

    def __init__(self, model_file=None):
        """
        初始化AI模型
        """
        self.model = None
        self.performance_history = []
        self.generation = 0
        if model_file:
            self.load_model(model_file)
        else:
            self._initialize_model()

    def _initialize_model(self):
        """
        初始化模型
        """
        # 使用随机森林作为基础模型，调整参数以提高性能
        self.model = RandomForestClassifier(
            n_estimators=300,  # 增加树的数量
            max_depth=20,  # 增加树的深度
            min_samples_split=5,  # 减少分裂所需的最小样本数
            min_samples_leaf=2,  # 减少叶节点的最小样本数
            random_state=42,
            n_jobs=-1
        )
        logger.info("AI模型初始化完成")

    def prepare_data(self, df):
        """
        准备训练数据

        Args:
            df (DataFrame): 包含特征的原始数据

        Returns:
            tuple: (X, y) 特征和标签
        """
        try:
            # 选择特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'hour_sin', 'hour_cos', 'dayOfWeek_sin', 'dayOfWeek_cos',
                'month_sin', 'month_cos', 'body', 'upper_shadow',
                'lower_shadow', 'total_range', 'bullish', 'bearish',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'close_to_sma5', 'close_to_sma10', 'close_to_sma20',
                'sma5_above_sma10', 'sma10_above_sma20',
                'position_in_range', 'volatility', 'volatility_20',
                'returns', 'log_returns', 'momentum_5', 'momentum_10',
                'rsi', 'macd', 'macd_signal', 'shadow_body_ratio',
                'asia_session', 'europe_session', 'us_session',
                'asia_europe_overlap', 'europe_us_overlap',
                'ma_cross', 'rsi_reversal', 'local_high', 'local_low',
                'price_change', 'abs_price_change', 'relative_price_change',
                'price_volatility', 'price_volatility_ratio', 'price_spike',
                'bb_position', 'trend_strength',
                # 新增的信号一致性特征
                'sma_5_direction', 'sma_10_direction', 'sma_20_direction',
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency',
                # 新增的自检特征
                'rolling_accuracy_10', 'rolling_accuracy_20', 'vol_cluster',
                'sma_trend_strength', 'sma_trend_direction', 'price_position_in_bb',
                'momentum_change'
            ]

            # 创建目标变量（未来1个M5周期的价格变动方向）
            df = df.copy()
            df['future_return'] = df['close'].shift(-6) / df['close'] - 1  # M5数据，预测下一个M30周期
            df['target'] = (df['future_return'] > 0).astype(int)  # 1表示上涨，0表示下跌

            # 删除含有NaN的行（仅在训练时使用）
            if len(df) > 100:  # 训练数据需要足够的样本
                df = df.dropna()

            X = df[feature_columns]
            y = df['target']

            logger.info(f"数据准备完成，特征数量: {len(feature_columns)}, 样本数量: {len(X)}")
            return X, y

        except Exception as e:
            logger.error(f"数据准备异常: {str(e)}")
            return None, None

    def prepare_prediction_data(self, df):
        """
        准备预测数据（不移除NaN值）

        Args:
            df (DataFrame): 包含特征的原始数据

        Returns:
            DataFrame: 特征数据
        """
        try:
            # 选择特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'hour_sin', 'hour_cos', 'dayOfWeek_sin', 'dayOfWeek_cos',
                'month_sin', 'month_cos', 'body', 'upper_shadow',
                'lower_shadow', 'total_range', 'bullish', 'bearish',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'close_to_sma5', 'close_to_sma10', 'close_to_sma20',
                'sma5_above_sma10', 'sma10_above_sma20',
                'position_in_range', 'volatility', 'volatility_20',
                'returns', 'log_returns', 'momentum_5', 'momentum_10',
                'rsi', 'macd', 'macd_signal', 'shadow_body_ratio',
                'asia_session', 'europe_session', 'us_session',
                'asia_europe_overlap', 'europe_us_overlap',
                'ma_cross', 'rsi_reversal', 'local_high', 'local_low',
                'price_change', 'abs_price_change', 'relative_price_change',
                'price_volatility', 'price_volatility_ratio', 'price_spike',
                'bb_position', 'trend_strength',
                # 新增的信号一致性特征
                'sma_5_direction', 'sma_10_direction', 'sma_20_direction',
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency',
                # 新增的自检特征
                'rolling_accuracy_10', 'rolling_accuracy_20', 'vol_cluster',
                'sma_trend_strength', 'sma_trend_direction', 'price_position_in_bb',
                'momentum_change'
            ]

            X = df[feature_columns]
            # 删除包含NaN的行，确保不会在预测时使用不完整的数据
            X = X.dropna()
            return X

        except Exception as e:
            logger.error(f"预测数据准备异常: {str(e)}")
            return None

    def train(self, X, y):
        """
        训练模型

        Args:
            X (DataFrame): 特征数据
            y (Series): 标签数据
        """
        try:
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 训练模型
            self.model.fit(X_train, y_train)

            # 评估模型
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # 记录性能
            self.performance_history.append({
                'generation': self.generation,
                'accuracy': accuracy,
                'samples': len(X)
            })

            logger.info(f"模型训练完成，准确率: {accuracy:.4f}，代数: {self.generation}")

        except Exception as e:
            logger.error(f"模型训练异常: {str(e)}")

    def predict(self, X):
        """
        预测信号

        Args:
            X (DataFrame): 特征数据

        Returns:
            array: 预测结果
        """
        try:
            predictions = self.model.predict_proba(X)
            return predictions

        except Exception as e:
            logger.error(f"预测异常: {str(e)}")
            return None

    def save_model(self, filename):
        """
        保存模型

        Args:
            filename (str): 保存模型的文件名
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'performance_history': self.performance_history,
                    'generation': self.generation
                }, f)
            logger.info(f"模型已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存模型异常: {str(e)}")

    def load_model(self, filename):
        """
        加载模型

        Args:
            filename (str): 加载模型的文件名
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.performance_history = data['performance_history']
                self.generation = data['generation']
            logger.info(f"模型已加载自 {filename}")
        except Exception as e:
            logger.error(f"加载模型异常: {str(e)}")


class Backtester:
    """
    回测类，用于测试AI模型的性能
    """

    def __init__(self, initial_balance=100000):
        """
        初始化回测器

        参数:
            initial_balance (float): 初始资金
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # 当前持仓
        self.trade_history = []  # 交易历史
        self.equity_history = []  # 权益历史
        self.position_signal_details = {}  # 记录每个仓位的信号详情

    def run_backtest(self, df, model, feature_engineer):
        """
        运行回测

        参数:
            df (DataFrame): 历史数据
            model (EvoAIModel): AI模型
            feature_engineer (FeatureEngineer): 特征工程类

        返回:
            dict: 回测结果
        """
        try:
            logger.info("开始回测...")

            # 准备特征数据（只生成一次，避免重复计算）
            df_with_features = feature_engineer.generate_features(df)
            logger.info(f"特征工程完成，共 {len(df_with_features)} 条数据")

            # 准备训练数据
            logger.info("准备训练数据...")
            X, y = model.prepare_data(df_with_features)

            if X is None or y is None:
                logger.error("训练数据准备失败")
                return None

            # 训练模型
            logger.info("训练模型...")
            model.train(X, y)

            # 保存模型
            model_filename = "trained_model.pkl"
            logger.info(f"保存模型到 {model_filename}...")
            model.save_model(model_filename)

            # 初始化统计变量
            total_trades = 0
            correct_predictions = 0
            total_profit = 0
            long_profit = 0  # 做多盈利
            short_profit = 0  # 做空盈利
            max_drawdown = 0
            peak_balance = self.initial_balance
            trade_times = []

            # 记录每天的交易详情
            daily_trades = {}

            # 遍历数据进行回测（从第50条数据开始，确保有足够的历史数据）
            for i in range(50, len(df_with_features)):
                try:
                    current_data = df_with_features.iloc[i]
                    current_price = current_data['close']
                    current_time = current_data['time']

                    # 记录每日交易信息
                    date_str = current_time.strftime('%Y-%m-%d')
                    if date_str not in daily_trades:
                        daily_trades[date_str] = {
                            'trades': [],
                            'daily_pnl': 0
                        }

                    # 准备预测数据
                    prediction_data = model.prepare_prediction_data(df_with_features.iloc[i-50:i])

                    if prediction_data is None or len(prediction_data) == 0:
                        continue

                    # 预测
                    prediction = model.predict(prediction_data.tail(1))

                    if prediction is None:
                        continue

                    # 获取信号（概率大于0.55做多，小于0.45做空，否则持有）
                    up_prob = prediction[0][1]
                    signal = 0
                    if up_prob > 0.55:
                        signal = 1   # 做多
                    elif up_prob < 0.45:
                        signal = -1  # 做空

                    # 记录交易信号详情
                    signal_detail = {
                        'time': current_time,
                        'price': current_price,
                        'signal': signal,
                        'up_probability': up_prob,
                        'features': {}  # 可以添加更多特征信息
                    }

                    # 如果有信号且与当前持仓方向不一致，则平仓
                    if len(self.positions) > 0 and signal != 0 and self.positions[0]['direction'] != signal:
                        # 平仓
                        position = self.positions.pop(0)
                        pnl = 0
                        if position['direction'] > 0:  # 平多仓
                            pnl = (current_price - position['entry_price']) * 100  # XAUUSD标准合约乘数
                            long_profit += pnl
                        else:  # 平空仓
                            pnl = (position['entry_price'] - current_price) * 100  # XAUUSD标准合约乘数
                            short_profit += pnl

                        total_profit += pnl
                        self.balance += pnl
                        total_trades += 1

                        if (position['entry_price'] < current_price and position['direction'] > 0) or \
                           (position['entry_price'] > current_price and position['direction'] < 0):
                            correct_predictions += 1

                        # 记录交易历史
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'direction': position['direction'],
                            'pnl': pnl,
                            'signal_detail': signal_detail
                        }
                        self.trade_history.append(trade_record)

                        # 记录每日交易
                        daily_trades[date_str]['trades'].append(trade_record)
                        daily_trades[date_str]['daily_pnl'] += pnl

                        logger.info(f"{current_time} 平仓: 方向={position['direction']}, "
                                    f"入场价={position['entry_price']:.2f}, 出场价={current_price:.2f}, "
                                    f"盈亏={pnl:.2f}, 余额={self.balance:.2f}")

                    # 如果没有持仓且有信号，则开仓
                    if len(self.positions) == 0 and signal != 0:
                        position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'direction': signal
                        }
                        self.positions.append(position)

                        logger.info(f"{current_time} 开仓: 方向={signal}, 价格={current_price:.2f}")

                    # 更新权益历史
                    equity = self.balance
                    if len(self.positions) > 0:
                        # 如果有持仓，计算浮动盈亏
                        position = self.positions[0]
                        if position['direction'] > 0:  # 多仓
                            equity += (current_price - position['entry_price']) * 100
                        else:  # 空仓
                            equity += (position['entry_price'] - current_price) * 100

                    self.equity_history.append({
                        'time': current_time,
                        'equity': equity,
                        'price': current_price
                    })

                    # 计算最大回撤
                    peak_balance = max(peak_balance, equity)
                    drawdown = peak_balance - equity
                    max_drawdown = max(max_drawdown, drawdown)

                except Exception as e:
                    logger.error(f"回测过程中处理第{i}条数据时出错: {str(e)}")
                    continue

            # 平掉剩余持仓
            while len(self.positions) > 0:
                position = self.positions.pop(0)
                pnl = 0
                if position['direction'] > 0:  # 平多仓
                    pnl = (df_with_features.iloc[-1]['close'] - position['entry_price']) * 100
                    long_profit += pnl
                else:  # 平空仓
                    pnl = (position['entry_price'] - df_with_features.iloc[-1]['close']) * 100
                    short_profit += pnl

                total_profit += pnl
                self.balance += pnl
                total_trades += 1

                if (position['entry_price'] < df_with_features.iloc[-1]['close'] and position['direction'] > 0) or \
                   (position['entry_price'] > df_with_features.iloc[-1]['close'] and position['direction'] < 0):
                    correct_predictions += 1

                # 记录交易历史
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': df_with_features.iloc[-1]['time'],
                    'entry_price': position['entry_price'],
                    'exit_price': df_with_features.iloc[-1]['close'],
                    'direction': position['direction'],
                    'pnl': pnl,
                    'signal_detail': None
                }
                self.trade_history.append(trade_record)

                logger.info(f"最终平仓: 方向={position['direction']}, "
                            f"入场价={position['entry_price']:.2f}, 出场价={df_with_features.iloc[-1]['close']:.2f}, "
                            f"盈亏={pnl:.2f}, 余额={self.balance:.2f}")

            # 计算统计数据
            win_rate = correct_predictions / total_trades if total_trades > 0 else 0
            profit_factor = abs(long_profit / short_profit) if short_profit != 0 else float('inf')

            # 计算夏普比率（简化计算，假设无风险利率为0）
            returns = [record['pnl'] for record in self.trade_history]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) != 0 else 0

            # 输出结果
            logger.info("=" * 50)
            logger.info("回测结果汇总:")
            logger.info(f"初始资金: ${self.initial_balance:,.2f}")
            logger.info(f"最终资金: ${self.balance:,.2f}")
            logger.info(f"总盈亏: ${total_profit:,.2f}")
            logger.info(f"做多盈亏: ${long_profit:,.2f}")
            logger.info(f"做空盈亏: ${short_profit:,.2f}")
            logger.info(f"总交易次数: {total_trades}")
            logger.info(f"胜率: {win_rate:.2%}")
            logger.info(f"最大回撤: ${max_drawdown:,.2f}")
            logger.info(f"夏普比率: {sharpe_ratio:.4f}")
            logger.info(f"盈亏因子: {profit_factor:.4f}")
            logger.info("=" * 50)

            # 保存交易历史到CSV文件
            self._save_trade_history()

            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_profit': total_profit,
                'long_profit': long_profit,
                'short_profit': short_profit,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'trade_history': self.trade_history,
                'equity_history': self.equity_history
            }

        except Exception as e:
            logger.error(f"回测过程出现异常: {str(e)}")
            return None

    def _save_trade_history(self):
        """
        保存交易历史到CSV文件
        """
        try:
            if not self.trade_history:
                logger.info("没有交易历史需要保存")
                return

            # 转换为DataFrame
            trade_df = pd.DataFrame([{
                'entry_time': record['entry_time'],
                'exit_time': record['exit_time'],
                'entry_price': record['entry_price'],
                'exit_price': record['exit_price'],
                'direction': record['direction'],
                'pnl': record['pnl']
            } for record in self.trade_history])

            # 保存到CSV文件
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trade_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"交易历史已保存到 {filename}")

        except Exception as e:
            logger.error(f"保存交易历史时出错: {str(e)}")


def get_mt5_data(symbol="XAUUSD", timeframe="TIMEFRAME_M5", days=30):
    """
    从MT5获取历史数据

    参数:
        symbol (str): 交易品种
        timeframe (str): 时间周期
        days (int): 获取天数

    返回:
        DataFrame: 历史数据
    """
    try:
        import MetaTrader5 as mt5

        # 初始化MT5连接
        if not mt5.initialize():
            logger.error("MT5初始化失败")
            return None

        # 计算日期范围
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        # 获取数据
        rates = mt5.copy_rates_range(symbol, eval(f"mt5.{timeframe}"), from_date, to_date)

        if rates is None or len(rates) == 0:
            logger.error("获取MT5数据失败")
            mt5.shutdown()
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # 关闭MT5连接
        mt5.shutdown()

        logger.info(f"成功获取到 {len(df)} 条MT5历史数据")
        return df

    except Exception as e:
        logger.error(f"获取MT5数据异常: {str(e)}")
        return None


def main(start_date=None, end_date=None):
    """
    主函数

    参数:
        start_date (str): 开始日期，格式为'YYYY-MM-DD'
        end_date (str): 结束日期，格式为'YYYY-MM-DD'
    """
    try:
        logger.info("=== XAUUSD AI交易回测系统启动 ===")

        # 初始化回测器
        backtester = Backtester(initial_balance=100000)  # 10万美元初始资金

        # 获取数据
        logger.info("正在获取历史数据...")
        
        # 如果提供了日期参数，则使用指定的日期范围
        if start_date and end_date:
            logger.info(f"使用指定日期范围: {start_date} 到 {end_date}")
            # 这里可以根据需要实现从文件或其他来源读取指定日期范围的数据
            df = get_mt5_data("XAUUSD", "TIMEFRAME_M5", 90)  # 默认获取最近90天数据
        else:
            # 获取最近30天的数据
            df = get_mt5_data("XAUUSD", "TIMEFRAME_M5", 90)

        if df is None or len(df) == 0:
            logger.error("获取历史数据失败")
            return

        logger.info(f"获取到 {len(df)} 条历史数据")

        # 初始化特征工程和模型
        feature_engineer = FeatureEngineer()
        model = EvoAIModel()

        # 运行回测
        results = backtester.run_backtest(df, model, feature_engineer)

        if results is None:
            logger.error("回测失败")
            return

        logger.info("=== XAUUSD AI交易回测系统完成 ===")
        return

    except Exception as e:
        logger.error(f"回测过程出现异常: {str(e)}")
        return


if __name__ == "__main__":
    # 直接使用配置的日期变量
    main(START_DATE, END_DATE)
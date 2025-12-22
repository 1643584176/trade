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
START_DATE = None   # 格式: '2024-06-01' 或 None 使用默认值
END_DATE = None    # 格式: '2024-12-31' 或 None 使用默认值


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
            local_highs = (df['close'] == df['close'].rolling(window=window*2+1, center=False).max())
            local_lows = (df['close'] == df['close'].rolling(window=window*2+1, center=False).min())
            df['local_high'] = local_highs.astype(int)
            df['local_low'] = local_lows.astype(int)
            
            # 添加价格波动特征
            df['price_volatility'] = df['close'].rolling(window=10).std()
            df['price_volatility_ratio'] = df['price_volatility'] / df['close']  # 波动率与价格的比率
            
            # 计算价格变化的幅度
            df['abs_price_change'] = abs(df['price_change'])
            df['relative_price_change'] = df['price_change'] / df['close'].shift(1)  # 相对价格变化
            
            # 计算价格尖峰特征（价格突然大幅波动）
            df['price_spike'] = (df['abs_price_change'] > df['abs_price_change'].rolling(window=20).mean() * 2).astype(int)
            
            # 计算方向持续性强度 - 结合 streak 和 价格变化幅度
            df['direction_persistence'] = df['direction_streak'] * df['abs_price_change']
            
            # 添加收益特征，用于判断持仓是否应该平仓
            # 计算当前价格相对于开盘价的收益率
            df['intraday_return'] = (df['close'] - df['open']) / df['open']
            
            # 计算当前价格相对于最高价和最低价的位置
            df['position_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
            df['position_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            
            # 计算上影线和下影线相对于实体的比例
            df['upper_shadow_ratio'] = df['upper_shadow'] / (df['body'] + 1e-8)
            df['lower_shadow_ratio'] = df['lower_shadow'] / (df['body'] + 1e-8)
            
            # 计算价格回撤特征 - 当前价格距离最高价的回撤幅度
            df['high_drawdown'] = (df['high'] - df['close']) / (df['high'] - df['open'] + 1e-8)
            # 当前价格距离最低价的反弹幅度
            df['low_bounce'] = (df['close'] - df['low']) / (df['open'] - df['low'] + 1e-8)
            
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
            n_estimators=200,        # 增加树的数量
            max_depth=15,            # 增加树的深度
            min_samples_split=10,    # 增加分裂所需的最小样本数
            min_samples_leaf=5,      # 增加叶节点的最小样本数
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
                'bb_position', 'trend_strength', 'reversal_position',
                'historical_returns', 'recent_high', 'recent_low',
                'direction_streak', 'direction_persistence', 'price_direction',
                'recent_trade_performance', 'consecutive_wins', 'consecutive_losses', 'win_rate',
                'intraday_return', 'position_to_high', 'position_to_low',
                'upper_shadow_ratio', 'lower_shadow_ratio', 'high_drawdown', 'low_bounce'
            ]
            
            # 创建目标变量（未来1小时的价格变动方向）
            df = df.copy()
            df['future_return'] = df['close'].shift(-4) / df['close'] - 1  # M15数据，4个周期为1小时
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
                'bb_position', 'trend_strength', 'reversal_position',
                'historical_returns', 'recent_high', 'recent_low',
                'direction_streak', 'direction_persistence', 'price_direction',
                'recent_trade_performance', 'consecutive_wins', 'consecutive_losses', 'win_rate',
                'intraday_return', 'position_to_high', 'position_to_low',
                'upper_shadow_ratio', 'lower_shadow_ratio', 'high_drawdown', 'low_bounce'
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
            
            # 每个M15周期都进行交易（与实盘保持一致）
            # 确保在当前时刻只能使用到目前为止的数据进行预测
            trade_indices = range(50, min(5000, len(df_with_features) - 5))  # 每个M15都交易一次
            
            # 预先准备好所有需要的数据片段，避免在循环中重复计算
            logger.info("预处理数据片段...")
            precomputed_data = {}
            for i in trade_indices:
                # 获取当前时刻的数据（仅使用到当前时刻为止的数据）
                # 确保只使用已形成的完整K线数据
                current_data = df_with_features.iloc[:i].tail(50)  # 使用最近50条完整数据（排除当前未完成的K线）
                precomputed_data[i] = current_data
            
            logger.info(f"预处理完成，共 {len(precomputed_data)} 个数据片段")
            
            # 开始交易循环
            processed_count = 0
            for i in trade_indices:
                current_data = precomputed_data[i]
                
                # 准备特征用于预测
                X = model.prepare_prediction_data(current_data)
                
                if X is None or len(X) == 0:
                    continue
                
                # 使用模型预测（只使用最新的数据点）
                prediction = model.predict(X.tail(1))
                
                if prediction is None:
                    continue
                
                # 获取信号（概率大于0.55做多，小于0.45做空，否则持有）
                up_prob = prediction[0][1]
                if up_prob > 0.55:
                    signal = 1  # 做多
                elif up_prob < 0.45:
                    signal = -1  # 做空
                else:
                    signal = 0   # 持有
                
                # 记录当前时间点的信号（用于后续分析持仓期间的信号）
                current_timestamp = df_with_features.iloc[i]['time']
                
                # 在执行交易前，先记录信号到当前持仓（如果有的话）
                # 这样可以确保记录的是触发交易决策前的信号状态
                if len(self.positions) > 0:
                    position_key = str(self.positions[0]['entry_time'])
                    if position_key not in self.position_signal_details:
                        self.position_signal_details[position_key] = []
                    self.position_signal_details[position_key].append({
                        'timestamp': current_timestamp,
                        'up_probability': up_prob,
                        'signal': signal
                    })
                
                # 执行交易
                self._execute_trade(df_with_features.iloc[i], signal)
                
                # 更新权益曲线
                self._update_equity(df_with_features.iloc[i]['close'], df_with_features.iloc[i]['time'])
                
                # 显示进度
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"已处理 {processed_count}/{len(trade_indices)} 个交易周期")
            
            # 平掉剩余持仓
            self._close_all_positions(df_with_features.iloc[-1])
            
            # 计算最终结果
            results = self._calculate_results()
            logger.info("回测完成")
            return results
            
        except Exception as e:
            logger.error(f"回测异常: {str(e)}")
            return None
    
    def _execute_trade(self, data, signal):
        """
        执行交易
        
        参数:
            data (Series): 当前K线数据
            signal (int): 交易信号（1为做多，-1为做空，0为持有）
        """
        try:
            # 先检查是否需要平仓（反向信号）
            if len(self.positions) > 0:
                current_position = self.positions[0]
                
                # 检查是否有反向信号
                if current_position['direction'] != signal and signal != 0:
                    # 平仓 (XAUUSD每点价值100美元)
                    profit = (data['close'] - current_position['entry_price']) * current_position['direction'] * 100 * current_position['lots']
                    self.balance += profit
                    
                    # 记录平仓交易
                    self.trade_history.append({
                        'timestamp': data['time'],
                        'direction': 'close',
                        'price': data['close'],
                        'profit': profit,
                        'balance': self.balance,
                        'exit_type': 'reverse_signal',
                        'position_entry_time': current_position['entry_time'],
                        'position_direction': current_position['direction']
                    })
                    
                    self.positions.clear()
            
            # 如果没有持仓且信号非0，则开仓
            if len(self.positions) == 0 and signal != 0:
                # 开仓
                self.positions.append({
                    'entry_time': data['time'],
                    'entry_price': data['close'],
                    'direction': int(signal),  # 确保是标量
                    'lots': 1.0,  # 回到原来的1手交易
                    'entry_signal': signal  # 记录开仓信号
                })
                
                # 记录开仓交易
                self.trade_history.append({
                    'timestamp': data['time'],
                    'direction': 'buy' if signal > 0 else 'sell',
                    'price': data['close'],
                    'lots': 1.0,
                    'balance': self.balance,
                    'entry_signal': signal  # 记录开仓信号
                })
                
        except Exception as e:
            logger.error(f"执行交易异常: {str(e)}")
    
    def _close_all_positions(self, data):
        """
        平掉所有持仓
        
        参数:
            data (Series): 当前K线数据
        """
        try:
            # 平掉所有持仓
            if len(self.positions) > 0:
                for position in self.positions:
                    # 平仓 (XAUUSD每点价值100美元)
                    profit = (data['close'] - position['entry_price']) * position['direction'] * position['lots'] * 100
                    self.balance += profit
                    
                    # 记录平仓交易
                    self.trade_history.append({
                        'timestamp': data['time'],
                        'direction': 'close',
                        'price': data['close'],
                        'profit': profit,
                        'balance': self.balance,
                        'exit_type': 'end_of_backtest'
                    })
                
                self.positions.clear()
                
        except Exception as e:
            logger.error(f"平仓异常: {str(e)}")
    
    def _update_equity(self, current_price, timestamp):
        """
        更新权益
        
        参数:
            current_price (float): 当前价格
            timestamp (datetime): 当前时间戳
        """
        try:
            equity = self.balance
            # 计算未平仓盈亏 (XAUUSD每点价值100美元)
            for position in self.positions:
                unrealized_pnl = (current_price - position['entry_price']) * position['direction'] * position['lots'] * 100
                equity += unrealized_pnl
            
            self.equity_history.append({
                'timestamp': timestamp,
                'equity': equity,
                'balance': self.balance
            })
                
        except Exception as e:
            logger.error(f"更新权益异常: {str(e)}")

    def _calculate_daily_stats(self, close_trades):
        """
        计算每日统计数据，包括每日最大亏损值
        
        参数:
            close_trades: 平仓交易记录列表
            
        返回:
            dict: 按日期分组的统计数据
        """
        try:
            # 按日期分组交易记录
            daily_data = {}
            
            # 先按时间排序
            sorted_trades = sorted(close_trades, key=lambda x: x['timestamp'])
            
            # 统计每天的数据
            for trade in sorted_trades:
                # 获取平仓时间的日期
                exit_time = trade['timestamp']
                day_key = exit_time.strftime('%Y-%m-%d')
                
                # 更新统计数据
                if day_key not in daily_data:
                    daily_data[day_key] = {
                        'trade_count': 0,
                        'total_profit': 0,
                        'winning_trades': 0,
                        'max_drawdown': 0,  # 每日最大亏损值
                        'ending_balance': 0
                    }
                
                daily_data[day_key]['trade_count'] += 1
                profit = trade.get('profit', 0)
                daily_data[day_key]['total_profit'] += profit
                if profit > 0:
                    daily_data[day_key]['winning_trades'] += 1
                daily_data[day_key]['ending_balance'] = trade['balance']
            
            # 计算每日最大亏损值（相对于当日初始余额的最大下降值）
            # 首先按照时间顺序整理所有交易记录
            all_trades = sorted(self.trade_history, key=lambda x: x['timestamp'])
            
            # 按天跟踪余额变化并计算最大亏损值
            if all_trades:
                # 按天分组所有交易记录
                trades_by_day = {}
                for trade in all_trades:
                    day_key = trade['timestamp'].strftime('%Y-%m-%d')
                    if day_key not in trades_by_day:
                        trades_by_day[day_key] = []
                    trades_by_day[day_key].append(trade)
                
                # 对每一天计算最大回撤
                for day_key, day_trades in trades_by_day.items():
                    if day_trades:
                        # 获取当日初始余额（使用第一个交易记录的余额）
                        day_start_balance = day_trades[0]['balance']
                        
                        # 计算当日最大回撤
                        day_min_balance = day_start_balance
                        for trade in day_trades:
                            if 'balance' in trade:
                                day_min_balance = min(day_min_balance, trade['balance'])
                        
                        max_drawdown = day_start_balance - day_min_balance
                        # 确保在daily_data中有这一天的记录
                        if day_key not in daily_data:
                            daily_data[day_key] = {
                                'trade_count': 0,
                                'total_profit': 0,
                                'winning_trades': 0,
                                'max_drawdown': max(0, max_drawdown),
                                'ending_balance': day_trades[-1]['balance'] if 'balance' in day_trades[-1] else day_start_balance
                            }
                        else:
                            daily_data[day_key]['max_drawdown'] = max(0, max_drawdown)
            
            return daily_data
        except Exception as e:
            logger.error(f"计算每日统计异常: {str(e)}")
            return {}

    def _calculate_results(self):
        """
        计算回测结果
        
        返回:
            dict: 回测结果
        """
        try:
            buy_trades = [t for t in self.trade_history if t['direction'] == 'buy']
            sell_trades = [t for t in self.trade_history if t['direction'] == 'sell']
            close_trades = [t for t in self.trade_history if t['direction'] == 'close']
            
            total_open_trades = len(buy_trades) + len(sell_trades)
            
            # 计算总的盈利交易数
            profitable_trades = sum(1 for trade in close_trades if trade.get('profit', 0) > 0)
            
            # 分别统计做多和做空的盈利次数
            buy_profitable = 0
            sell_profitable = 0
            
            # 统计正常平仓的盈利交易
            for trade in close_trades:
                if trade.get('profit', 0) > 0:
                    # 查找对应的开仓交易
                    position_direction = trade.get('position_direction', 0)
                    if position_direction > 0:
                        buy_profitable += 1
                    elif position_direction < 0:
                        sell_profitable += 1
            
            max_balance = max([t['balance'] for t in self.trade_history]) if self.trade_history else self.initial_balance
            min_balance = min([t['balance'] for t in self.trade_history]) if self.trade_history else self.initial_balance
            
            total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
            
            # 获取前10条交易记录用于显示，并关联信号历史
            trade_details = []
            open_position = None
            for trade in self.trade_history:
                if trade['direction'] in ['buy', 'sell']:  # 开仓
                    open_position = trade
                elif trade['direction'] == 'close' and open_position is not None:  # 平仓
                    # 获取持仓期间的信号详情
                    position_signals = []
                    position_key = str(open_position['timestamp'])
                    if position_key in self.position_signal_details:
                        position_signals = self.position_signal_details[position_key]
                    
                    detail = {
                        'entry_time': open_position['timestamp'],
                        'entry_price': open_position['price'],
                        'direction': open_position['direction'],
                        'exit_time': trade['timestamp'],
                        'exit_price': trade['price'],
                        'profit': trade['profit'],
                        'exit_type': trade.get('exit_type', 'unknown'),
                        'signals_during_position': position_signals,
                        'entry_signal': open_position.get('entry_signal', 0)  # 获取开仓信号
                    }
                    trade_details.append(detail)
                    open_position = None
            
            # 按月统计交易结果
            monthly_stats = self._calculate_monthly_stats(close_trades)
            
            # 按日统计交易结果，包括每日最大亏损值
            daily_stats = self._calculate_daily_stats(close_trades)
            
            results = {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return_pct': total_return,
                'total_trades': total_open_trades,
                'profitable_trades': profitable_trades,
                'win_rate': profitable_trades / max(total_open_trades, 1) * 100,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'buy_win_rate': buy_profitable / max(len(buy_trades), 1) * 100,
                'sell_win_rate': sell_profitable / max(len(sell_trades), 1) * 100,
                'max_balance': max_balance,
                'min_balance': min_balance,
                'trade_history': self.trade_history,
                'trade_details': trade_details[:10],  # 只保留前10条详细记录
                'monthly_stats': monthly_stats,  # 添加月度统计
                'daily_stats': daily_stats  # 添加每日统计，包括每日最大亏损值
            }
            
            return results
        except Exception as e:
            logger.error(f"计算回测结果异常: {str(e)}")
            return None
    
    def _calculate_monthly_stats(self, close_trades):
        """
        计算每月统计数据
        
        参数:
            close_trades: 平仓交易记录列表
            
        返回:
            dict: 按月份分组的统计数据
        """
        try:
            # 按月份分组交易记录
            monthly_data = {}
            
            # 先按时间排序
            sorted_trades = sorted(close_trades, key=lambda x: x['timestamp'])
            
            # 统计每个月的数据
            for trade in sorted_trades:
                # 获取平仓时间的年月
                exit_time = trade['timestamp']
                month_key = exit_time.strftime('%Y-%m')
                
                # 更新统计数据
                if month_key not in monthly_data:
                    monthly_data[month_key] = {
                        'trade_count': 0,
                        'total_profit': 0,
                        'winning_trades': 0,
                        'ending_balance': 0
                    }
                
                monthly_data[month_key]['trade_count'] += 1
                profit = trade.get('profit', 0)
                monthly_data[month_key]['total_profit'] += profit
                if profit > 0:
                    monthly_data[month_key]['winning_trades'] += 1
                monthly_data[month_key]['ending_balance'] = trade['balance']
            
            # 补充连续月份数据（如果有交易记录的话）
            if sorted_trades:
                # 获取第一个和最后一个交易的月份
                first_month = sorted_trades[0]['timestamp'].replace(day=1)
                last_month = sorted_trades[-1]['timestamp'].replace(day=1)
                
                # 生成从第一个月到最后一个月的所有月份
                current = first_month
                while current <= last_month:
                    month_key = current.strftime('%Y-%m')
                    if month_key not in monthly_data:
                        # 获取前一个月的余额
                        prev_balance = 0
                        # 查找前一个月的数据
                        temp_current = current
                        while temp_current > first_month:
                            if temp_current.month == 1:
                                temp_current = temp_current.replace(year=temp_current.year-1, month=12)
                            else:
                                temp_current = temp_current.replace(month=temp_current.month-1)
                            
                            prev_month_key = temp_current.strftime('%Y-%m')
                            if prev_month_key in monthly_data:
                                prev_balance = monthly_data[prev_month_key]['ending_balance']
                                break
                        
                        monthly_data[month_key] = {
                            'trade_count': 0,
                            'total_profit': 0,
                            'winning_trades': 0,
                            'ending_balance': prev_balance if prev_balance > 0 else 100000  # 默认初始资金
                        }
                    
                    # 移动到下一个月
                    if current.month == 12:
                        current = current.replace(year=current.year+1, month=1)
                    else:
                        current = current.replace(month=current.month+1)
            
            return monthly_data
        except Exception as e:
            logger.error(f"计算月度统计异常: {str(e)}")
            return {}


def get_mt5_data(symbol="XAUUSD", timeframe="TIMEFRAME_M15", days=365):
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
            raise Exception("MT5初始化失败")
        
        # 计算日期范围
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # 获取数据
        rates = mt5.copy_rates_range(symbol, eval(f"mt5.{timeframe}"), from_date, to_date)
        
        if rates is None or len(rates) == 0:
            raise Exception("获取MT5数据失败")
        
        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 关闭MT5连接
        mt5.shutdown()
        
        logger.info(f"成功获取到 {len(df)} 条MT5历史数据")
        return df
        
    except Exception as e:
        logger.error(f"获取MT5数据异常: {str(e)}")
        raise Exception(f"无法获取MT5数据: {str(e)}")


def get_mt5_data_by_range(symbol="XAUUSD", timeframe="TIMEFRAME_M15", start_date=None, end_date=None):
    """
    从MT5获取指定时间范围的历史数据
    
    参数:
        symbol (str): 交易品种
        timeframe (str): 时间周期
        start_date (datetime): 开始日期
        end_date (datetime): 结束日期
    
    返回:
        DataFrame: 历史数据
    """
    try:
        import MetaTrader5 as mt5
        
        # 初始化MT5连接
        if not mt5.initialize():
            raise Exception("MT5初始化失败")
        
        # 如果未指定结束日期，默认为当前时间
        if end_date is None:
            end_date = datetime.now()
        
        # 如果未指定开始日期，默认为结束日期往前推365天
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # 获取数据
        rates = mt5.copy_rates_range(symbol, eval(f"mt5.{timeframe}"), start_date, end_date)
        
        if rates is None or len(rates) == 0:
            raise Exception("获取MT5数据失败")
        
        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # 关闭MT5连接
        mt5.shutdown()
        
        logger.info(f"成功获取到 {len(df)} 条MT5历史数据，时间范围: {start_date} 到 {end_date}")
        return df
        
    except Exception as e:
        logger.error(f"获取MT5数据异常: {str(e)}")
        raise Exception(f"无法获取MT5数据: {str(e)}")


def main(start_date=None, end_date=None):
    """
    主函数 - 回测入口
    
    参数:
        start_date (str): 回测开始日期，格式 'YYYY-MM-DD'
        end_date (str): 回测结束日期，格式 'YYYY-MM-DD'
    """
    logger.info("开始AI交易模型回测...")
    
    try:
        # 使用全局配置变量（如果未通过参数传递）
        if start_date is None and START_DATE is not None:
            start_date = START_DATE
            
        if end_date is None and END_DATE is not None:
            end_date = END_DATE
            
        # 解析日期参数
        parsed_start_date = None
        parsed_end_date = None
        
        if start_date:
            parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d')
            logger.info(f"设置回测开始日期: {parsed_start_date}")
        
        if end_date:
            parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
            logger.info(f"设置回测结束日期: {parsed_end_date}")
        
        # 1. 初始化组件
        feature_engineer = FeatureEngineer()
        # 直接加载已训练好的模型，而不是重新训练
        model = EvoAIModel("xauusd_trained_model.pkl")
        
        # 2. 获取数据（优先从MT5获取真实数据）
        logger.info("获取历史数据...")
        try:
            if parsed_start_date or parsed_end_date:
                df = get_mt5_data_by_range("XAUUSD", "TIMEFRAME_M15", parsed_start_date, parsed_end_date)
            else:
                df = get_mt5_data()
            logger.info(f"获取到 {len(df)} 条历史数据")
        except Exception as e:
            logger.error(f"无法获取MT5数据: {str(e)}")
            logger.error("由于无法获取真实市场数据且禁用模拟数据，程序将退出。")
            return
        
        # 3. 回测测试（使用已有模型进行预测）
        logger.info("开始回测测试...")
        backtester = Backtester()
        backtest_results = backtester.run_backtest(df, model, feature_engineer)
        
        if backtest_results:
            logger.info("=== 回测结果 ===")
            logger.info(f"初始资金: ${backtest_results['initial_balance']:,.2f}")
            logger.info(f"最终资金: ${backtest_results['final_balance']:,.2f}")
            logger.info(f"总收益率: {backtest_results['total_return_pct']:.2f}%")
            logger.info(f"总交易次数: {backtest_results['total_trades']}")
            logger.info(f"胜率: {backtest_results['win_rate']:.2f}%")
            
            # 打印前10条交易记录
            logger.info("=== 前10条交易记录 ===")
            for i, trade in enumerate(backtest_results['trade_details'][:10]):
                # 构建开仓信号字符串
                entry_signal_str = "未知"
                entry_signal = trade.get('entry_signal', 0)
                if entry_signal == 1:
                    entry_signal_str = "涨"
                elif entry_signal == -1:
                    entry_signal_str = "跌"
                elif entry_signal == 0:
                    entry_signal_str = "持"
                
                # 构建持仓期间信号序列字符串
                signal_sequence = ""
                
                if 'signals_during_position' in trade and trade['signals_during_position']:
                    signals = []
                    for signal_detail in trade['signals_during_position']:
                        up_prob = signal_detail['up_probability']
                        signal_value = signal_detail['signal']
                        if signal_value == 1:
                            signals.append(f"涨({up_prob:.2f})")
                        elif signal_value == -1:
                            signals.append(f"跌({up_prob:.2f})")
                        else:
                            signals.append(f"持({up_prob:.2f})")
                    signal_sequence = " ".join(signals)
                
                if not signal_sequence:
                    signal_sequence = "无信号"
                
                logger.info(f"{i+1}. 买入时间: {trade['entry_time']}, 价格: {trade['entry_price']:.5f}, 方向: {trade['direction']} | "
                           f"卖出时间: {trade['exit_time']}, 价格: {trade['exit_price']:.5f} | 收益: ${trade['profit']:.2f} | "
                           f"开仓信号: {entry_signal_str} | 持仓期间信号: {signal_sequence}")
            
            # 打印月度统计
            logger.info("=== 月度统计 ===")
            monthly_stats = backtest_results['monthly_stats']
            for month, stats in sorted(monthly_stats.items()):
                win_rate = stats['winning_trades'] / max(stats['trade_count'], 1) * 100
                logger.info(f"{month}: 交易次数={stats['trade_count']}, 盈利次数={stats['winning_trades']}, "
                           f"胜率={win_rate:.2f}%, 总收益=${stats['total_profit']:.2f}, 月末余额=${stats['ending_balance']:.2f}")
            
            # 打印每日统计，包括每日最大亏损值
            logger.info("=== 每日统计（包含每日最大亏损值） ===")
            daily_stats = backtest_results['daily_stats']
            for day, stats in sorted(daily_stats.items()):
                win_rate = stats['winning_trades'] / max(stats['trade_count'], 1) * 100
                logger.info(f"{day}: 交易次数={stats['trade_count']}, 盈利次数={stats['winning_trades']}, "
                           f"胜率={win_rate:.2f}%, 总收益=${stats['total_profit']:.2f}, 月末余额=${stats['ending_balance']:.2f}, "
                           f"当日最大亏损值=${stats['max_drawdown']:.2f}")
            
        logger.info("AI交易模型回测完成!")
        
    except FileNotFoundError:
        logger.error("未找到已训练的模型文件，请先运行训练程序")
        return
    except ValueError as ve:
        logger.error(f"日期格式错误: {str(ve)}，请使用 YYYY-MM-DD 格式")
        return
    except Exception as e:
        logger.error(f"回测过程出现异常: {str(e)}")
        return


if __name__ == "__main__":
    # 直接使用配置的日期变量
    main(START_DATE, END_DATE)

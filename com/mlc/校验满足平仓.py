import MetaTrader5 as mt5
import logging
from datetime import datetime, date, timedelta
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 全局变量存储程序启动时的今日初始余额
startup_today_initial_balance = None
startup_datetime = datetime.now()

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balance_monitor.log', encoding='utf-8'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 市场时段分析器类
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

            # 添加信号一致性特征
            df_with_all_features = self._add_signal_consistency_features(df_with_all_features)

            return df_with_all_features

        except Exception as e:
            logger.error(f"生成特征异常: {str(e)}")
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


class EvoAIModel:
    """
    具备自主进化的AI模型类
    """

    def __init__(self, model_path=None):
        """
        初始化AI模型

        参数:
            model_path (str): 模型保存路径，如果提供则加载现有模型
        """
        self.model = None
        self.performance_history = []
        self.generation = 0

        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_model()

    def _initialize_model(self):
        """
        初始化模型
        """
        # 使用随机森林作为基础模型，调整参数以提高性能
        self.model = RandomForestClassifier(
            n_estimators=200,  # 增加树的数量
            max_depth=15,  # 增加树的深度
            min_samples_split=10,  # 增加分裂所需的最小样本数
            min_samples_leaf=5,  # 增加叶节点的最小样本数
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
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency'
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
                'bb_position', 'trend_strength',
                # 新增的信号一致性特征
                'sma_5_direction', 'sma_10_direction', 'sma_20_direction',
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency'
            ]
            
            # 检查是否存在训练时没有的特征
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                logger.error(f"特征名称不匹配，缺少以下特征: {missing_features}")
                return None

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

    def save_model(self, model_path):
        """
        保存模型

        参数:
            model_path (str): 模型保存路径
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'performance_history': self.performance_history,
                    'generation': self.generation
                }, f)
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.error(f"模型保存异常: {str(e)}")

    def load_model(self, model_path):
        """
        加载模型

        参数:
            model_path (str): 模型保存路径
        """
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.performance_history = data['performance_history']
                self.generation = data['generation']
            logger.info(f"模型已从 {model_path} 加载")
        except Exception as e:
            logger.error(f"模型加载异常: {str(e)}")
            self._initialize_model()


class SignalMonitor:
    """
    信号监控类，用于获取和分析来自不同交易策略的预测信号
    """
    
    def __init__(self):
        """
        初始化信号监控器
        """
        self.xauusd_model = None
        self.real_time_model = None
        self.feature_engineer = FeatureEngineer()
        
        # 尝试加载模型
        try:
            # 加载XAUUSD模型 - 使用与xauusd_trader_m15.py相同的模型路径
            self.xauusd_model = EvoAIModel("D:/newProject/Trader/com/mlc/xauusd/xauusd_trained_model.pkl")
            logger.info("XAUUSD模型加载成功")
        except Exception as e:
            logger.error(f"XAUUSD模型加载失败: {str(e)}")
            # 初始化一个新模型
            self.xauusd_model = EvoAIModel()
            
        try:
            # 加载实时交易模型 - 使用与real_time_trader_m15.py相同的模型路径
            self.real_time_model = EvoAIModel("D:/newProject/Trader/com/mlc/test/trained_model.pkl")
            logger.info("实时交易模型加载成功")
        except Exception as e:
            logger.error(f"实时交易模型加载失败: {str(e)}")
            # 初始化一个新模型
            self.real_time_model = EvoAIModel()
    
    def get_latest_data(self, symbol, timeframe, count=50):
        """
        获取最新数据

        参数:
            symbol (str): 交易品种
            timeframe: 时间周期
            count (int): 获取K线数量

        返回:
            DataFrame: 最新数据
        """
        try:
            # 初始化MT5连接
            if not mt5.initialize():
                logger.error("MT5初始化失败")
                return None

            # 从MT5获取实时数据，获取额外一根K线以确保我们有足够数据
            rates = mt5.copy_rates_from_pos(symbol, eval(f"mt5.{timeframe}"), 0, count + 1)

            if rates is None or len(rates) == 0:
                logger.warning("获取MT5数据失败或数据为空")
                return None

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

            return df

        except Exception as e:
            logger.error(f"获取最新数据异常: {str(e)}")
            return None

    def make_decision(self, df, model, model_name="未知模型"):
        """
        做出交易决策

        参数:
            df (DataFrame): 数据
            model: AI模型实例
            model_name: 模型名称，用于日志输出

        返回:
            int: 交易信号（1做多，-1做空，0观望）
        """
        try:
            # 特征工程
            df_with_features = self.feature_engineer.generate_features(df)

            # 准备数据用于预测
            X = model.prepare_prediction_data(df_with_features.tail(50))  # 使用最近50条数据

            if X is None or len(X) == 0:
                logger.info(f"{model_name}特征数据为空，返回观望信号")
                return 0

            # 确保我们使用的是最后一行数据进行预测
            if len(X) > 0:
                # 预测
                prediction = model.predict(X.iloc[[-1]])  # 只使用最新的数据点

                if prediction is None:
                    logger.info(f"{model_name}预测结果为空，返回观望信号")
                    return 0

                # 根据预测概率确定信号
                # 获取信号（概率大于0.55做多，小于0.45做空，否则持有）
                up_prob = prediction[0][1]

                # 记录预测概率到日志文件
                log_message = f"{model_name}预测概率 - 上涨: {up_prob:.4f}, 下跌: {1 - up_prob:.4f}"
                logger.info(log_message)

                # 当上涨概率大于0.55时做多，小于0.45时做空，否则观望
                if up_prob > 0.55:
                    decision_message = f"{model_name}决策: 做多"
                    logger.info(decision_message)
                    return 1   # 做多
                elif up_prob < 0.45:
                    decision_message = f"{model_name}决策: 做空"
                    logger.info(decision_message)
                    return -1  # 做空
                else:
                    decision_message = f"{model_name}决策: 观望 (概率区间0.45-0.55，当前概率: {up_prob:.4f})"
                    logger.info(decision_message)
                    return 0   # 观望
            else:
                logger.info(f"{model_name}没有足够的有效数据进行预测，返回观望信号")
                return 0

        except Exception as e:
            logger.error(f"{model_name}做出交易决策异常: {str(e)}")
            return 0

    def get_signals(self):
        """
        获取两个策略的预测信号

        返回:
            dict: 包含两个策略信号的字典
        """
        try:
            # 获取最新市场数据
            df = self.get_latest_data("XAUUSD", "TIMEFRAME_M15", 100)

            if df is None or len(df) < 50:
                logger.warning("数据不足，无法获取预测信号")
                return {
                    'xauusd_signal': 0,
                    'real_time_signal': 0,
                    'xauusd_prob': 0.5,
                    'real_time_prob': 0.5
                }

            logger.info(f"获取到 {len(df)} 根M15 K线数据用于信号分析")

            # 获取XAUUSD策略信号
            xauusd_signal = self.make_decision(df, self.xauusd_model, "XAUUSD策略")
            
            # 获取实时交易策略信号
            real_time_signal = self.make_decision(df, self.real_time_model, "实时交易策略")

            return {
                'xauusd_signal': xauusd_signal,
                'real_time_signal': real_time_signal,
                'xauusd_prob': getattr(self, '_get_last_prob', 0.5),
                'real_time_prob': getattr(self, '_get_last_prob', 0.5)
            }

        except Exception as e:
            logger.error(f"获取预测信号异常: {str(e)}")
            return {
                'xauusd_signal': 0,
                'real_time_signal': 0,
                'xauusd_prob': 0.5,
                'real_time_prob': 0.5
            }


def get_today_initial_balance():
    """
    获取今日初始余额（即今日未交易时的余额）
    这是当天交易开始前的余额
    """
    global startup_today_initial_balance
    
    try:
        # 如果程序启动时的初始余额还没有设置，设置它
        if startup_today_initial_balance is None:
            # 初始化MT5连接
            if not mt5.initialize():
                logger.error("MT5初始化失败")
                return None
            
            current_balance = mt5.account_info().balance
            startup_today_initial_balance = current_balance
            mt5.shutdown()
            
            logger.info(f"程序启动时今日初始余额已设置为: {current_balance:.2f}USD")
            
        # 获取今日日期
        today = date.today()
        
        # 获取今天的时间范围
        today_start = datetime(today.year, today.month, today.day)
        today_end = datetime.now()
        
        # 获取今日历史交易记录
        today_deals = mt5.history_deals_get(today_start, today_end)
        if today_deals is None or len(today_deals) == 0:
            # 如果今天没有任何交易记录，返回程序启动时记录的初始余额
            return startup_today_initial_balance
        
        # 将交易记录按时间排序
        sorted_deals = sorted(today_deals, key=lambda x: x.time)
        
        # 检查今天是否在程序启动后发生了交易
        # 找到今天第一个交易的时间
        first_deal_time = None
        for deal in sorted_deals:
            deal_time = datetime.fromtimestamp(deal.time)
            if deal_time.date() == today:
                first_deal_time = deal_time
                break
        
        # 如果今天第一个交易发生在程序启动之后，那么程序启动时的余额就是今日初始余额
        # 否则，我们需要计算今日初始余额
        if first_deal_time and first_deal_time > startup_datetime:
            # 今天在程序启动后有交易，程序启动时的余额就是今日初始余额
            return startup_today_initial_balance
        else:
            # 今天在程序启动前就有交易，或没有交易，按原方法计算
            # 计算今天所有交易的盈亏
            today_trading_profit = 0
            for deal in sorted_deals:
                deal_time = datetime.fromtimestamp(deal.time)
                if deal_time.date() == today:
                    # 排除余额调整类型的交易
                    if not (hasattr(mt5, 'DEAL_TYPE_BALANCE') and deal.type == mt5.DEAL_TYPE_BALANCE):
                        if not (hasattr(mt5, 'DEAL_TYPE_DEPOSIT') and deal.type == mt5.DEAL_TYPE_DEPOSIT):
                            if not (hasattr(mt5, 'DEAL_TYPE_WITHDRAW') and deal.type == mt5.DEAL_TYPE_WITHDRAW):
                                today_trading_profit += deal.profit
            
            # 当前余额
            if not mt5.initialize():
                logger.error("MT5初始化失败")
                return None
            current_balance = mt5.account_info().balance
            mt5.shutdown()
            
            # 计算今日初始余额
            today_initial_balance = current_balance - today_trading_profit
            
            return today_initial_balance
        
    except Exception as e:
        logger.error(f"获取今日初始余额时出错: {str(e)}")
        return None

def check_balance_and_close_positions():
    """
    检查账户余额并根据条件平仓
    条件1: 如果当前余额 > 11010，则平掉所有仓位
    条件2: 如果当前余额 < 初始余额 并且 亏损 >= 450则平掉所有仓位
    """
    
    # 初始化MT5连接
    if not mt5.initialize():
        logger.error("MT5初始化失败")
        return False
    
    try:
        # 获取账户信息
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            return False
        
        # 获取当前余额和权益
        current_balance = account_info.balance
        equity = account_info.equity
        currency = account_info.currency
        
        logger.info(f"账户查询 - 余额: {current_balance:.2f}{currency}, 权益: {equity:.2f}{currency}")
        
        # 获取今日初始余额（即今天没交易时候的余额）
        today_initial_balance = get_today_initial_balance()
        if today_initial_balance is None:
            logger.error("无法获取今日初始余额")
            return False
        
        logger.info(f"今日初始余额（今日未交易时的余额）: {today_initial_balance:.2f}{currency}")
        
        # 获取所有持仓
        positions = mt5.positions_total()
        logger.info(f"当前持仓数量: {positions if positions is not None else 0}")
        
        if positions and positions > 0:
            # 获取持仓详情
            open_positions = mt5.positions_get()
            if open_positions:
                total_profit = sum(pos.profit for pos in open_positions)
                logger.info(f"持仓总盈亏: {total_profit:.2f}{currency}")
        else:
            logger.info("持仓总盈亏: 0.00USD")  # 没有持仓时显示0盈亏
        
        # 使用今日初始余额作为基准
        base_balance = today_initial_balance
        
        # 检查条件1: 如果当前余额 > 11010，则平掉所有仓位
        if current_balance > 11010.0:
            logger.info(f"当前余额 {current_balance:.2f} 超过阈值 11010.0，执行平仓操作")
            close_all_positions()
            return True
        
        # 检查条件2: 如果当前余额 < 基准余额 并且 差值 >= 450，则平掉所有仓位
        balance_diff = base_balance - current_balance
        if current_balance < base_balance and abs(balance_diff) >= 450.0:
            logger.info(f"当前余额 {current_balance:.2f} 低于基准余额 {base_balance:.2f}，且差值 {abs(balance_diff):.2f} >= 450，执行平仓操作")
            close_all_positions()
            return True
        
        # 获取预测信号用于风控分析
        signal_monitor = SignalMonitor()
        signals = signal_monitor.get_signals()
        
        logger.info(f"XAUUSD策略信号: {signals['xauusd_signal']}, 实时交易策略信号: {signals['real_time_signal']}")
        
        logger.info("余额检查完成，未达到平仓条件")
        return False
        
    except Exception as e:
        logger.error(f"检查余额和执行平仓操作时发生异常: {str(e)}")
        return False
    finally:
        # 关闭MT5连接
        mt5.shutdown()

def close_all_positions():
    """
    平掉所有持仓
    """
    try:
        # 初始化MT5连接
        if not mt5.initialize():
            logger.error("MT5初始化失败")
            return False
        
        # 获取所有持仓
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.info("没有持仓需要平仓")
            return True
        
        logger.info(f"发现 {len(positions)} 个持仓，开始平仓...")
        
        closed_count = 0
        for position in positions:
            # 创建平仓请求
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "position": position.ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": position.magic,
                "comment": "风控自动平仓",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # 发送平仓请求
            result = mt5.order_send(close_request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"平仓成功，订单号: {result.order}, 交易品种: {position.symbol}, 手数: {position.volume}")
                closed_count += 1
            else:
                logger.error(f"平仓失败，错误码: {result.retcode}, 交易品种: {position.symbol}")
        
        logger.info(f"平仓操作完成，共平掉 {closed_count} 个持仓")
        return True
        
    except Exception as e:
        logger.error(f"平仓操作异常: {str(e)}")
        return False
    finally:
        # 关闭MT5连接
        mt5.shutdown()

def check_balance_with_custom_thresholds(today_balance, remaining_balance, 
                                       profit_threshold=11010.0, 
                                       loss_threshold=450.0):
    """
    使用自定义阈值检查余额并决定是否平仓
    
    参数:
        today_balance (float): 今日余额
        remaining_balance (float): 剩余余额
        profit_threshold (float): 盈利平仓阈值，默认11010
        loss_threshold (float): 亏损平仓阈值，默认450
    
    返回:
        bool: 是否执行了平仓操作
    """
    logger.info(f"今日余额: {today_balance:.2f}, 剩余余额: {remaining_balance:.2f}")
    
    # 条件1: 如果今日余额 > 盈利阈值，则平掉所有仓位
    if today_balance > profit_threshold:
        logger.info(f"今日余额 {today_balance:.2f} 超过盈利阈值 {profit_threshold:.2f}，需要平仓")
        return True
    
    # 条件2: 如果今日余额 < 剩余余额 且 差值 >= 亏损阈值，则平掉所有仓位
    balance_diff = remaining_balance - today_balance
    if today_balance < remaining_balance and balance_diff >= loss_threshold:
        logger.info(f"今日余额 {today_balance:.2f} 低于剩余余额 {remaining_balance:.2f}，且亏损 {balance_diff:.2f} >= {loss_threshold:.2f}，需要平仓")
        return True
    
    logger.info("余额检查完成，未达到平仓条件")
    return False

def monitor_account_and_manage_risk():
    """
    监控账户并进行风险管理
    """
    logger.info("开始账户监控和风险管理...")
    
    # 获取当前账户信息
    if not mt5.initialize():
        logger.error("MT5初始化失败")
        return
    
    try:
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("无法获取账户信息")
            return
        
        current_balance = account_info.balance
        equity = account_info.equity
        currency = account_info.currency
        
        # 获取今日初始余额（即今天没交易时候的余额）
        today_initial_balance = get_today_initial_balance()
        if today_initial_balance is None:
            logger.error("无法获取今日初始余额")
            return
        
        today_balance = current_balance
        remaining_balance = today_initial_balance  # 使用今日初始余额作为基准
        
        logger.info(f"账户状态 - 余额: {today_balance:.2f}{currency}, 权益: {equity:.2f}{currency}, 今日初始余额: {remaining_balance:.2f}{currency}")
        
        # 获取预测信号用于风控分析
        signal_monitor = SignalMonitor()
        signals = signal_monitor.get_signals()
        
        logger.info(f"XAUUSD策略信号: {signals['xauusd_signal']}, 实时交易策略信号: {signals['real_time_signal']}")
        
        # 检查是否需要平仓
        should_close = check_balance_with_custom_thresholds(
            today_balance, 
            remaining_balance
        )
        
        if should_close:
            logger.info("触发风控条件，开始平仓...")
            close_all_positions()
        else:
            logger.info("账户状态正常，无需平仓")
            
    finally:
        mt5.shutdown()

def continuous_monitoring():
    """
    持续监控账户，每分钟检查一次
    以程序启动时的余额作为基准进行风控判断
    """
    logger.info("启动持续监控模式，每分钟检查一次账户余额...")
    
    try:
        while True:
            logger.info("执行定期账户余额检查...")
            check_balance_and_close_positions()
            
            # 等待60秒后再次检查
            logger.info("等待60秒后进行下次检查...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("监控已停止（用户中断）")
    except Exception as e:
        logger.error(f"监控过程中发生异常: {str(e)}")

if __name__ == "__main__":
    # 运行一次检查
    # check_balance_and_close_positions()
    
    # 或者使用持续监控模式
    continuous_monitoring()
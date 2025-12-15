import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            window = 5
            local_highs = (df['close'] == df['close'].rolling(window=window*2+1, center=True).max())
            local_lows = (df['close'] == df['close'].rolling(window=window*2+1, center=True).min())
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
                'bb_position', 'trend_strength'
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
                'bb_position', 'trend_strength'
            ]
            
            X = df[feature_columns]
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


class RealTimeTrader:
    """
    实时交易类
    """
    
    def __init__(self, model_path="trained_model.pkl"):
        """
        初始化实时交易器
        
        参数:
            model_path (str): 模型路径
        """
        self.feature_engineer = FeatureEngineer()
        self.model = EvoAIModel(model_path)
        self.is_running = False
        self.current_position = None  # 当前持仓信息
        
        # 初始化MT5连接
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.error("MT5初始化失败")
                raise Exception("MT5初始化失败")
            self.mt5 = mt5
            logger.info("MT5连接成功")
        except Exception as e:
            logger.error(f"MT5连接异常: {str(e)}")
            self.mt5 = None
    
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
            if self.mt5 is None:
                logger.error("MT5未初始化")
                return None
                
            # 从MT5获取实时数据
            rates = self.mt5.copy_rates_from_pos(symbol, eval(f"self.mt5.{timeframe}"), 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning("获取MT5数据失败或数据为空")
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return df
            
        except Exception as e:
            logger.error(f"获取最新数据异常: {str(e)}")
            return None
    
    def make_decision(self, df):
        """
        做出交易决策
        
        参数:
            df (DataFrame): 数据
            
        返回:
            int: 交易信号（1做多，-1做空，0观望）
        """
        try:
            # 特征工程
            df_with_features = self.feature_engineer.generate_features(df)
            
            # 准备数据用于预测
            X = self.model.prepare_prediction_data(df_with_features.tail(50))  # 使用最近50条数据
            
            if X is None or len(X) == 0:
                return 0
            
            # 预测
            prediction = self.model.predict(X.tail(1))  # 只使用最新的数据点
            
            if prediction is None:
                return 0
            
            # 根据预测概率确定信号
            # 获取信号（概率大于0.55做多，小于0.45做空，否则持有）
            up_prob = prediction[0][1]
            
            if up_prob > 0.55:
                return 1  # 做多
            elif up_prob < 0.45:
                return -1  # 做空
            else:
                return 0  # 观望
                
        except Exception as e:
            logger.error(f"做出交易决策异常: {str(e)}")
            return 0
    
    def check_and_close_position(self, symbol, current_price):
        """
        检查并强制平仓（如果盈亏超过2000美元）
        
        参数:
            symbol (str): 交易品种
            current_price (float): 当前价格
            
        返回:
            bool: 是否进行了强制平仓操作
        """
        try:
            if self.current_position is not None:
                entry_price = self.current_position['entry_price']
                direction = self.current_position['direction']
                lots = self.current_position['lots']
                
                # 计算当前盈亏 (XAUUSD每点价值100美元)
                profit = (current_price - entry_price) * direction * 100 * lots
                
                # 如果盈利或亏损超过2000美元，则强制平仓
                if abs(profit) >= 2000:
                    logger.info(f"强制平仓，当前盈亏: ${profit:.2f}，超过2000美元限制")
                    
                    # 在实际应用中，这里会调用MT5的平仓函数
                    # self.close_position_mt5(symbol)
                    
                    self.current_position = None
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"检查强制平仓异常: {str(e)}")
            return False
    
    def execute_trade(self, symbol, signal, lot_size=1.0, current_price=None):
        """
        执行交易
        
        参数:
            symbol (str): 交易品种
            signal (int): 信号
            lot_size (float): 手数
            current_price (float): 当前价格，用于检查强制平仓
        """
        try:
            # 如果提供了当前价格，检查是否需要强制平仓
            if current_price is not None:
                forced_closed = self.check_and_close_position(symbol, current_price)
                if forced_closed:
                    logger.info("已强制平仓")
                    return
            
            # 检查是否有相反信号需要平仓
            if self.current_position is not None and self.current_position['direction'] != signal and signal != 0:
                logger.info(f"平仓 {symbol}，反向信号出现")
                # 在实际应用中，这里会调用MT5的平仓函数
                # self.close_position_mt5(symbol)
                self.current_position = None
            
            # 如果没有持仓且信号非0，则开仓
            if self.current_position is None and signal != 0:
                logger.info(f"开仓 {symbol}，方向: {'做多' if signal > 0 else '做空'}，手数: {lot_size}")
                # 在实际应用中，这里会调用MT5的开仓函数
                # self.open_position_mt5(symbol, signal, lot_size)
                # 使用从MT5获取的最新数据时间作为入场时间
                df = self.get_latest_data(symbol, "TIMEFRAME_M15", 1)
                entry_time = df['time'].iloc[-1] if df is not None and len(df) > 0 else datetime.now()
                
                self.current_position = {
                    'entry_time': entry_time,
                    'entry_price': current_price if current_price is not None else 0,
                    'direction': int(signal),
                    'lots': lot_size
                }
            elif signal == 0:
                logger.info("观望，无交易信号")
                
        except Exception as e:
            logger.error(f"执行交易异常: {str(e)}")
    
    def run(self, symbol="XAUUSD", lot_size=1.0):
        """
        运行实时交易
        
        参数:
            symbol (str): 交易品种
            lot_size (float): 手数
        """
        try:
            logger.info(f"开始实时交易 {symbol}，手数: {lot_size}")
            self.is_running = True
            
            while self.is_running:
                try:
                    # 获取最新数据
                    df = self.get_latest_data(symbol, "TIMEFRAME_M15", 100)
                    
                    if df is None or len(df) < 50:
                        logger.warning("数据不足，等待下次更新")
                        time.sleep(60)  # 等待1分钟
                        continue
                    
                    # 获取当前价格
                    current_price = df['close'].iloc[-1]
                    
                    # 做出交易决策
                    signal = self.make_decision(df)
                    
                    # 执行交易
                    self.execute_trade(symbol, signal, lot_size, current_price)
                    
                    # 等待1分钟
                    logger.info("等待1分钟后继续执行")
                    time.sleep(60)
                    
                except KeyboardInterrupt:
                    logger.info("收到停止信号，正在退出...")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"交易循环异常: {str(e)}")
                    time.sleep(60)  # 出错后等待1分钟再试
                    
        except Exception as e:
            logger.error(f"运行实时交易异常: {str(e)}")
    
    def _wait_for_next_m15_period(self, symbol="XAUUSD"):
        """
        等待到下一个M15周期的开始时间
        
        参数:
            symbol (str): 交易品种
        """
        try:
            # 获取XAUUSD的当前市场时间
            df = self.get_latest_data(symbol, "TIMEFRAME_M15", 1)
            if df is not None and len(df) > 0:
                # 使用最新的K线时间作为当前时间
                now = df['time'].iloc[-1].to_pydatetime()
            else:
                # 如果无法获取市场时间，则使用系统时间并记录警告
                now = datetime.now()
                logger.warning("无法获取市场时间，使用系统时间")
            
            # 计算下一个M15周期的开始时间
            # M15周期开始时间应该是00:00, 00:15, 00:30, 00:45, 01:00, ...
            minutes = now.minute
            next_minutes = ((minutes // 15) + 1) * 15
            if next_minutes >= 60:
                # 跨小时处理
                next_hour = now.hour + 1
                if next_hour >= 24:
                    # 跨天处理
                    next_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    next_time = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            else:
                next_time = now.replace(minute=next_minutes, second=0, microsecond=0)
            
            # 计算需要等待的时间
            wait_seconds = (next_time - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"等待到下一个M15周期开始时间: {next_time.strftime('%Y-%m-%d %H:%M:%S')} (等待 {wait_seconds:.0f} 秒)")
                time.sleep(wait_seconds)
            else:
                logger.info("已在M15周期开始时间点")
                
        except Exception as e:
            logger.error(f"等待M15周期异常: {str(e)}")
            # 出错时默认等待15分钟
            time.sleep(15 * 60)
    
    def shutdown(self):
        """
        关闭交易器
        """
        try:
            self.is_running = False
            # 关闭MT5连接
            if self.mt5 is not None:
                self.mt5.shutdown()
            logger.info("实时交易器已关闭")
        except Exception as e:
            logger.error(f"关闭交易器异常: {str(e)}")


def main():
    """
    主函数
    """
    trader = RealTimeTrader()
    
    # 运行实时交易（在实际应用中取消注释下面一行）
    trader.run("XAUUSD", 1.0)
    logger.info("实时交易系统启动")

if __name__ == "__main__":
    main()
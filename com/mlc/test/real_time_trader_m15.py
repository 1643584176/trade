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


class RealTimeTraderM15:
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
            
            # 初始化时检查现有持仓
            self._check_existing_positions("XAUUSD")
        except Exception as e:
            logger.error(f"MT5连接异常: {str(e)}")
            self.mt5 = None

    def _check_existing_positions(self, symbol):
        """
        检查MT5中现有的持仓
        
        参数:
            symbol (str): 交易品种
        """
        try:
            if self.mt5 is None:
                logger.error("MT5未初始化")
                return
                
            # 获取当前持仓
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None:
                logger.warning("无法获取持仓信息")
                return
                
            if len(positions) > 0:
                # 取第一个持仓作为当前持仓
                position = positions[0]
                self.current_position = {
                    'ticket': position.ticket,
                    'entry_time': datetime.fromtimestamp(position.time),
                    'entry_price': position.price_open,
                    'direction': 1 if position.type == self.mt5.ORDER_TYPE_BUY else -1,
                    'lots': position.volume
                }
                direction_str = "做多" if self.current_position['direction'] > 0 else "做空"
                logger.info(f"检测到现有持仓: {direction_str}, 入场价格: {self.current_position['entry_price']:.5f}, 手数: {self.current_position['lots']}")
            else:
                logger.info("未检测到现有持仓")
                
        except Exception as e:
            logger.error(f"检查现有持仓异常: {str(e)}")

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
                logger.info("特征数据为空，返回观望信号")
                return 0
            
            # 预测
            prediction = self.model.predict(X.tail(1))  # 只使用最新的数据点
            
            if prediction is None:
                logger.info("模型预测结果为空，返回观望信号")
                return 0
            
            # 根据预测概率确定信号
            # 获取信号（概率大于0.55做多，小于0.45做空，否则持有）
            up_prob = prediction[0][1]
            
            logger.info(f"预测概率 - 上涨: {up_prob:.4f}, 下跌: {1-up_prob:.4f}")
            
            if up_prob > 0.55:
                logger.info("决策: 做多")
                return 1  # 做多
            elif up_prob < 0.45:
                logger.info("决策: 做空")
                return -1  # 做空
            else:
                logger.info(f"决策: 观望 (概率区间0.45-0.55，当前概率: {up_prob:.4f})")
                return 0  # 观望
                
        except Exception as e:
            logger.error(f"做出交易决策异常: {str(e)}")
            return 0
    
    def check_and_close_position(self, symbol, current_price):
        """
        检查并强制平仓（如果当日盈亏超过2000美元）
        
        参数:
            symbol (str): 交易品种
            current_price (float): 当前价格
            
        返回:
            bool: 是否进行了强制平仓操作
        """
        # 移除此方法的功能，始终返回False
        return False
    
    def update_daily_profit_loss(self):
        """
        更新当日盈亏
        """
        # 移除此方法的功能，保持空实现
        pass

    def close_all_positions(self, symbol):
        """
        平掉指定品种的所有持仓
        
        参数:
            symbol (str): 交易品种
        """
        try:
            if self.mt5 is None:
                logger.error("MT5未初始化")
                return False
                
            # 获取当前持仓
            positions = self.mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                logger.info("没有找到持仓")
                return True
                
            # 平掉所有持仓
            for position in positions:
                # 创建平仓请求
                close_request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": self.mt5.ORDER_TYPE_SELL if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": self.mt5.symbol_info_tick(symbol).bid if position.type == self.mt5.ORDER_TYPE_BUY else self.mt5.symbol_info_tick(symbol).ask,
                    "deviation": 20,
                    "magic": 10032025,
                    "comment": "AI策略平仓",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                
                # 发送平仓请求
                result = self.mt5.order_send(close_request)
                if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                    logger.error(f"平仓失败, 错误码: {result.retcode}")
                    return False
                else:
                    logger.info(f"平仓成功, 订单号: {result.order}")
            
            # 平仓后重置持仓信息
            self.current_position = None
                    
            return True
            
        except Exception as e:
            logger.error(f"平仓异常: {str(e)}")
            return False

    def close_position_mt5(self, symbol):
        """
        平掉指定品种的当前持仓
        
        参数:
            symbol (str): 交易品种
        """
        try:
            if self.mt5 is None:
                logger.error("MT5未初始化")
                return False
                
            if self.current_position is None:
                logger.info("没有持仓需要平仓")
                return True
                
            # 创建平仓请求
            position_type = self.mt5.ORDER_TYPE_BUY if self.current_position['direction'] > 0 else self.mt5.ORDER_TYPE_SELL
            close_request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": self.current_position['lots'],
                "type": self.mt5.ORDER_TYPE_SELL if position_type == self.mt5.ORDER_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": self.current_position['ticket'],
                "price": self.mt5.symbol_info_tick(symbol).bid if position_type == self.mt5.ORDER_TYPE_BUY else self.mt5.symbol_info_tick(symbol).ask,
                "deviation": 20,
                "magic": 10032025,
                "comment": "AI策略平仓",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            # 发送平仓请求
            result = self.mt5.order_send(close_request)
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"平仓失败, 错误码: {result.retcode}")
                return False
            else:
                logger.info(f"平仓成功, 订单号: {result.order}")
                # 平仓后重置持仓信息
                self.current_position = None
                return True
                
        except Exception as e:
            logger.error(f"平仓异常: {str(e)}")
            return False

    def open_position_mt5(self, symbol, signal, lot_size):
        """
        在MT5中开仓
        
        参数:
            symbol (str): 交易品种
            signal (int): 交易信号（1做多，-1做空）
            lot_size (float): 手数
            
        返回:
            int: 订单号，如果失败返回None
        """
        try:
            if self.mt5 is None:
                logger.error("MT5未初始化")
                return None
                
            # 确定订单类型
            order_type = self.mt5.ORDER_TYPE_BUY if signal > 0 else self.mt5.ORDER_TYPE_SELL
            
            # 获取当前价格
            tick_info = self.mt5.symbol_info_tick(symbol)
            if tick_info is None:
                logger.error("无法获取品种报价信息")
                return None
                
            price = tick_info.ask if signal > 0 else tick_info.bid
            
            # 创建订单请求
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": 0.0,  # 止损
                "tp": 0.0,  # 止盈
                "deviation": 20,
                "magic": 10032025,
                "comment": "AI策略开仓",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            # 发送订单请求
            result = self.mt5.order_send(request)
            if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                logger.error(f"开仓失败, 错误码: {result.retcode}")
                return None
            else:
                logger.info(f"开仓成功, 订单号: {result.order}")
                return result.order
                
        except Exception as e:
            logger.error(f"开仓异常: {str(e)}")
            return None

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
            # 检查是否有相反信号需要平仓并开新仓
            if self.current_position is not None and self.current_position['direction'] != signal and signal != 0:
                logger.info(f"平仓 {symbol}，反向信号出现")
                # 平掉当前持仓
                self.close_all_positions(symbol)
                self.current_position = None
                
                # 反向开仓
                logger.info(f"开仓 {symbol}，方向: {'做多' if signal > 0 else '做空'}，手数: {lot_size}")
                # 执行实际下单
                ticket = self.open_position_mt5(symbol, signal, lot_size)
                if ticket is not None:
                    # 使用从MT5获取的最新数据时间作为入场时间
                    df = self.get_latest_data(symbol, "TIMEFRAME_M15", 1)
                    entry_time = df['time'].iloc[-1] if df is not None and len(df) > 0 else datetime.now()
                    
                    self.current_position = {
                        'ticket': ticket,
                        'entry_time': entry_time,
                        'entry_price': current_price if current_price is not None else 0,
                        'direction': int(signal),
                        'lots': lot_size
                    }
                else:
                    logger.error("开仓失败")
            # 如果没有持仓且信号非0，则开仓
            elif self.current_position is None and signal != 0:
                logger.info(f"开仓 {symbol}，方向: {'做多' if signal > 0 else '做空'}，手数: {lot_size}")
                # 执行实际下单
                ticket = self.open_position_mt5(symbol, signal, lot_size)
                if ticket is not None:
                    # 使用从MT5获取的最新数据时间作为入场时间
                    df = self.get_latest_data(symbol, "TIMEFRAME_M15", 1)
                    entry_time = df['time'].iloc[-1] if df is not None and len(df) > 0 else datetime.now()
                    
                    self.current_position = {
                        'ticket': ticket,
                        'entry_time': entry_time,
                        'entry_price': current_price if current_price is not None else 0,
                        'direction': int(signal),
                        'lots': lot_size
                    }
                else:
                    logger.error("开仓失败")
            elif signal == 0:
                pass  # 不再记录无交易信号的情况
                
        except Exception as e:
            logger.error(f"执行交易异常: {str(e)}")
    
    def run(self, symbol="XAUUSD", lot_size=1.0):
        """
        运行实时交易
        基于M15周期数据进行交易，当预测方向出现反向则平仓否则继续持仓
        
        参数:
            symbol (str): 交易品种
            lot_size (float): 手数
        """
        try:
            logger.info(f"开始基于M15周期的实时交易 {symbol}，手数: {lot_size}")
            logger.info("策略: 当预测方向出现反向则平仓否则继续持仓")
            self.is_running = True
            first_run = True
            
            # 如果已经有持仓，显示持仓信息
            if self.current_position is not None:
                direction_str = "做多" if self.current_position['direction'] > 0 else "做空"
                logger.info(f"启动时检测到持仓: {direction_str}, 入场价格: {self.current_position['entry_price']:.5f}")
            
            while self.is_running:
                try:
                    # 获取最新数据
                    df = self.get_latest_data(symbol, "TIMEFRAME_M15", 100)
                    
                    if df is None or len(df) < 50:
                        logger.warning("数据不足，等待下次更新")
                        time.sleep(60)  # 等待1分钟
                        continue
                    
                    logger.info(f"获取到 {len(df)} 根M15 K线数据用于分析")
                    
                    # 获取当前价格
                    current_price = df['close'].iloc[-1]
                    
                    # 只在第一次运行时打印基本信息
                    if first_run:
                        logger.info(f"当前价格: {current_price:.5f}")
                        first_run = False
                    
                    # 做出交易决策
                    signal = self.make_decision(df)
                    
                    # 执行交易
                    self.execute_trade(symbol, signal, lot_size, current_price)
                    
                    # 打印当前持仓状态
                    if self.current_position is not None:
                        logger.info(f"当前持仓方向: {'做多' if self.current_position['direction'] > 0 else '做空'}, 入场价格: {self.current_position['entry_price']:.5f}")
                    else:
                        logger.info("当前无持仓")
                    
                    # 等待到下一个M15周期
                    now = datetime.now()
                    minutes = now.minute
                    # 计算下一个15分钟周期的分钟数 (0, 15, 30, 45)
                    next_minute = ((minutes // 15) + 1) * 15
                    if next_minute == 60:
                        next_minute = 0
                    
                    # 计算需要等待的秒数
                    if next_minute > minutes:
                        wait_minutes = next_minute - minutes
                    else:
                        wait_minutes = (60 - minutes) + next_minute
                    
                    wait_seconds = wait_minutes * 60 - now.second
                    
                    logger.info(f"等待 {wait_seconds} 秒到下一个M15周期")
                    time.sleep(wait_seconds)
                    
                except KeyboardInterrupt:
                    logger.info("收到停止信号，正在退出...")
                    self.is_running = False
                    break
                except Exception as e:
                    logger.error(f"交易循环异常: {str(e)}")
                    # 出错后等待到下一个M15周期
                    now = datetime.now()
                    minutes = now.minute
                    next_minute = ((minutes // 15) + 1) * 15
                    if next_minute == 60:
                        next_minute = 0
                    
                    if next_minute > minutes:
                        wait_minutes = next_minute - minutes
                    else:
                        wait_minutes = (60 - minutes) + next_minute
                    
                    wait_seconds = wait_minutes * 60 - now.second
                    
                    logger.info(f"出错后等待 {wait_seconds} 秒到下一个M15周期")
                    time.sleep(wait_seconds)
                    
        except Exception as e:
            logger.error(f"运行实时交易异常: {str(e)}")
    
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
    trader = RealTimeTraderM15()
    
    # 运行实时交易（在实际应用中取消注释下面一行）
    trader.run("XAUUSD", 1.0)
    logger.info("基于M15周期的实时交易系统启动")

if __name__ == "__main__":
    main()
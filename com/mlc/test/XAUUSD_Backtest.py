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
            
            # 准备特征数据
            df_with_features = feature_engineer.generate_features(df)
            
            # 用于1小时交易的索引（每4个M15蜡烛交易一次）
            # 确保在当前时刻只能使用到目前为止的数据进行预测
            trade_indices = range(50, min(5000, len(df_with_features) - 5), 4)  # 限制回测数据量以加快速度
            
            for i in trade_indices:
                # 获取当前时刻的数据（仅使用到当前时刻为止的数据）
                # 确保只使用已形成的完整K线数据
                current_data = df_with_features.iloc[:i].tail(50)  # 使用最近50条完整数据（排除当前未完成的K线）
                
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
                    
                # 执行交易
                self._execute_trade(df_with_features.iloc[i], signal)
                
                # 更新权益曲线
                self._update_equity(df_with_features.iloc[i]['close'])
            
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
            # 检查现有持仓是否需要强制平仓（盈利或亏损超过2000美元）
            if len(self.positions) > 0:
                current_position = self.positions[0]
                entry_price = current_position['entry_price']
                direction = current_position['direction']
                
                # 计算当前盈亏 (XAUUSD每点价值100美元)
                profit = (data['close'] - entry_price) * direction * 100 * current_position['lots']
                
                # 如果盈利或亏损超过2000美元，则强制平仓
                if abs(profit) >= 2000:
                    self.balance += profit
                    
                    # 记录平仓交易
                    exit_type = 'take_profit' if profit > 0 else 'stop_loss'
                    self.trade_history.append({
                        'timestamp': data['time'],
                        'direction': 'close',
                        'price': data['close'],
                        'profit': profit,
                        'balance': self.balance,
                        'exit_type': exit_type
                    })
                    
                    self.positions.clear()
            
            # 平掉相反方向的持仓（如果有新的反向信号）
            if len(self.positions) > 0:
                current_position = self.positions[0]
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
                        'exit_type': 'reverse_signal'
                    })
                    
                    self.positions.clear()
            
            # 如果没有持仓且信号非0，则开仓
            if len(self.positions) == 0 and signal != 0:
                # 开仓
                self.positions.append({
                    'entry_time': data['time'],
                    'entry_price': data['close'],
                    'direction': int(signal),  # 确保是标量
                    'lots': 1.0  # 回到原来的1手交易
                })
                
                # 记录开仓交易
                self.trade_history.append({
                    'timestamp': data['time'],
                    'direction': 'buy' if signal > 0 else 'sell',
                    'price': data['close'],
                    'lots': 1.0,
                    'balance': self.balance
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
    
    def _update_equity(self, current_price):
        """
        更新权益
        
        参数:
            current_price (float): 当前价格
        """
        try:
            equity = self.balance
            # 计算未平仓盈亏 (XAUUSD每点价值100美元)
            for position in self.positions:
                unrealized_pnl = (current_price - position['entry_price']) * position['direction'] * position['lots'] * 100
                equity += unrealized_pnl
            
            self.equity_history.append({
                'timestamp': datetime.now(),
                'equity': equity,
                'balance': self.balance
            })
        except Exception as e:
            logger.error(f"更新权益异常: {str(e)}")
    
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
            
            # 获取前10条交易记录用于显示
            trade_details = []
            open_position = None
            for trade in self.trade_history:
                if trade['direction'] in ['buy', 'sell']:  # 开仓
                    open_position = trade
                elif trade['direction'] == 'close' and open_position is not None:  # 平仓
                    detail = {
                        'entry_time': open_position['timestamp'],
                        'entry_price': open_position['price'],
                        'direction': open_position['direction'],
                        'exit_time': trade['timestamp'],
                        'exit_price': trade['price'],
                        'profit': trade['profit'],
                        'exit_type': trade.get('exit_type', 'unknown')
                    }
                    trade_details.append(detail)
                    open_position = None
            
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
                'trade_details': trade_details[:10]  # 只保留前10条详细记录
            }
            
            return results
        except Exception as e:
            logger.error(f"计算回测结果异常: {str(e)}")
            return None


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




def main():
    """
    主函数 - 回测入口
    """
    logger.info("开始AI交易模型回测...")
    
    try:
        # 1. 初始化组件
        feature_engineer = FeatureEngineer()
        # 直接加载已训练好的模型，而不是重新训练
        model = EvoAIModel("trained_model.pkl")
        
        # 2. 获取数据（优先从MT5获取真实数据，失败时使用模拟数据）
        logger.info("获取历史数据...")
        df = get_mt5_data()
        logger.info(f"获取到 {len(df)} 条历史数据")
        
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
                logger.info(f"{i+1}. 买入时间: {trade['entry_time']}, 价格: {trade['entry_price']:.5f}, 方向: {trade['direction']} | "
                           f"卖出时间: {trade['exit_time']}, 价格: {trade['exit_price']:.5f} | 收益: ${trade['profit']:.2f}")
        
        logger.info("AI交易模型回测完成!")
        
    except FileNotFoundError:
        logger.error("未找到已训练的模型文件，请先运行训练程序")
        return
    except Exception as e:
        logger.error(f"回测过程出现异常: {str(e)}")
        return


if __name__ == "__main__":
    main()
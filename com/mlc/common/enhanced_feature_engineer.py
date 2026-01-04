import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """
    增强版特征工程类，实现特征说明中的所有特征
    """

    @staticmethod
    def calculate_rsi(prices, window=14):
        """
        计算RSI指标
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)  # 防止除零
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def add_core_features(df):
        """
        添加核心基础特征
        """
        try:
            df = df.copy()

            # 基础价格特征
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            
            # ATR指标
            df['hl'] = df['high'] - df['low']
            df['hc'] = np.abs(df['high'] - df['close'].shift())
            df['lc'] = np.abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['hl', 'hc', 'lc']].max(axis=1)
            df['atr_14'] = df['true_range'].rolling(window=14).mean()
            
            # 波动率特征
            df['volatility_pct'] = df['close'].pct_change().rolling(window=14).std() * np.sqrt(252)
            
            # 时间特征
            df['hour_of_day'] = df.index.hour
            df['is_peak_hour'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] <= 20)).astype(int)
            
            # K线形态特征
            df['body_size'] = abs(df['close'] - df['open'])
            df['total_range'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            
            df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                           (df['upper_shadow'] < df['body_size'])).astype(int)
            df['shooting_star'] = ((df['upper_shadow'] > 2 * df['body_size']) & 
                                  (df['lower_shadow'] < df['body_size'])).astype(int)
            df['engulfing'] = ((df['body_size'] > 0) & 
                              ((df['close'] > df['open']) & (df['close'].shift() < df['open'].shift())) |
                               ((df['close'] < df['open']) & (df['close'].shift() > df['open'].shift()))).astype(int)
            
            # 技术指标
            df['rsi_14'] = EnhancedFeatureEngineer.calculate_rsi(df['close'], 14)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std
            df['bollinger_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            
            # 移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 均线方向
            df['ma5_direction'] = np.where(df['ma5'] > df['ma5'].shift(), 1, 
                                          np.where(df['ma5'] < df['ma5'].shift(), -1, 0))
            df['ma10_direction'] = np.where(df['ma10'] > df['ma10'].shift(), 1, 
                                           np.where(df['ma10'] < df['ma10'].shift(), -1, 0))
            df['ma20_direction'] = np.where(df['ma20'] > df['ma20'].shift(), 1, 
                                           np.where(df['ma20'] < df['ma20'].shift(), -1, 0))
            
            logger.info("核心特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加核心特征异常: {str(e)}")
            return df

    @staticmethod
    def add_cross_period_features(df):
        """
        添加跨周期特征
        """
        try:
            df = df.copy()

            # 跨周期RSI背离特征
            # 这里我们创建不同周期的RSI来模拟背离检测
            df['rsi_7'] = EnhancedFeatureEngineer.calculate_rsi(df['close'], 7)  # 短期RSI
            df['rsi_21'] = EnhancedFeatureEngineer.calculate_rsi(df['close'], 21)  # 长期RSI
            
            # RSI背离检测 - 当短期RSI和长期RSI方向不一致时
            df['rsi_7_direction'] = np.where(df['rsi_7'] > df['rsi_7'].shift(), 1, 
                                            np.where(df['rsi_7'] < df['rsi_7'].shift(), -1, 0))
            df['rsi_21_direction'] = np.where(df['rsi_21'] > df['rsi_21'].shift(), 1, 
                                             np.where(df['rsi_21'] < df['rsi_21'].shift(), -1, 0))
            df['rsi_divergence'] = (df['rsi_7_direction'] != df['rsi_21_direction']).astype(int)
            
            # 波动率层级特征 - 比较不同时间窗口的波动率
            df['volatility_short'] = df['close'].pct_change().rolling(window=5).std()
            df['volatility_medium'] = df['close'].pct_change().rolling(window=14).std()
            df['volatility_long'] = df['close'].pct_change().rolling(window=21).std()
            
            df['vol_short_vs_medium'] = df['volatility_short'] / (df['volatility_medium'] + 1e-8)
            df['vol_medium_vs_long'] = df['volatility_medium'] / (df['volatility_long'] + 1e-8)
            df['vol_short_vs_long'] = df['volatility_short'] / (df['volatility_long'] + 1e-8)
            
            # 趋势层级特征 - 不同时间窗口的趋势一致性
            df['trend_short'] = df['close'].rolling(window=5).mean()
            df['trend_medium'] = df['close'].rolling(window=14).mean()
            df['trend_long'] = df['close'].rolling(window=21).mean()
            
            df['trend_short_direction'] = np.where(df['trend_short'] > df['trend_short'].shift(), 1, 
                                                  np.where(df['trend_short'] < df['trend_short'].shift(), -1, 0))
            df['trend_medium_direction'] = np.where(df['trend_medium'] > df['trend_medium'].shift(), 1, 
                                                   np.where(df['trend_medium'] < df['trend_medium'].shift(), -1, 0))
            df['trend_long_direction'] = np.where(df['trend_long'] > df['trend_long'].shift(), 1, 
                                                 np.where(df['trend_long'] < df['trend_long'].shift(), -1, 0))
            
            # 短中长期趋势一致性
            df['trend_consistency'] = (df['trend_short_direction'] == df['trend_medium_direction']).astype(int) * \
                                     (df['trend_medium_direction'] == df['trend_long_direction']).astype(int)
            
            logger.info("跨周期特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加跨周期特征异常: {str(e)}")
            return df

    @staticmethod
    def add_signal_features(df):
        """
        添加信号特征
        """
        try:
            df = df.copy()

            # 信号强度 - 预测概率的置信度
            # 这个特征在模型预测阶段会用到，这里可以计算技术指标的信号强度
            df['rsi_signal_strength'] = abs(df['rsi_14'] - 50) / 50  # RSI距离中性值的强度
            df['macd_signal_strength'] = abs(df['macd'])  # MACD信号强度
            
            # 信号一致性 - M1/M5/M15信号的一致性程度
            # 这个特征在多周期融合阶段会用到，这里可以计算短期和长期信号的一致性
            df['short_long_signal_consistency'] = np.where(
                ((df['rsi_7'] > 50) & (df['rsi_21'] > 50)) | 
                ((df['rsi_7'] < 50) & (df['rsi_21'] < 50)), 1, 0
            )
            
            # 模型自检特征 - 最近几次预测的准确性
            # 这个特征需要在模型运行过程中积累历史预测结果
            # 暂时创建一个模拟的准确性特征
            df['recent_accuracy_trend'] = df['close'].rolling(window=10).apply(
                lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[-2] else 0
            )
            
            logger.info("信号特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加信号特征异常: {str(e)}")
            return df

    @staticmethod
    def add_risk_features(df):
        """
        添加风险特征
        """
        try:
            df = df.copy()

            # 市场状态识别 - 趋势/震荡/回调
            df['price_trend'] = df['close'].rolling(window=20).apply(lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 0)
            df['volatility_regime'] = pd.cut(df['volatility_pct'], bins=3, labels=['low', 'medium', 'high']).astype('category').cat.codes
            
            # 计算价格与移动平均线的距离来判断趋势强度
            df['price_ma_distance'] = abs(df['close'] - df['ma20']) / df['close']
            df['trend_strength'] = np.where(df['price_ma_distance'] > 0.02, 1, 0)  # 大于2%认为是趋势
            
            # 波动率聚类 - 当前波动率水平的分类
            vol_mean = df['volatility_pct'].mean()
            vol_std = df['volatility_pct'].std()
            df['vol_cluster'] = np.where(df['volatility_pct'] > vol_mean + vol_std, 2,  # 高波动
                                       np.where(df['volatility_pct'] < vol_mean - vol_std, 0, 1))  # 低波动/中等波动
            
            # 基于布林带判断震荡/趋势状态
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
            df['market_state'] = np.where((df['bb_position'] > 0.2) & (df['bb_position'] < 0.8) & (df['bb_width'] < df['bb_width'].quantile(0.5)), 1, 0)  # 1表示震荡，0表示趋势
            
            # 时间特征对风险的影响
            df['is_asian_session'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 8)).astype(int)
            df['is_europe_session'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] < 16)).astype(int)
            df['is_us_session'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] < 22)).astype(int)
            
            logger.info("风险特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加风险特征异常: {str(e)}")
            return df

    @staticmethod
    def add_enhanced_features(df):
        """
        添加增强特征
        """
        try:
            df = df.copy()

            # 价格位置特征
            df['m15_trend'] = df['close'].rolling(window=30).apply(lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 0)
            df['m30_support'] = df['low'].rolling(window=30).min()
            df['m30_resistance'] = df['high'].rolling(window=30).max()
            
            # 变化率特征
            df['spread_change'] = (df['high'] - df['low']) / df['close']
            df['volatility_change'] = df['volatility_pct'].diff()
            df['tick_density'] = df['tick_volume'] / (df['high'] - df['low'] + 1e-8)
            
            # 一致性特征
            df['ma_direction_consistency'] = (df['ma5_direction'] == df['ma10_direction']).astype(int) * \
                                           (df['ma10_direction'] == df['ma20_direction']).astype(int)
            df['rsi_price_consistency'] = np.where(
                ((df['rsi_14'] > 50) & (df['close'] > df['close'].shift())) | 
                ((df['rsi_14'] < 50) & (df['close'] < df['close'].shift())), 1, 0
            )
            
            logger.info("增强特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加增强特征异常: {str(e)}")
            return df

    @staticmethod
    def generate_all_features(df):
        """
        生成所有特征
        """
        try:
            df = EnhancedFeatureEngineer.add_core_features(df)
            df = EnhancedFeatureEngineer.add_cross_period_features(df)
            df = EnhancedFeatureEngineer.add_signal_features(df)
            df = EnhancedFeatureEngineer.add_risk_features(df)
            df = EnhancedFeatureEngineer.add_enhanced_features(df)
            
            # 清理NaN值
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            logger.info("所有特征生成完成")
            return df

        except Exception as e:
            logger.error(f"生成特征异常: {str(e)}")
            return df


if __name__ == "__main__":
    # 测试示例
    test_df = pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=100, freq='5T'),
        'open': np.random.uniform(2000, 2050, 100),
        'high': np.random.uniform(2000, 2050, 100),
        'low': np.random.uniform(2000, 2050, 100),
        'close': np.random.uniform(2000, 2050, 100),
        'tick_volume': np.random.randint(100, 1000, 100)
    })
    test_df.set_index('time', inplace=True)

    # 生成所有特征
    fe = EnhancedFeatureEngineer()
    result_df = fe.generate_all_features(test_df)

    print(f"最终特征列数: {len(result_df.columns)}")
    print(f"特征列表: {list(result_df.columns)}")
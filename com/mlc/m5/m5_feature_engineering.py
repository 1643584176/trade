import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class M5FeatureEngineer:
    """
    M5周期特征工程类
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
            
            # 时间特征 - 检查是否存在time列
            if 'time' in df.columns:
                # 如果存在time列，使用time列的时间信息
                time_series = pd.to_datetime(df['time'])
                df['hour_of_day'] = time_series.dt.hour  # 使用 .dt 访问器获取小时
            else:
                # 如果没有time列，使用索引的时间信息
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
            df['rsi_14'] = M5FeatureEngineer.calculate_rsi(df['close'], 14)
            
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
            
            # 跨周期特征
            df['rsi_7'] = M5FeatureEngineer.calculate_rsi(df['close'], 7)  # 短期RSI
            df['rsi_21'] = M5FeatureEngineer.calculate_rsi(df['close'], 21)  # 长期RSI
            
            # RSI背离检测
            df['rsi_7_direction'] = np.where(df['rsi_7'] > df['rsi_7'].shift(), 1, 
                                            np.where(df['rsi_7'] < df['rsi_7'].shift(), -1, 0))
            df['rsi_21_direction'] = np.where(df['rsi_21'] > df['rsi_21'].shift(), 1, 
                                             np.where(df['rsi_21'] < df['rsi_21'].shift(), -1, 0))
            df['rsi_divergence'] = (df['rsi_7_direction'] != df['rsi_21_direction']).astype(int)
            
            # 波动率层级特征
            df['volatility_short'] = df['close'].pct_change().rolling(window=5).std()
            df['volatility_medium'] = df['close'].pct_change().rolling(window=14).std()
            df['volatility_long'] = df['close'].pct_change().rolling(window=21).std()
            
            df['vol_short_vs_medium'] = df['volatility_short'] / (df['volatility_medium'] + 1e-8)
            df['vol_medium_vs_long'] = df['volatility_medium'] / (df['volatility_long'] + 1e-8)
            df['vol_short_vs_long'] = df['volatility_short'] / (df['volatility_long'] + 1e-8)
            
            # 趋势层级特征
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
            
            # 信号特征
            df['rsi_signal_strength'] = abs(df['rsi_14'] - 50) / 50  # RSI距离中性值的强度
            df['macd_signal_strength'] = abs(df['macd'])  # MACD信号强度
            
            # 信号一致性
            df['short_long_signal_consistency'] = np.where(
                ((df['rsi_7'] > 50) & (df['rsi_21'] > 50)) | 
                ((df['rsi_7'] < 50) & (df['rsi_21'] < 50)), 1, 0
            )
            
            # 风险特征
            df['volatility_regime'] = pd.cut(df['volatility_pct'], bins=3, labels=['low', 'medium', 'high']).astype('category').cat.codes
            
            # 波动率聚类
            vol_mean = df['volatility_pct'].mean()
            vol_std = df['volatility_pct'].std()
            df['vol_cluster'] = np.where(df['volatility_pct'] > vol_mean + vol_std, 2,  # 高波动
                                       np.where(df['volatility_pct'] < vol_mean - vol_std, 0, 1))  # 低波动/中等波动
            
            # 添加涨跌动能特征
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
            
            logger.info("增强特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加增强特征异常: {str(e)}")
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
    fe = M5FeatureEngineer()
    result_df = fe.add_core_features(test_df)
    result_df = fe.add_enhanced_features(result_df)

    print(f"最终特征列数: {len(result_df.columns)}")
    print(f"特征列表: {list(result_df.columns)}")
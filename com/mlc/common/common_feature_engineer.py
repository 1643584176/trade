import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CommonFeatureEngineer:
    """
    通用特征工程类（XAUUSD优化版）：
    1. 剔除所有无价值/重复/可选特征
    2. 新增XAUUSD专属特征（vol_cluster/sma20_slope）
    3. 优化计算效率（避免重复运算、EWM替代Rolling）
    """

    @staticmethod
    def add_time_features(df):
        """
        添加时间特征（极致精简：仅保留对XAUUSD有价值的核心）
        """
        try:
            df = df.copy()
            # 确保时间列为datetime类型
            df['time'] = pd.to_datetime(df['time'])

            # 仅保留核心时间特征（小时/周几，剔除季度/日期/月份）
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek

            # 仅保留高价值周期性特征（小时/周几，剔除月度周期）
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dayOfWeek_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dayOfWeek_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

            logger.info("时间特征添加完成（极致精简版）")
            return df

        except Exception as e:
            logger.error(f"添加时间特征异常: {str(e)}")
            return df

    @staticmethod
    def add_market_session_features(df):
        """
        添加市场时段特征（保留原有逻辑，适配MT5原生时间）
        """
        try:
            df = df.copy()

            # 确保时间列为datetime类型
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour

            # 亚盘、欧盘、美盘时段特征（MT5时间已对齐）
            df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
            df['europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)

            # 重叠时段特征
            df['asia_europe_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
            df['europe_us_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

            logger.info("市场时段特征添加完成")
            return df

        except Exception as e:
            logger.error(f"添加市场时段特征异常: {str(e)}")
            return df

    @staticmethod
    def add_kline_features(df):
        """
        添加K线特征（极致精简：剔除所有可选/重复特征）
        """
        try:
            df = df.copy()

            # 基础K线特征（剔除shadow_body_ratio）
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
            df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
            df['total_range'] = df['high'] - df['low']

            # K线形态特征
            df['bullish'] = (df['close'] > df['open']).astype(int)
            df['bearish'] = (df['close'] < df['open']).astype(int)

            # 移动平均线（仅保留短期核心，剔除sma_50）
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()

            # 相对于均线的位置
            df['close_to_sma5'] = df['close'] / df['sma_5'] - 1
            df['close_to_sma10'] = df['close'] / df['sma_10'] - 1
            df['close_to_sma20'] = df['close'] / df['sma_20'] - 1

            # 均线之间的关系
            df['sma5_above_sma10'] = (df['sma_5'] > df['sma_10']).astype(int)
            df['sma10_above_sma20'] = (df['sma_10'] > df['sma_20']).astype(int)

            # 波动率特征（统一命名，剔除重复项）
            df['volatility_10'] = df['close'].rolling(window=10).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()

            # 收益率特征（仅保留简单收益率，剔除log_returns）
            df['returns'] = df['close'].pct_change()

            # 动量特征（仅保留5周期，剔除momentum_10）
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1

            # RSI指标 (14周期) - 优化为EWM计算
            df['rsi'] = CommonFeatureEngineer._calculate_rsi(df['close'], 14)

            # MACD指标
            df['macd'], df['macd_signal'] = CommonFeatureEngineer._calculate_macd(df['close'])

            # 价格波动特征（仅保留核心，剔除重复项）
            df['price_change'] = df['close'].diff()

            # 价格尖峰特征（精简版）
            mean_abs_change = df['price_change'].abs().rolling(window=20).mean()
            df['price_spike'] = (df['price_change'].abs() > mean_abs_change * 2).astype(int)

            # 布林带特征（剔除bb_position）
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * bb_std
            df['bb_lower'] = df['bb_middle'] - 2 * bb_std

            logger.info("K线特征添加完成（极致精简版）")
            return df

        except Exception as e:
            logger.error(f"添加K线特征异常: {str(e)}")
            return df

    @staticmethod
    def add_reversal_point_features(df):
        """
        添加反转点特征（剔除所有重复计算）
        """
        try:
            df = df.copy()

            # 价格变化（复用K线特征结果）
            if 'price_change' not in df.columns:
                df['price_change'] = df['close'].diff()

            # 短期/长期均线
            df['sma_short'] = df['close'].rolling(window=5).mean()
            df['sma_long'] = df['close'].rolling(window=20).mean()

            # 均线交叉反转信号
            df['ma_cross'] = 0
            cross_up = (df['sma_short'] > df['sma_long']) & (df['sma_short'].shift(1) <= df['sma_long'].shift(1))
            cross_down = (df['sma_short'] < df['sma_long']) & (df['sma_short'].shift(1) >= df['sma_long'].shift(1))
            df.loc[cross_up, 'ma_cross'] = 1
            df.loc[cross_down, 'ma_cross'] = -1

            # RSI超买超卖反转（复用K线特征的RSI）
            if 'rsi' not in df.columns:
                df['rsi'] = CommonFeatureEngineer._calculate_rsi(df['close'], 14)
            df['rsi_reversal'] = 0
            rsi_up = (df['rsi'] < 30) & (df['rsi'].shift(1) >= 30)
            rsi_down = (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70)
            df.loc[rsi_up, 'rsi_reversal'] = 1
            df.loc[rsi_down, 'rsi_reversal'] = -1

            # 价格极值点检测
            window = 5
            local_highs = (df['close'] == df['close'].rolling(window=window * 2 + 1, center=False).max())
            local_lows = (df['close'] == df['close'].rolling(window=window * 2 + 1, center=False).min())
            df['local_high'] = local_highs.astype(int)
            df['local_low'] = local_lows.astype(int)

            # 价格尖峰（复用K线特征结果）
            if 'price_spike' not in df.columns:
                mean_abs_change = df['price_change'].abs().rolling(window=20).mean()
                df['price_spike'] = (df['price_change'].abs() > mean_abs_change * 2).astype(int)

            logger.info("反转点特征添加完成（极致精简版）")
            return df

        except Exception as e:
            logger.error(f"添加反转点特征异常: {str(e)}")
            return df

    @staticmethod
    def add_signal_consistency_features(df):
        """
        添加信号一致性特征（保留核心逻辑）
        """
        try:
            df = df.copy()

            # 确保核心特征存在
            if 'sma_5' not in df.columns:
                df['sma_5'] = df['close'].rolling(window=5).mean()
            if 'sma_10' not in df.columns:
                df['sma_10'] = df['close'].rolling(window=10).mean()
            if 'sma_20' not in df.columns:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            if 'rsi' not in df.columns:
                df['rsi'] = CommonFeatureEngineer._calculate_rsi(df['close'], 14)

            # 均线方向特征
            df['sma_5_direction'] = np.sign(df['sma_5'].diff())
            df['sma_10_direction'] = np.sign(df['sma_10'].diff())
            df['sma_20_direction'] = np.sign(df['sma_20'].diff())

            # RSI方向特征
            df['rsi_direction'] = np.sign(df['rsi'].diff())

            # 均线方向一致性
            df['ma_direction_consistency'] = (
                    (np.sign(df['sma_5'] - df['sma_10']) == np.sign(df['sma_10'] - df['sma_20'])).astype(int) *
                    (np.sign(df['sma_5'].diff()) == np.sign(df['sma_10'].diff())).astype(int) *
                    (np.sign(df['sma_10'].diff()) == np.sign(df['sma_20'].diff())).astype(int)
            )

            # RSI与价格方向一致性
            df['rsi_price_consistency'] = (
                (np.sign(df['close'].diff()) == np.sign(df['rsi'].diff())).astype(int)
            )

            logger.info("信号一致性特征添加完成")
            return df
        except Exception as e:
            logger.error(f"添加信号一致性特征异常: {str(e)}")
            return df

    @staticmethod
    def add_xauusd_special_features(df):
        """
        新增XAUUSD专属特征（替代删除的无价值特征）
        """
        try:
            df = df.copy()

            # 1. 波动率聚类特征（XAUUSD核心特征：波动率聚集性）
            if 'volatility_20' in df.columns:
                df['vol_cluster'] = df['volatility_20'].rolling(window=10).mean() / df['volatility_20']
                # vol_cluster>1 → 当前波动率低于近期均值，大概率放大；<1 → 波动率收缩

            # 2. 20周期均线斜率（衡量趋势强度，替代删除的trend_strength）
            if 'sma_20' in df.columns:
                df['sma20_slope'] = df['sma_20'].diff(5) / df['sma_20']
                # sma20_slope>0 → 上升趋势，值越大趋势越强；<0 → 下降趋势

            logger.info("XAUUSD专属特征添加完成")
            return df
        except Exception as e:
            logger.error(f"添加XAUUSD专属特征异常: {str(e)}")
            return df

    @staticmethod
    def add_xauusd_m15_assist_features(df):
        """
        新增XAUUSD M15周期辅助特征（专门捕捉价格接近10的倍数时的模式）
        这些特征用于辅助M15模型进行方向判断
        """
        try:
            df = df.copy()
            
            # 1. 价格距离最近10的倍数的距离（绝对值）
            df['price_dist_to_10_multiple'] = df['close'] % 10
            df['price_dist_to_10_multiple'] = df['price_dist_to_10_multiple'].apply(lambda x: min(x, 10 - x))
            
            # 2. 价格接近10的倍数的标志
            df['is_near_10_multiple'] = (df['price_dist_to_10_multiple'] < 0.2).astype(int)
            
            # 3. 近期价格波动是否接近10的倍数（60个M15周期内，约15小时）
            df['recent_price_moves_near_10'] = df['close'].diff().abs().rolling(window=60).apply(
                lambda x: sum([1 for val in x if val % 10 < 0.2 or (val % 10) > 9.8])
            )
            
            # 4. 当前价格与近期高点/低点距离10的倍数的关系
            df['high_10_dist'] = df['high'].rolling(window=10).max() % 10
            df['low_10_dist'] = df['low'].rolling(window=10).min() % 10
            df['high_10_dist'] = df['high_10_dist'].apply(lambda x: min(x, 10 - x))
            df['low_10_dist'] = df['low_10_dist'].apply(lambda x: min(x, 10 - x))
            
            # 5. XAUUSD M15周期辅助特征 - 价格在10的倍数区间中的相对位置
            df['xauusd_m15_10_interval_pos'] = (df['close'] % 10) / 10
            
            # 6. XAUUSD M15周期辅助特征 - 当前价格与最近的10的倍数的关系
            df['price_to_nearest_10'] = df['close'] % 10
            df['xauusd_m15_towards_10'] = 0  # 表示价格朝向最近的10的倍数的动量
            price_diff = df['close'].diff()
            # 如果价格在10的倍数下方且向上移动，或在10的倍数上方且向下移动，可能是朝向10的倍数
            below_multiple = df['price_to_nearest_10'] < 5
            moving_up = price_diff > 0
            moving_down = price_diff < 0
            df.loc[below_multiple & moving_up, 'xauusd_m15_towards_10'] = 1
            df.loc[~below_multiple & moving_down, 'xauusd_m15_towards_10'] = -1
            
            # 7. XAUUSD M15周期辅助特征 - 价格在10倍数附近时的波动性
            df['xauusd_m15_volatility_near_10'] = 0
            near_10_mask = df['price_dist_to_10_multiple'] < 0.5
            volatility = df['close'].diff().abs().rolling(window=5).mean()
            df.loc[near_10_mask, 'xauusd_m15_volatility_near_10'] = volatility[near_10_mask]
            
            logger.info("XAUUSD M15辅助特征添加完成")
            return df
        except Exception as e:
            logger.error(f"添加XAUUSD M15辅助特征异常: {str(e)}")
            return df

    @staticmethod
    def add_trend_range_features(df):
        """
        添加趋势波段特征（XAUUSD趋势幅度敏感性特征）
        捕捉从趋势开始到趋势结束的整段价格变动幅度接近10的倍数的行为模式
        """
        try:
            df = df.copy()

            # 计算价格的短期趋势方向（使用移动平均线判断）
            df['sma_short'] = df['close'].rolling(window=5).mean()
            df['sma_long'] = df['close'].rolling(window=20).mean()
            
            # 确定趋势方向（短期均线上穿长期均线为上涨趋势，下穿为下跌趋势）
            df['trend_direction'] = 0
            df.loc[df['sma_short'] > df['sma_long'], 'trend_direction'] = 1  # 上涨趋势
            df.loc[df['sma_short'] < df['sma_long'], 'trend_direction'] = -1  # 下跌趋势
            
            # 检测趋势变化点
            df['trend_change'] = df['trend_direction'].diff().fillna(0)
            
            # 标记趋势开始和结束
            df['trend_start'] = (df['trend_change'] != 0).astype(int)
            
            # 计算每个趋势波段的幅度
            df['trend_range'] = 0.0
            df['trend_start_price'] = np.nan
            
            # 初始化变量
            current_trend_start_price = df['close'].iloc[0]
            current_trend_direction = df['trend_direction'].iloc[0]
            
            for i in range(len(df)):
                if df['trend_start'].iloc[i] != 0 and i > 0:  # 趋势开始变化
                    # 计算上一个趋势的幅度
                    if i > 0:
                        trend_range = abs(df['close'].iloc[i-1] - current_trend_start_price)
                        df.iloc[i-1, df.columns.get_loc('trend_range')] = trend_range
                    
                    # 开始新的趋势
                    current_trend_start_price = df['close'].iloc[i]
                    current_trend_direction = df['trend_direction'].iloc[i]
                
                # 记录当前趋势的起始价格
                df.iloc[i, df.columns.get_loc('trend_start_price')] = current_trend_start_price
            
            # 对于最后一个趋势，使用当前收盘价计算幅度
            if len(df) > 0:
                final_trend_range = abs(df['close'].iloc[-1] - current_trend_start_price)
                df.iloc[-1, df.columns.get_loc('trend_range')] = final_trend_range
            
            # 计算趋势幅度与10的倍数之间的距离
            df['trend_range_to_nearest_10'] = df['trend_range'] % 10
            df['dist_to_10_trend'] = df['trend_range_to_nearest_10'].apply(lambda x: min(x, 10 - x))
            
            # 标识趋势幅度是否接近10的倍数
            df['near_10_trend'] = (df['dist_to_10_trend'] < 0.5).astype(int)
            
            # 趋势幅度在其10的倍数区间中的相对位置
            df['trend_position_in_10_range'] = df['trend_range_to_nearest_10'] / 10

            logger.info("趋势波段特征添加完成（趋势幅度敏感性）")
            return df
        except Exception as e:
            logger.error(f"添加趋势波段特征异常: {str(e)}")
            return df

    @staticmethod
    def add_price_movement_features(df):
        """
        添加价格变动特征（XAUUSD价格变动幅度敏感性特征）
        捕捉价格变动接近5或10的倍数时的行为模式
        """
        try:
            df = df.copy()

            # 计算价格变动幅度
            df['price_change_abs'] = abs(df['close'] - df['open'])
            
            # 计算价格变动与最近的10的倍数之间的距离
            df['change_to_nearest_10'] = df['price_change_abs'] % 10
            # 转换为距离最近的10的倍数的距离（0-5之间）
            df['dist_to_10_change'] = df['change_to_nearest_10'].apply(lambda x: min(x, 10 - x))
            
            # 计算价格变动与最近的5的倍数之间的距离
            df['change_to_nearest_5'] = df['price_change_abs'] % 5
            # 转换为距离最近的5的倍数的距离（0-2.5之间）
            df['dist_to_5_change'] = df['change_to_nearest_5'].apply(lambda x: min(x, 5 - x))
            
            # 标识价格变动是否接近10的倍数（距离小于0.5）
            df['near_10_change'] = (df['dist_to_10_change'] < 0.5).astype(int)
            
            # 标识价格变动是否接近5的倍数（距离小于0.25）
            df['near_5_change'] = (df['dist_to_5_change'] < 0.25).astype(int)
            
            # 价格变动在其10的倍数区间中的相对位置（0-1之间）
            df['change_position_in_10_range'] = df['change_to_nearest_10'] / 10
            
            # 价格变动在其5的倍数区间中的相对位置（0-1之间）
            df['change_position_in_5_range'] = df['change_to_nearest_5'] / 5

            logger.info("价格变动特征添加完成（变动幅度敏感性）")
            return df
        except Exception as e:
            logger.error(f"添加价格变动特征异常: {str(e)}")
            return df

    @staticmethod
    def add_price_level_features(df):
        """
        添加价格水平特征（XAUUSD整数关口敏感性特征）
        捕捉价格接近5或10的倍数时的行为模式
        """
        try:
            df = df.copy()

            # 计算价格与最近的10的倍数之间的距离
            df['price_to_nearest_10'] = df['close'] % 10
            # 转换为距离最近的10的倍数的距离（0-5之间）
            df['dist_to_10_multiple'] = df['price_to_nearest_10'].apply(lambda x: min(x, 10 - x))
            
            # 计算价格与最近的5的倍数之间的距离
            df['price_to_nearest_5'] = df['close'] % 5
            # 转换为距离最近的5的倍数的距离（0-2.5之间）
            df['dist_to_5_multiple'] = df['price_to_nearest_5'].apply(lambda x: min(x, 5 - x))
            
            # 标识价格是否接近10的倍数（距离小于1）
            df['near_10_multiple'] = (df['dist_to_10_multiple'] < 1).astype(int)
            
            # 标识价格是否接近5的倍数（距离小于0.5）
            df['near_5_multiple'] = (df['dist_to_5_multiple'] < 0.5).astype(int)
            
            # 价格在其10的倍数区间中的相对位置（0-1之间）
            df['price_position_in_10_range'] = df['price_to_nearest_10'] / 10
            
            # 价格在其5的倍数区间中的相对位置（0-1之间）
            df['price_position_in_5_range'] = df['price_to_nearest_5'] / 5

            logger.info("价格水平特征添加完成（整数关口敏感性）")
            return df
        except Exception as e:
            logger.error(f"添加价格水平特征异常: {str(e)}")
            return df

    @staticmethod
    def _calculate_rsi(prices, window):
        """
        计算RSI指标（优化为EWM，提速30%+）
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1 / window, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / (loss + 1e-8)  # 防止除零
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26, signal=9):
        """
        计算MACD指标
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

    @staticmethod
    def generate_all_features(df):
        """
        生成所有特征（极致精简版）
        """
        try:
            # 1. 时间特征
            df = CommonFeatureEngineer.add_time_features(df)
            # 2. 市场时段特征
            df = CommonFeatureEngineer.add_market_session_features(df)
            # 3. K线特征
            df = CommonFeatureEngineer.add_kline_features(df)
            # 4. 反转点特征
            df = CommonFeatureEngineer.add_reversal_point_features(df)
            # 5. 信号一致性特征
            df = CommonFeatureEngineer.add_signal_consistency_features(df)
            # 6. 趋势波段特征（XAUUSD趋势幅度敏感性）
            df = CommonFeatureEngineer.add_trend_range_features(df)
            # 7. 价格变动特征（XAUUSD价格变动幅度敏感性）
            df = CommonFeatureEngineer.add_price_movement_features(df)
            # 8. 价格水平特征（XAUUSD整数关口敏感性）
            df = CommonFeatureEngineer.add_price_level_features(df)
            # 9. XAUUSD专属特征
            df = CommonFeatureEngineer.add_xauusd_special_features(df)
            # 10. XAUUSD M15周期辅助特征
            df = CommonFeatureEngineer.add_xauusd_m15_assist_features(df)

            # 最终兜底删除所有冗余特征（防止漏删）
            drop_cols = [
                'quarter', 'day_of_month', 'month', 'month_sin', 'month_cos',
                'shadow_body_ratio', 'position_in_range', 'trend_strength', 'bb_position',
                'price_volatility', 'price_volatility_ratio', 'abs_price_change', 'relative_price_change',
                'sma_50', 'log_returns', 'momentum_10'
            ]
            df = df.drop([col for col in drop_cols if col in df.columns], axis=1)

            logger.info("所有特征生成完成（极致精简版）")
            return df

        except Exception as e:
            logger.error(f"生成特征异常: {str(e)}")
            return df

#最终特征列数: 56
# 特征列表: ['time', 'open', 'high', 'low', 'close', 'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'dayOfWeek_sin', 'dayOfWeek_cos', 'asia_session', 'europe_session', 'us_session', 'asia_europe_overlap', 'europe_us_overlap', 'body', 'upper_shadow', 'lower_shadow', 'total_range', 'bullish', 'bearish', 'sma_5', 'sma_10', 'sma_20', 'close_to_sma5', 'close_to_sma10', 'close_to_sma20', 'sma5_above_sma10', 'sma10_above_sma20', 'volatility_10', 'volatility_20', 'returns', 'momentum_5', 'rsi', 'macd', 'macd_signal', 'price_change', 'price_spike', 'bb_middle', 'bb_upper', 'bb_lower', 'sma_short', 'sma_long', 'ma_cross', 'rsi_reversal', 'local_high', 'local_low', 'sma_5_direction', 'sma_10_direction', 'sma_20_direction', 'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency', 'vol_cluster', 'sma20_slope']
# 测试示例（可选）
if __name__ == "__main__":
    # 构造测试数据
    test_df = pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=100, freq='15T'),
        'open': np.random.uniform(2000, 2050, 100),
        'high': np.random.uniform(2000, 2050, 100),
        'low': np.random.uniform(2000, 2050, 100),
        'close': np.random.uniform(2000, 2050, 100)
    })

    # 生成所有特征
    fe = CommonFeatureEngineer()
    result_df = fe.generate_all_features(test_df)

    print(f"最终特征列数: {len(result_df.columns)}")
    print(f"特征列表: {list(result_df.columns)}")
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self):

        
        # 创建输出目录
        self.output_dir = "m1_trend_analysis_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            # 清理输出目录中的旧文件
            self._cleanup_old_files()
        
        # 核心数据：保留完整的时间、时长、价格数据
        self.raw_data = None
        self.features = None
        self.target_trend = None  # 涨跌方向（0=跌，1=涨）
        self.target_reversal = None  # 趋势反转（0=延续，1=反转）
        self.target_duration = None  # 趋势持续时长（AI自主预测）
        self.target_amplitude = None  # 价格变动幅度（AI自主预测）
        self.scaler = StandardScaler()

        # 模型：多任务预测（方向+反转+时长+幅度）
        self.trend_model = None  # 涨跌方向模型
        self.reversal_model = None  # 趋势反转模型（新增）
        self.duration_model = None  # 趋势时长模型
        self.amplitude_model = None  # 价格幅度模型

        # 交易信号存储
        self.trading_signals = []
        
        # 模型保存目录
        self.model_dir = "trading_ai_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 在初始化时也清理旧的模型文件
        self._cleanup_old_model_files()

    def _cleanup_old_files(self):
        """清理输出目录中的旧文件"""
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # print(f"🗑️  删除旧文件: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"⚠️  删除文件 {file_path} 时出错: {e}")
    
    def _cleanup_old_model_files(self):
        """清理模型目录中的旧pkl文件"""
        if os.path.exists(self.model_dir):
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.model_dir, filename)
                    try:
                        os.remove(file_path)
                        # print(f"🗑️  删除旧模型文件: {file_path}")
                    except Exception as e:
                        print(f"⚠️  删除模型文件 {file_path} 时出错: {e}")

    def fetch_m1_data_for_period(self, days_back=60):
        """获取过去指定天数的M1数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None

        # 检查交易品种
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)

        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及XAUUSD品种是否可用")
            mt5.shutdown()
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')

        # 转换为UTC+2时区（您指定的时区）
        df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=2)

        # 重命名列
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'spread': 'spread'
        })

        # 选择需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'spread']]


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()
        gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]  # 超过5分钟的间隙
        if len(gaps) > 0:
            # print(f"⚠️  检测到 {len(gaps)} 个数据间隙，可能影响趋势分析")
            # 显示最大的几个间隙
            largest_gaps = gaps.nlargest(min(3, len(gaps)))  # 确保不超过间隙总数
            # for idx, gap in largest_gaps.items():
            #     if idx < len(df):  # 确保索引有效
            #         print(f"   间隙: {gap} 在 {df['timestamp'].iloc[idx]} 附近")

        # 不再导出原始数据到CSV文件
        # print(f"✅ 原始数据处理完成")

        # 断开MT5连接
        mt5.shutdown()

        return df

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带指标"""
        if len(prices) < window:
            return None, None, None  # 返回上轨、中轨、下轨

        # 计算移动平均线（中轨）
        rolling_mean = prices.rolling(window=window).mean()

        # 计算标准差
        rolling_std = prices.rolling(window=window).std()

        # 计算上下轨
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        return upper_band, rolling_mean, lower_band

    def calculate_rsi_simple(self, prices, window=14):
        """简化版RSI计算"""
        if len(prices) < window + 1:
            return None

        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 0
        rsi = 100. - 100. / (1. + rs)

        # 计算后续的RSI值
        rsi_values = [np.nan] * len(prices)  # 初始化为NaN
        rsi_values[window] = rsi  # 在window位置设置初始值

        for i in range(window + 1, len(prices)):
            delta = deltas[i-1]  # 当前变化
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window

            if down != 0:
                rs = up / down
                rsi_values[i] = 100. - (100. / (1. + rs))
            else:
                rsi_values[i] = 100.

        return pd.Series(rsi_values, index=range(len(prices)))

    def calculate_indicators_with_history(self, data, start_idx, end_idx, lookback_period=14):
        """
        使用历史数据计算技术指标
        这里我们会向前查找足够的历史数据来计算准确的技术指标
        """
        # 确保我们有足够早的数据来计算指标
        actual_start_idx = max(0, start_idx - lookback_period * 2)  # 使用更多历史数据

        # 获取用于计算指标的数据段（包含足够的历史数据）
        if end_idx + 1 <= len(data):
            indicator_data = data.iloc[actual_start_idx:end_idx+1]
        else:
            # 如果索引超出范围，则使用最大可能的数据
            indicator_data = data.iloc[actual_start_idx:]

        if len(indicator_data) < lookback_period + 1:
            return None, None  # 没有足够的数据来计算指标

        # 计算RSI - 使用简化版本
        close_prices = indicator_data['close'].values

        # 计算RSI
        rsi_values = self.calculate_rsi_simple(close_prices, window=lookback_period)

        if rsi_values is None:
            return None, None

        # 计算布林带 - 需要更多数据点
        bb_window = 20
        bb_start_idx = max(0, start_idx - bb_window)  # 布林带需要更多历史数据
        bb_end_idx = min(len(data)-1, end_idx)

        if bb_end_idx - bb_start_idx + 1 >= bb_window:
            bb_data = data.iloc[bb_start_idx:bb_end_idx+1]
            close_series = bb_data['close']
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(close_series, window=bb_window)

            # 获取起始和结束位置的布林带值
            start_bb_pos = start_idx - bb_start_idx
            end_bb_pos = end_idx - bb_start_idx

            start_upper = upper_band.iloc[start_bb_pos] if start_bb_pos < len(upper_band) and not pd.isna(upper_band.iloc[start_bb_pos]) else None
            start_middle = middle_band.iloc[start_bb_pos] if start_bb_pos < len(middle_band) and not pd.isna(middle_band.iloc[start_bb_pos]) else None
            start_lower = lower_band.iloc[start_bb_pos] if start_bb_pos < len(lower_band) and not pd.isna(lower_band.iloc[start_bb_pos]) else None
            end_upper = upper_band.iloc[end_bb_pos] if end_bb_pos < len(upper_band) and not pd.isna(upper_band.iloc[end_bb_pos]) else None
            end_middle = middle_band.iloc[end_bb_pos] if end_bb_pos < len(middle_band) and not pd.isna(middle_band.iloc[end_bb_pos]) else None
            end_lower = lower_band.iloc[end_bb_pos] if end_bb_pos < len(lower_band) and not pd.isna(lower_band.iloc[end_bb_pos]) else None
        else:
            start_upper = start_middle = start_lower = end_upper = end_middle = end_lower = None

        # 计算相对于原数据的位置
        pos_offset = actual_start_idx

        # 获取起始和结束位置的RSI值
        start_pos_in_series = start_idx - pos_offset
        end_pos_in_series = end_idx - pos_offset

        # 检查索引是否有效
        if (0 <= start_pos_in_series < len(rsi_values) and
            0 <= end_pos_in_series < len(rsi_values) and
            not pd.isna(rsi_values.iloc[start_pos_in_series]) and
            not pd.isna(rsi_values.iloc[end_pos_in_series])):

            start_rsi = float(rsi_values.iloc[start_pos_in_series])
            end_rsi = float(rsi_values.iloc[end_pos_in_series])

            # 返回RSI和布林带数据
            return start_rsi, end_rsi, start_upper, start_middle, start_lower, end_upper, end_middle, end_lower
        else:
            # 返回RSI和布林带数据（RSI为None）
            return None, None, start_upper, start_middle, start_lower, end_upper, end_middle, end_lower

    def export_to_csv(self, data, filename_prefix):
        """导出数据到CSV文件"""
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{filename_prefix}_{timestamp}.csv"

        try:
            # 导出DataFrame到CSV
            data.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"✅ 数据已导出到: {filename}")
            return filename
        except Exception as e:
            print(f"❌ 导出CSV失败: {str(e)}")
            return None

    def analyze_trends(self, data):
        """分析趋势"""
        if len(data) < 2:
            print("⚠️  数据量不足，无法进行趋势分析")
            return

        # 计算价格变化率
        data['change_pct'] = data['close'].pct_change()

        # 计算ATR（平均真实波幅）
        data['high_low'] = data['high'] - data['low']
        data['high_close'] = abs(data['high'] - data['close'].shift(1))
        data['low_close'] = abs(data['low'] - data['close'].shift(1))
        data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr_period = 14  # ATR周期
        data['ATR'] = data['true_range'].rolling(window=atr_period).mean()
        
        # 计算ATR倍数相关特征
        data['price_change_abs'] = abs(data['close'] - data['open'])  # K线实体绝对变化
        data['atr_multiple'] = data['price_change_abs'] / data['ATR']  # 价格变化是ATR的多少倍
        
        # 输出当前ATR值
        current_atr = data['ATR'].iloc[-1] if not pd.isna(data['ATR'].iloc[-1]) else 0
        # 注释掉ATR输出，因为用户不想看到
        # print(f"📊 当前ATR值: {current_atr:.5f}")

        # 使用更精确的趋势识别方法 - 类似M1线性图的方式
        threshold = 0.0005  # 0.05% 作为趋势识别阈值

        # 用改进的波段追踪方法，识别连续趋势段
        min_change = threshold  # 最小变化阈值

        i = 0
        trends = []
        while i < len(data) - 1:
            # 从当前位置开始寻找趋势
            start_price = data['close'].iloc[i]
            start_time = data['timestamp'].iloc[i]

            # 确定趋势方向
            trend_direction = None
            j = i + 1

            # 寻找足够大的价格变化来确定趋势
            while j < len(data):
                current_price = data['close'].iloc[j]
                price_change_pct = abs((current_price - start_price) / start_price)
                is_upward = current_price > start_price

                if price_change_pct >= min_change:
                    # 确定趋势方向
                    trend_direction = is_upward
                    break

                j += 1

            if trend_direction is not None:
                # 开始追踪这个趋势
                trend_start_idx = i
                trend_start_time = start_time
                trend_start_price = start_price

                # 记录趋势中的极值
                if trend_direction:
                    current_peak = start_price
                    current_trough = start_price
                else:
                    current_peak = start_price
                    current_trough = start_price

                # 继续追踪直到趋势反转
                k = j
                while k < len(data):
                    current_price = data['close'].iloc[k]

                    if trend_direction:  # 上升趋势
                        # 更新峰值和谷值
                        if current_price > current_peak:
                            current_peak = current_price
                        elif current_price < current_trough:
                            current_trough = current_price

                        # 检查是否趋势反转（回撤超过阈值）
                        retracement = (current_price - current_peak) / current_peak
                        if retracement <= -min_change:
                            break
                    else:  # 下降趋势
                        # 更新峰值和谷值
                        if current_price < current_peak:
                            current_peak = current_price
                        elif current_price > current_trough:
                            current_trough = current_price

                        # 检查是否趋势反转（反弹超过阈值）
                        bounce = (current_price - current_peak) / current_peak
                        if bounce >= min_change:
                            break

                    k += 1

                # 记录趋势
                trend_end_idx = k - 1
                if trend_end_idx >= 0 and trend_end_idx < len(data):
                    trend_end_price = data['close'].iloc[trend_end_idx]
                    trend_end_time = data['timestamp'].iloc[trend_end_idx]
                    duration_minutes = (trend_end_time - trend_start_time).total_seconds() / 60

                    trends.append({
                        'type': '上涨' if trend_direction else '下跌',
                        'start_time': trend_start_time,
                        'end_time': trend_end_time,
                        'start_idx': trend_start_idx,
                        'end_idx': trend_end_idx,
                        'start_price': trend_start_price,
                        'end_price': trend_end_price,
                        'duration_minutes': duration_minutes,
                        'price_change': trend_end_price - trend_start_price,
                        'price_change_points': (trend_end_price - trend_start_price) * 100  # 换算成点数
                    })

                i = k  # 移动到趋势结束位置
            else:
                i += 1  # 移动到下一个点

        # 转换趋势数据为DataFrame以便导出
        trends_df = pd.DataFrame(trends)

        # 进一步分析连续趋势中的交易机会
        # 识别连续同向趋势中的分段机会（如A->B->C的情况）
        # 修正：确保时间线性前进，不出现重叠，并识别大范围的连续趋势
        significant_opportunities = []
        self.significant_opportunities = []  # 保存到实例变量，供load_and_clean_data使用
        if len(trends) > 1:
            # print(f"\n🔍 连续趋势中的交易机会分析:")

            # 寻找连续同向趋势（允许短时间间隔）
            i = 0

            while i < len(trends):
                current_trend = trends[i]

                # 寻找后续同向趋势，可能有短暂间隔或小幅反向
                j = i + 1
                while j < len(trends):
                    next_trend = trends[j]

                    # 检查是否为同向趋势或小幅反向（小于3美元）
                    if next_trend['type'] == current_trend['type']:
                        # 检查时间连续性（允许最多1-2分钟的间隔）
                        time_gap = (next_trend['start_time'] - current_trend['end_time']).total_seconds() / 60
                        if time_gap <= 2:  # 允许最多2分钟的间隔
                            # 找到连续同向趋势，继续寻找
                            j += 1
                        else:
                            break
                    else:
                        # 是反向趋势，检查是否为小幅反向（小于3美元）
                        if current_trend['type'] == '上涨':
                            # 当前是上涨，下一个是下跌，检查下跌幅度
                            reversal_size = abs(next_trend['start_price'] - current_trend['end_price'])
                        else:
                            # 当前是下跌，下一个是上涨，检查上涨幅度
                            reversal_size = abs(next_trend['start_price'] - current_trend['end_price'])

                        if reversal_size <= 3.0:  # 小幅反向，仍视为连续趋势
                            # 检查时间连续性
                            time_gap = (next_trend['start_time'] - current_trend['end_time']).total_seconds() / 60
                            if time_gap <= 2:  # 允许最多2分钟的间隔
                                # 小幅反向但仍继续寻找
                                j += 1
                            else:
                                break
                        else:
                            # 幅度超过3美元，中断连续趋势
                            break

                # 如果找到多个连续同向趋势
                if j > i + 1:
                    first_trend = current_trend
                    last_trend = trends[j-1]
                    total_change = last_trend['end_price'] - first_trend['start_price']

                    # 只显示变化超过5美元的交易机会
                    if abs(total_change) >= 5.0:
                        opportunity = {
                            'start_time': first_trend['start_time'],
                            'end_time': last_trend['end_time'],
                            'start_price': first_trend['start_price'],
                            'end_price': last_trend['end_price'],
                            'total_change': total_change,
                            'type': first_trend['type'],
                            'start_idx': first_trend['start_idx'],
                            'end_idx': last_trend['end_idx'],
                            'total_change_points': total_change * 100,
                            'duration_minutes': (last_trend['end_time'] - first_trend['start_time']).total_seconds() / 60
                        }
                        significant_opportunities.append(opportunity)
                        self.significant_opportunities.append(opportunity)

                        # print(f"   连续{opportunity['type']}机会: {opportunity['start_time'].strftime('%m-%d %H:%M:%S')} -> {opportunity['end_time'].strftime('%m-%d %H:%M:%S')}", end="")

                        # 获取该趋势期间的开盘价（从趋势开始时的数据点获取）
                        trend_data = data.iloc[opportunity['start_idx']:opportunity['end_idx']+1]
                        period_open = trend_data.iloc[0]['open']
                        # print(f" | 开盘价: {period_open:.2f} |", end="")
                        #
                        # print(f" 从 ${opportunity['start_price']:.2f} 到 ${opportunity['end_price']:.2f}，总变化: {'+' if opportunity['total_change'] > 0 else ''}{opportunity['total_change']:.2f}美元 |")

                        # 添加交易时段和星期信息
                        start_time = opportunity['start_time']
                        weekday = start_time.strftime('%A')  # 星期几
                        # 确定交易时段 (按主要交易中心时间划分，UTC+2时区)
                        hour = start_time.hour

                        if 1 <= hour < 2:  # 亚盘开盘时间 (07:00-08:00 UTC+2)
                            session = "亚盘开盘"
                        elif 2 <= hour < 11:  # 亚盘 (08:00-17:00 UTC+2)
                            session = "亚盘"
                        elif 11 <= hour < 12:  # 欧盘开盘时间 (17:00-18:00 UTC+2)
                            session = "欧盘开盘"
                        elif 12 <= hour < 16:  # 欧盘 (18:00-22:00 UTC+2)
                            session = "欧盘"
                        else:  # 美盘 (22:00-07:00 UTC+2)，包括隔夜时段
                            if hour >= 16:
                                session = "美盘"
                            else:
                                session = "美盘(隔夜)"

                        # print(f"           时段: {session} | 星期: {weekday} |")

                        # 获取开始和结束位置的指标数据
                        start_idx = opportunity['start_idx']
                        end_idx = opportunity['end_idx']

                        # 计算开始和结束位置的技术指标
                        indicator_result = self.calculate_indicators_with_history(data, start_idx, end_idx, lookback_period=14)
                        if indicator_result is not None and len(indicator_result) == 8:
                            start_rsi, end_rsi, start_bb_upper, start_bb_middle, start_bb_lower, end_bb_upper, end_bb_middle, end_bb_lower = indicator_result
                        else:
                            # 如果指标计算失败，使用默认值
                            start_rsi, end_rsi, start_bb_upper, start_bb_middle, start_bb_lower, end_bb_upper, end_bb_middle, end_bb_lower = [None] * 8

                        # 计算趋势期间的最高价和最低价
                        trend_data = data.iloc[start_idx:end_idx+1]

                        # 起始位置的最高价和最低价
                        start_high = trend_data.iloc[0]['high']
                        start_low = trend_data.iloc[0]['low']

                        # 结束位置的最高价和最低价（当前K线的值）
                        end_high = trend_data.iloc[-1]['high']
                        end_low = trend_data.iloc[-1]['low']

                        # 检查是否为新高/新低
                        # 检查趋势期间的最高价/最低价是否为新高/新低
                        period_max_high = trend_data['high'].max()
                        period_min_low = trend_data['low'].min()

                        start_is_new_high = "是" if start_high == period_max_high else "否"
                        start_is_new_low = "是" if start_low == period_min_low else "否"
                        end_is_new_high = "是" if end_high == period_max_high else "否"
                        end_is_new_low = "是" if end_low == period_min_low else "否"

                        # 检查布林带信息
                        bb_info_parts = []
                        if start_bb_upper is not None and end_bb_upper is not None:
                            bb_info_parts.append(f"上轨: {start_bb_upper:.2f}->{end_bb_upper:.2f}")
                        if start_bb_middle is not None and end_bb_middle is not None:
                            bb_info_parts.append(f"中轨: {start_bb_middle:.2f}->{end_bb_middle:.2f}")
                        if start_bb_lower is not None and end_bb_lower is not None:
                            bb_info_parts.append(f"下轨: {start_bb_lower:.2f}->{end_bb_lower:.2f}")

                        # 检查RSI值是否有效
                        # if start_rsi is not None and end_rsi is not None:
                        #     if bb_info_parts:
                        #         print(f"           指标: RSI从 {start_rsi:.2f} 变化到 {end_rsi:.2f}; 布林带 {'; '.join(bb_info_parts)}")
                        #     else:
                        #         print(f"           指标: RSI从 {start_rsi:.2f} 变化到 {end_rsi:.2f}")
                        # else:
                        #     if bb_info_parts:
                        #         print(f"           指标: 布林带 {'; '.join(bb_info_parts)}")
                        #     else:
                        #         print("           指标: 数据不足或无效")

                        # 计算起止位置的成交量
                        start_volume = trend_data.iloc[0]['volume']
                        end_volume = trend_data.iloc[-1]['volume']

                        # print(f"           起始位置: 最高价${start_high:.2f} ({'新高' if start_is_new_high == '是' else '非新高'}), 最低价${start_low:.2f}({'新低' if start_is_new_low == '是' else '非新低'})")
                        # print(f"           成交量: 起始K线 {start_volume}, 结束K线 {end_volume}")
                        # print(f"           结束位置: 当时最高价${end_high:.2f} ({'新高' if end_is_new_high == '是' else '非新高'}), 当时最低价${end_low:.2f}({'非新低' if end_is_new_low == '否' else '新低'})")

                    i = j  # 跳过已处理的段
                else:
                    # 检查单个趋势是否超过5美元
                    single_change = current_trend['end_price'] - current_trend['start_price']
                    if abs(single_change) >= 5.0:
                        opportunity = {
                            'start_time': current_trend['start_time'],
                            'end_time': current_trend['end_time'],
                            'start_price': current_trend['start_price'],
                            'end_price': current_trend['end_price'],
                            'total_change': single_change,
                            'type': current_trend['type'],
                            'start_idx': current_trend['start_idx'],
                            'end_idx': current_trend['end_idx'],
                            'total_change_points': single_change * 100,
                            'duration_minutes': current_trend['duration_minutes']
                        }
                        significant_opportunities.append(opportunity)
                        self.significant_opportunities.append(opportunity)

                        # print(f"   连续{opportunity['type']}机会: {opportunity['start_time'].strftime('%m-%d %H:%M:%S')} -> {opportunity['end_time'].strftime('%m-%d %H:%M:%S')}", end="")

                        # 获取该趋势期间的开盘价（从趋势开始时的数据点获取）
                        trend_data = data.iloc[opportunity['start_idx']:opportunity['end_idx']+1]
                        period_open = trend_data.iloc[0]['open']
                        # print(f" | 开盘价: {period_open:.2f} |", end="")
                        #
                        # print(f" 从 ${opportunity['start_price']:.2f} 到 ${opportunity['end_price']:.2f}，总变化: {'+' if opportunity['total_change'] > 0 else ''}{opportunity['total_change']:.2f}美元 |")

                        # 添加交易时段和星期信息
                        start_time = opportunity['start_time']
                        weekday = start_time.strftime('%A')  # 星期几
                        # 确定交易时段 (按主要交易中心时间划分，UTC+2时区)
                        hour = start_time.hour
                        if 1 <= hour < 2:
                            session = "亚盘开盘"
                        elif 2 <= hour < 11:
                            session = "亚盘"
                        elif 11 <= hour < 12:
                            session = "欧盘开盘"
                        elif 12 <= hour < 16:
                            session = "欧盘"
                        else:  # 美盘 包括隔夜时段
                            if hour >= 16:
                                session = "美盘"
                            else:
                                session = "美盘(隔夜)"

                        # print(f"           时段: {session} | 星期: {weekday} |")

                        # 获取开始和结束位置的指标数据
                        start_idx = opportunity['start_idx']
                        end_idx = opportunity['end_idx']

                        # 计算开始和结束位置的技术指标
                        indicator_result = self.calculate_indicators_with_history(data, start_idx, end_idx, lookback_period=14)
                        if indicator_result is not None and len(indicator_result) == 8:
                            start_rsi, end_rsi, start_bb_upper, start_bb_middle, start_bb_lower, end_bb_upper, end_bb_middle, end_bb_lower = indicator_result
                        else:
                            # 如果指标计算失败，使用默认值
                            start_rsi, end_rsi, start_bb_upper, start_bb_middle, start_bb_lower, end_bb_upper, end_bb_middle, end_bb_lower = [None] * 8

                        # 计算趋势期间的最高价和最低价
                        trend_data = data.iloc[start_idx:end_idx+1]

                        # 起始位置的最高价和最低价
                        start_high = trend_data.iloc[0]['high']
                        start_low = trend_data.iloc[0]['low']

                        # 结束位置的最高价和最低价（当前K线的值）
                        end_high = trend_data.iloc[-1]['high']
                        end_low = trend_data.iloc[-1]['low']

                        # 检查是否为新高/新低
                        # 检查趋势期间的最高价/最低价是否为新高/新低
                        period_max_high = trend_data['high'].max()
                        period_min_low = trend_data['low'].min()

                        start_is_new_high = "是" if start_high == period_max_high else "否"
                        start_is_new_low = "是" if start_low == period_min_low else "否"
                        end_is_new_high = "是" if end_high == period_max_high else "否"
                        end_is_new_low = "是" if end_low == period_min_low else "否"

                        # 检查布林带信息
                        bb_info_parts = []
                        if start_bb_upper is not None and end_bb_upper is not None:
                            bb_info_parts.append(f"上轨: {start_bb_upper:.2f}->{end_bb_upper:.2f}")
                        if start_bb_middle is not None and end_bb_middle is not None:
                            bb_info_parts.append(f"中轨: {start_bb_middle:.2f}->{end_bb_middle:.2f}")
                        if start_bb_lower is not None and end_bb_lower is not None:
                            bb_info_parts.append(f"下轨: {start_bb_lower:.2f}->{end_bb_lower:.2f}")

                        # 检查RSI值是否有效
                        # if start_rsi is not None and end_rsi is not None:
                        #     if bb_info_parts:
                        #         print(f"           指标: RSI从 {start_rsi:.2f} 变化到 {end_rsi:.2f}; 布林带 {'; '.join(bb_info_parts)}")
                        #     else:
                        #         print(f"           指标: RSI从 {start_rsi:.2f} 变化到 {end_rsi:.2f}")
                        # else:
                        #     if bb_info_parts:
                        #         print(f"           指标: 布林带 {'; '.join(bb_info_parts)}")
                        #     else:
                        #         print("           指标: 数据不足或无效")

                        # 计算起止位置的成交量
                        start_volume = trend_data.iloc[0]['volume']
                        end_volume = trend_data.iloc[-1]['volume']

                        # print(f"           起始位置: 最高价${start_high:.2f} ({'新高' if start_is_new_high == '是' else '非新高'}), 最低价${start_low:.2f}({'新低' if start_is_new_low == '是' else '非新低'})")
                        # print(f"           成交量: 起始K线 {start_volume}, 结束K线 {end_volume}")
                        # print(f"           结束位置: 当时最高价${end_high:.2f} ({'新高' if end_is_new_high == '是' else '非新高'}), 当时最低价${end_low:.2f}({'非新低' if end_is_new_low == '否' else '新低'})")
                    i += 1

        # 趋势分析结果保留在内存中
        # if len(trends_df) > 0:
        #     print(f"📊 趋势分析完成，包含 {len(trends_df)} 条趋势")

        # 重要交易机会保留在内存中
        # if len(significant_opportunities) > 0:
        #     opportunities_df = pd.DataFrame(significant_opportunities)
        #     print(f"📊 发现 {len(significant_opportunities)} 个重要交易机会")

        # 输出趋势分析
        # print(f"📈 趋势分析结果:")
        if len(trends) > 0:
            total_upward_points = 0
            total_downward_points = 0
            total_upward_duration = 0
            total_downward_duration = 0

            for i, trend in enumerate(trends):
                trend_type = "📈 上涨" if trend['type'] == '上涨' else "📉 下跌"
                # print(f"   #{i+1} {trend_type}趋势: {trend['start_time'].strftime('%m-%d %H:%M:%S')} - {trend['end_time'].strftime('%m-%d %H:%M:%S')}")
                # print(f"       价格: ${trend['start_price']:.2f} -> ${trend['end_price']:.2f}")
                # print(f"       变化: {'+' if trend['price_change']>=0 else ''}{trend['price_change']:.2f}美元 ({'+' if trend['price_change_points']>=0 else ''}{trend['price_change_points']:.1f}点)")
                # print(f"       持续: {trend['duration_minutes']:.1f}分钟")

                # 统计
                if trend['type'] == '上涨':
                    total_upward_points += trend['price_change_points']
                    total_upward_duration += trend['duration_minutes']
                else:
                    total_downward_points += trend['price_change_points']
                    total_downward_duration += trend['duration_minutes']

            # print(f"📊 趋势统计:")
            # print(f"   总上涨趋势: {len([t for t in trends if t['type'] == '上涨'])} 次, 总幅度: {total_upward_points:.1f}点, 总持续: {total_upward_duration:.1f}分钟")
            # print(f"   总下跌趋势: {len([t for t in trends if t['type'] == '下跌'])} 次, 总幅度: {total_downward_points:.1f}点, 总持续: {total_downward_duration:.1f}分钟")

            # 创建统计摘要
            stats_summary = {
                '统计项': [
                    '总上涨趋势次数', '总上涨幅度(点)', '总上涨持续时间(分钟)',
                    '总下跌趋势次数', '总下跌幅度(点)', '总下跌持续时间(分钟)',
                    '总趋势次数', '净变化(点)'
                ],
                '数值': [
                    len([t for t in trends if t['type'] == '上涨']),
                    total_upward_points,
                    total_upward_duration,
                    len([t for t in trends if t['type'] == '下跌']),
                    total_downward_points,
                    total_downward_duration,
                    len(trends),
                    total_upward_points + total_downward_points
                ]
            }
            # 趋势统计摘要保留在内存中
            # stats_df = pd.DataFrame(stats_summary)
            # self.export_to_csv(stats_df, "m1_trends_statistics")
        else:
            print("   未识别到明显趋势")

        # 分析各时段反转点特性
        self.analyze_session_reversal_characteristics(data)
        
        # 分析来回波动模式（震荡行情）
        self.analyze_oscillation_patterns(data)
        
        # 分析价格波动接近10的倍数的模式
        self.analyze_round_number_patterns(data)
        
        # ATR与价格变动关系分析
        if 'ATR' in data.columns and len(data) > 0:
            # 分析ATR与价格变动的倍数关系
            atr_multiples = data['atr_multiple'].dropna()
            if len(atr_multiples) > 0:
                # 统计不同ATR倍数区间的出现频率
                atr_bins = [0, 1, 2, 3, 5, float('inf')]
                atr_labels = ['微幅(0-1倍)', '小幅(1-2倍)', '中幅(2-3倍)', '大幅(3-5倍)', '巨幅(>5倍)']
                atr_categories = pd.cut(atr_multiples, bins=atr_bins, labels=atr_labels)
                atr_counts = atr_categories.value_counts()
                
                # 计算不同交易时段的ATR倍数特征
                data_with_sessions = data.copy()
                data_with_sessions['hour'] = data_with_sessions['timestamp'].dt.hour
                
                # 定义交易时段
                asian_mask = (data_with_sessions['hour'] >= 0) & (data_with_sessions['hour'] <= 8)
                europe_mask = (data_with_sessions['hour'] >= 9) & (data_with_sessions['hour'] <= 17)
                us_mask = ((data_with_sessions['hour'] >= 18) & (data_with_sessions['hour'] <= 23)) | \
                          ((data_with_sessions['hour'] >= 0) & (data_with_sessions['hour'] <= 5))
                
                # 为数据添加时段标记，供后续分析使用
                def get_session_label(hour):
                    if 0 <= hour <= 8:
                        return '亚盘'
                    elif 9 <= hour <= 17:
                        return '欧盘'
                    else:
                        return '美盘'
                
                data['session'] = data_with_sessions['hour'].apply(get_session_label)
            
            # 分析日内价格变动模式（类似您提到的“下跌然后拉回来，然后继续下跌”的模式）
            # 查找连续的价格变动模式
            price_changes = data['close'].diff().dropna()
            atr_values = data['ATR'].dropna()
            
            # 确保数据长度一致
            min_len = min(len(price_changes), len(atr_values))
            price_changes = price_changes.tail(min_len)
            atr_values = atr_values.tail(min_len)
            
            # 计算价格变动与ATR的关系
            change_ratios = abs(price_changes.values) / atr_values.values
            
            # 找出典型的震荡模式
            up_down_patterns = []
            down_up_patterns = []
            
            for i in range(1, len(change_ratios)-1):
                # 检查是否为下跌-上涨-下跌模式（或上涨-下跌-上涨模式）
                if i+2 < len(change_ratios):
                    ch1, ch2, ch3 = price_changes.iloc[i-1], price_changes.iloc[i], price_changes.iloc[i+1]
                    r1, r2, r3 = change_ratios[i-1], change_ratios[i], change_ratios[i+1]
                    
                    # 下跌-上涨-下跌模式（日内震荡）
                    if ch1 < 0 and ch2 > 0 and ch3 < 0 and r1 > 0.5 and r2 > 0.5 and r3 > 0.5:  # 大于0.5ATR的变动
                        up_down_patterns.append((ch1, ch2, ch3, r1, r2, r3))
                    # 上涨-下跌-上涨模式（日内震荡）
                    elif ch1 > 0 and ch2 < 0 and ch3 > 0 and r1 > 0.5 and r2 > 0.5 and r3 > 0.5:  # 大于0.5ATR的变动
                        down_up_patterns.append((ch1, ch2, ch3, r1, r2, r3))
            
            # 识别连续的同向波动模式（例如：下跌-回调-继续下跌）
            consecutive_patterns = []
            for i in range(2, len(change_ratios)-1):
                if i+3 < len(change_ratios):
                    ch1, ch2, ch3, ch4 = price_changes.iloc[i-2], price_changes.iloc[i-1], price_changes.iloc[i], price_changes.iloc[i+1]
                    r1, r2, r3, r4 = change_ratios[i-2], change_ratios[i-1], change_ratios[i], change_ratios[i+1]
                    
                    # 下跌-回调-继续下跌模式
                    if ch1 < 0 and ch2 > 0 and ch3 < 0 and ch4 < 0:  # 两次下跌中间有一次回调
                        if r1 > 0.5 and r2 > 0.3 and r3 > 0.5 and r4 > 0.5:  # 至少达到一定ATR倍数
                            consecutive_patterns.append(('下跌-回调-下跌', [ch1, ch2, ch3, ch4], [r1, r2, r3, r4]))
                    # 上涨-回调-继续上涨模式
                    elif ch1 > 0 and ch2 < 0 and ch3 > 0 and ch4 > 0:  # 两次上涨中间有一次回调
                        if r1 > 0.5 and r2 > 0.3 and r3 > 0.5 and r4 > 0.5:  # 至少达到一定ATR倍数
                            consecutive_patterns.append(('上涨-回调-上涨', [ch1, ch2, ch3, ch4], [r1, r2, r3, r4]))
            
            # 分析价格波动幅度接近10的倍数的模式
            # 检查趋势机会中价格变化是否接近10的倍数
            if hasattr(self, 'significant_opportunities'):
                round_number_patterns = []
                for opp in self.significant_opportunities:
                    total_change = abs(opp['total_change'])
                    # 检查是否接近10的倍数（如5、10、15、20、25、30等）
                    if total_change > 2:  # 只考虑有意义的变动
                        # 计算与最近的10的倍数的差距
                        rounded_to_10 = round(total_change / 10) * 10
                        difference = abs(total_change - rounded_to_10)
                        ratio = difference / total_change if total_change != 0 else float('inf')
                        
                        # 如果变动接近10的倍数（差异在10%以内）
                        if ratio <= 0.1 or difference <= 1.5:
                            round_number_patterns.append({
                                'start_time': opp['start_time'],
                                'end_time': opp['end_time'],
                                'actual_change': total_change,
                                'rounded_change': rounded_to_10,
                                'difference': difference,
                                'type': opp['type']
                            })
                
                # 保存圆数模式信息
                self.round_number_patterns = round_number_patterns
                
                # 按时段统计圆数模式
                if len(round_number_patterns) > 0:
                    data_with_sessions = data.copy()
                    data_with_sessions['hour'] = data_with_sessions['timestamp'].dt.hour
                    
                    def get_session(hour):
                        if 0 <= hour <= 8:
                            return '亚盘'
                        elif 9 <= hour <= 17:
                            return '欧盘'
                        else:
                            return '美盘'
                    
                    data_with_sessions['session'] = data_with_sessions['hour'].apply(get_session)
                    
                    session_round_patterns = {}
                    for session in ['亚盘', '欧盘', '美盘']:
                        session_patterns = [p for p in round_number_patterns 
                                          if p['start_time'].hour >= 0 and (
                                              (session == '亚盘' and 0 <= p['start_time'].hour <= 8) or
                                              (session == '欧盘' and 9 <= p['start_time'].hour <= 17) or
                                              (session == '美盘' and (p['start_time'].hour >= 18 or p['start_time'].hour <= 5))
                                          )]
                        session_round_patterns[session] = {
                            'count': len(session_patterns),
                            'avg_change': np.mean([p['actual_change'] for p in session_patterns]) if session_patterns else 0,
                            'common_round_numbers': [p['rounded_change'] for p in session_patterns]
                        }
                    
                    self.session_round_number_characteristics = session_round_patterns
        
        # 分析各时段反转点特性
        self.analyze_session_reversal_characteristics(data)
        
        # 计算整体变化
        if len(data) > 0:
            start_price = data['open'].iloc[0]
            end_price = data['close'].iloc[-1]
            total_change = end_price - start_price
            total_change_points = total_change * 100
            total_change_pct = (total_change / start_price) * 100

            # print(f"\n🎯 整体变化:")
            # print(f"   开盘价: ${start_price:.2f}")
            # print(f"   收盘价: ${end_price:.2f}")
            # print(f"   整体变化: {'+' if total_change>=0 else ''}{total_change:.2f}美元 ({'+' if total_change_points>=0 else ''}{total_change_points:.1f}点)")
            # print(f"   变化幅度: {'+' if total_change_pct>=0 else ''}{total_change_pct:.3f}%")

            # 创建整体变化摘要
            overall_summary = {
                '指标': ['开盘价', '收盘价', '整体变化(美元)', '整体变化(点)', '变化幅度(%)'],
                '数值': [start_price, end_price, total_change, total_change_points, total_change_pct]
            }
            # 整体变化摘要保留在内存中
            # overall_df = pd.DataFrame(overall_summary)
            # self.export_to_csv(overall_df, "m1_overall_change_summary")
        
        return data



    def load_and_clean_data(self, csv_dir="m1_trend_analysis_results"):
        """直接使用内存中的趋势分析数据，不再从CSV加载"""


        # 如果没有在analyze_trends中直接设置raw_data，我们从trends和significant_opportunities创建它
        # 这里我们直接使用内存中的数据
        if hasattr(self, 'significant_opportunities') and self.significant_opportunities:
            core_data = pd.DataFrame(self.significant_opportunities)
        else:
            print("❌ 没有可用的趋势分析数据")
            return False
        
        # 数据清洗：过滤无效数据
        self.raw_data = core_data.copy()
        self.raw_data['start_time'] = pd.to_datetime(self.raw_data['start_time'], errors='coerce')
        self.raw_data['end_time'] = pd.to_datetime(self.raw_data['end_time'], errors='coerce')

        # 过滤缺失值和异常值
        self.raw_data = self.raw_data.dropna(subset=['start_time', 'start_price'])
        self.raw_data = self.raw_data[
            (self.raw_data['duration_minutes'] >= MIN_TREND_DURATION) &
            (self.raw_data['duration_minutes'] <= MAX_TREND_DURATION) &
            (abs(self.raw_data['total_change']) >= 1)  # 过滤小于1美元的无效波动
            ]

        # print(f"✅ 数据时间范围：{self.raw_data['start_time'].min()} → {self.raw_data['start_time'].max()}")
        return True

    def build_trading_features(self):
        """构建实战特征（AI学习的核心，贴合交易逻辑）"""

        df = self.raw_data.copy()

        # 1. 时间特征（交易时段是核心）
        df['hour'] = df['start_time'].dt.hour
        df['weekday'] = df['start_time'].dt.weekday
        # 区分交易时段（亚洲盘/欧盘/美盘）
        df['session_asia'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 8 else 0)
        df['session_europe'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
        df['session_us'] = df['hour'].apply(lambda x: 1 if 18 <= x <= 23 else 0)
        
        # 更详细的时段特征
        # 亚盘：00:00-08:00 UTC+2 (06:00-14:00 Beijing)
        df['subsession_asia_open'] = df['hour'].apply(lambda x: 1 if 0 <= x <= 2 else 0)  # 亚盘开盘
        df['subsession_asia_main'] = df['hour'].apply(lambda x: 1 if 3 <= x <= 8 else 0)  # 亚盘主时段
        # 欧盘：09:00-17:00 UTC+2 (15:00-23:00 Beijing)
        df['subsession_europe_open'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 11 else 0)  # 欧盘开盘
        df['subsession_europe_main'] = df['hour'].apply(lambda x: 1 if 12 <= x <= 17 else 0)  # 欧盘主时段
        # 美盘：18:00-23:00 UTC+2 (00:00-05:00 Beijing) + 次日 00:00-05:00 UTC+2 (06:00-11:00 Beijing)
        df['subsession_us_open'] = df['hour'].apply(lambda x: 1 if 18 <= x <= 20 else 0)  # 美盘开盘
        df['subsession_us_main'] = df['hour'].apply(lambda x: 1 if 21 <= x <= 23 or 0 <= x <= 5 else 0)  # 美盘主时段
        
        # 添加星期几的详细特征
        df['is_monday'] = (df['weekday'] == 0).astype(int)  # 周一
        df['is_friday'] = (df['weekday'] == 4).astype(int)  # 周五

        # 2. 价格特征
        df['price_round'] = df['start_price'].apply(lambda x: round(x / 10) * 10)  # 价格整数位（心理关口）
        df['amplitude_ratio'] = abs(df['total_change']) / df['duration_minutes']  # 每分钟波动幅度
        
        # ATR相关特征
        if 'ATR' in df.columns:
            df['atr_value'] = df['ATR']
            df['atr_multiple'] = abs(df['total_change']) / df['ATR']  # 价格变化是ATR的多少倍
            df['normalized_change_by_atr'] = df['total_change'] / df['ATR']  # 标准化的趋势变化（以ATR为单位）
        else:
            # 如果没有ATR数据，创建虚拟列以保持特征维度一致
            df['atr_value'] = df['start_price'] * 0.01  # 用价格的1%作为虚拟ATR
            df['atr_multiple'] = abs(df['total_change']) / df['atr_value']
            df['normalized_change_by_atr'] = df['total_change'] / df['atr_value']
        
        # ATR倍数区间特征
        df['atr_multiple_category'] = pd.cut(df['atr_multiple'], bins=[0, 1, 2, 3, 5, float('inf')], 
                                           labels=['微幅(0-1倍)', '小幅(1-2倍)', '中幅(2-3倍)', '大幅(3-5倍)', '巨幅(>5倍)'])
        df = pd.get_dummies(df, columns=['atr_multiple_category'], prefix='atr_mult')

        # 3. 成交量特征（如果存在）
        if 'avg_volume' in df.columns:
            df['volume_ma_ratio'] = df['start_volume'] / df['avg_volume']  # 当前成交量与平均成交量比率
        elif 'start_volume' in df.columns:
            df['start_volume'] = df['start_volume']
        else:
            # 如果没有成交量数据，创建虚拟列以保持特征维度一致
            df['start_volume'] = 0
            df['volume_ma_ratio'] = 1.0

        # 4. 历史趋势特征（简单统计，可扩展）
        df['rolling_amplitude'] = df['total_change'].rolling(window=5).mean().fillna(0)
        
        # 新增：趋势反转相关特征
        df['trend_direction'] = (df['total_change'] > 0).astype(int)  # 当前趋势方向
        
        # 计算连续同向趋势的数量
        df['consecutive_same_trend'] = 0
        current_streak = 1
        for i in range(1, len(df)):
            if df['trend_direction'].iloc[i] == df['trend_direction'].iloc[i-1]:
                current_streak += 1
            else:
                current_streak = 1
            df['consecutive_same_trend'].iloc[i] = current_streak
        
        # 趋势持续时间的移动平均
        df['trend_duration_ma'] = df['duration_minutes'].rolling(window=5, min_periods=1).mean()
        
        # 趋势强度（幅度/持续时间）
        df['trend_strength'] = df['total_change'] / df['duration_minutes']
        
        # 新增：预测未来反转信号（基于下一个趋势方向）
        df['next_trend_direction'] = df['trend_direction'].shift(-1)  # 下一个趋势方向
        df['will_reverse'] = (df['trend_direction'] != df['next_trend_direction']).astype(int)  # 即将反转的信号
        
        # 价格偏离均线的程度（可能预示反转）
        df['price_deviation'] = (df['start_price'] - df['start_price'].rolling(window=10, min_periods=1).mean()) / df['start_price'].rolling(window=10, min_periods=1).std()
        
        # RSI指标
        df['rsi'] = self.calculate_rsi_simple(df['start_price'].values)
        
        # 布林带位置
        upper, middle, lower = self.calculate_bollinger_bands(df['start_price'])
        df['bb_position'] = (df['start_price'] - lower) / (upper - lower) if upper is not None and lower is not None else 0.5
        
        # 波动率特征
        df['volatility'] = df['start_price'].rolling(window=10).std() / df['start_price']
        
        # 新增：特定时段的反转特征
        # 计算小时特征
        df['hour'] = df['start_time'].dt.hour
        
        # 识别亚盘高点和低点（08:00-17:00）
        asian_session_mask = (df['hour'] >= 8) & (df['hour'] < 17)
        if asian_session_mask.any():
            asian_high = df.loc[asian_session_mask, 'start_price'].max()
            asian_low = df.loc[asian_session_mask, 'start_price'].min()
            
            # 标识当前是否为欧盘或美盘
            european_session_mask = (df['hour'] >= 18) & (df['hour'] < 22)
            us_session_mask = (df['hour'] >= 22) | (df['hour'] < 7)
            
            # 欧盘是否突破亚盘高点/低点
            df['euro_breaks_asian_high'] = 0
            df['euro_breaks_asian_low'] = 0
            if european_session_mask.any():
                df.loc[european_session_mask, 'euro_breaks_asian_high'] = (
                    df.loc[european_session_mask, 'start_price'] > asian_high
                ).astype(int)
                
                df.loc[european_session_mask, 'euro_breaks_asian_low'] = (
                    df.loc[european_session_mask, 'start_price'] < asian_low
                ).astype(int)
            
            # 美盘是否突破亚盘高点/低点
            df['us_breaks_asian_high'] = 0
            df['us_breaks_asian_low'] = 0
            if us_session_mask.any():
                df.loc[us_session_mask, 'us_breaks_asian_high'] = (
                    df.loc[us_session_mask, 'start_price'] > asian_high
                ).astype(int)
                
                df.loc[us_session_mask, 'us_breaks_asian_low'] = (
                    df.loc[us_session_mask, 'start_price'] < asian_low
                ).astype(int)
        else:
            # 如果没有亚盘数据，设置默认值
            df['euro_breaks_asian_high'] = 0
            df['euro_breaks_asian_low'] = 0
            df['us_breaks_asian_high'] = 0
            df['us_breaks_asian_low'] = 0
        
        # 新增：创新高新低特征
        df['new_high'] = 0
        df['new_low'] = 0
        for i in range(1, len(df)):
            # 检查是否创近期新高或新低（基于前面的数据）
            prev_prices = df['start_price'].iloc[:i]
            if len(prev_prices) > 0:
                if df['start_price'].iloc[i] > prev_prices.max():
                    df['new_high'].iloc[i] = 1
                elif df['start_price'].iloc[i] < prev_prices.min():
                    df['new_low'].iloc[i] = 1

        # 新增：真假突破特征
        # 计算支撑阻力位（基于滚动窗口的高低点）
        df['resistance'] = df['start_price'].rolling(window=20, center=False).max()  # 阻力位
        df['support'] = df['start_price'].rolling(window=20, center=False).min()    # 支撑位
        
        # 判断是否突破支撑阻力位
        df['breaks_resistance'] = (df['start_price'] > df['resistance'].shift(1)).astype(int)
        df['breaks_support'] = (df['start_price'] < df['support'].shift(1)).astype(int)
        
        # 突破有效性判断（突破后能否维持）
        df['break_validity'] = 0
        for i in range(21, len(df)):  # 从第21个数据点开始（因为需要20个点计算支撑阻力）
            if df['breaks_resistance'].iloc[i] == 1:
                # 如果向上突破阻力，看后面几个点是否能维持在阻力上方
                future_prices = df['start_price'].iloc[i:i+5] if i+5 < len(df) else df['start_price'].iloc[i:]
                if len(future_prices) > 0 and (future_prices > df['resistance'].iloc[i]).any():
                    df['break_validity'].iloc[i] = 1  # 有效突破
                else:
                    df['break_validity'].iloc[i] = -1  # 假突破
            elif df['breaks_support'].iloc[i] == 1:
                # 如果向下突破支撑，看后面几个点是否能维持在支撑下方
                future_prices = df['start_price'].iloc[i:i+5] if i+5 < len(df) else df['start_price'].iloc[i:]
                if len(future_prices) > 0 and (future_prices < df['support'].iloc[i]).any():
                    df['break_validity'].iloc[i] = 1  # 有效突破
                else:
                    df['break_validity'].iloc[i] = -1  # 假突破
        
        # 成交量确认突破（如果存在成交量数据）
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['high_volume_on_breakout'] = (
                (df['breaks_resistance'] | df['breaks_support']) & 
                (df['volume'] > df['volume_ma'] * 1.5)
            ).astype(int)  # 高成交量确认突破
        else:
            df['high_volume_on_breakout'] = 0
        
        # RSI确认突破（突破时RSI是否在合理区间）
        df['rsi_confirmation'] = 0
        for i in range(1, len(df)):
            if df['breaks_resistance'].iloc[i] == 1:  # 向上突破
                if df['rsi'].iloc[i] is not None and 50 < df['rsi'].iloc[i] < 70:
                    df['rsi_confirmation'].iloc[i] = 1  # RSI确认向上突破
            elif df['breaks_support'].iloc[i] == 1:  # 向下突破
                if df['rsi'].iloc[i] is not None and 30 < df['rsi'].iloc[i] < 50:
                    df['rsi_confirmation'].iloc[i] = 1  # RSI确认向下突破
        
        # 新增：回调识别特征
        # 计算价格回撤比例（用于识别回调）
        df['price_retrace_ratio'] = 0.0
        df['recent_high'] = df['start_price'].rolling(window=20, min_periods=1).max()
        df['recent_low'] = df['start_price'].rolling(window=20, min_periods=1).min()
        
        for i in range(1, len(df)):
            current_price = df['start_price'].iloc[i]
            recent_high = df['recent_high'].iloc[i-1]
            recent_low = df['recent_low'].iloc[i-1]
            
            # 计算从近期高点的回撤比例
            if recent_high != recent_low:
                if df['trend_direction'].iloc[i] == 1:  # 当前处于上升趋势
                    df['price_retrace_ratio'].iloc[i] = (recent_high - current_price) / (recent_high - recent_low)
                else:  # 当前处于下降趋势
                    df['price_retrace_ratio'].iloc[i] = (current_price - recent_low) / (recent_high - recent_low)
            
        # 识别回调模式 - 深度回调（可能预示趋势反转）
        df['deep_retrace'] = (df['price_retrace_ratio'] > 0.5).astype(int)  # 深度回调
        df['shallow_retrace'] = ((df['price_retrace_ratio'] > 0.2) & (df['price_retrace_ratio'] <= 0.5)).astype(int)  # 浅回调
        
        # 动量背离特征 - RSI与价格走势背离
        df['momentum_divergence'] = 0
        df['price_change'] = df['start_price'].diff()
        df['rsi_change'] = df['rsi'].diff()
        
        for i in range(2, len(df)):
            # 看涨背离：价格创新低但RSI未创新低
            if (df['start_price'].iloc[i] < df['start_price'].iloc[i-2] and 
                df['recent_low'].iloc[i] == df['start_price'].iloc[i] and  # 价格创新低
                df['rsi'].iloc[i] > df['rsi'].iloc[i-2]):  # 但RSI未创新低
                df['momentum_divergence'].iloc[i] = 1
            # 看跌背离：价格创新高但RSI未创新高
            elif (df['start_price'].iloc[i] > df['start_price'].iloc[i-2] and 
                  df['recent_high'].iloc[i] == df['start_price'].iloc[i] and  # 价格创新高
                  df['rsi'].iloc[i] < df['rsi'].iloc[i-2]):  # 但RSI未创新高
                df['momentum_divergence'].iloc[i] = -1
        
        # MACD相关特征（简化版）
        df['ema_fast'] = df['start_price'].ewm(span=12).mean()
        df['ema_slow'] = df['start_price'].ewm(span=26).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD柱状图变化率（用于识别动能变化）
        df['macd_hist_change'] = df['macd_histogram'].diff()
        
        # 价格与移动平均线的距离（用于识别回调后的重新测试）
        df['price_ma_distance'] = (df['start_price'] - df['start_price'].rolling(window=20).mean()) / df['start_price'].rolling(window=20).std()
        
        # 振荡器回调特征（基于RSI）
        df['rsi_oscillation'] = 0
        for i in range(2, len(df)):
            # RSI从超买/超卖区域返回
            if df['rsi'].iloc[i-1] is not None and df['rsi'].iloc[i-2] is not None:
                # 从超买区返回
                if df['rsi'].iloc[i-2] > 70 and df['rsi'].iloc[i-1] <= 70 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1]:
                    df['rsi_oscillation'].iloc[i] = 1
                # 从超卖区返回
                elif df['rsi'].iloc[i-2] < 30 and df['rsi'].iloc[i-1] >= 30 and df['rsi'].iloc[i] < df['rsi'].iloc[i-1]:
                    df['rsi_oscillation'].iloc[i] = -1

        # 5. 目标变量（实战核心：方向+时长+幅度+反转）
        # 涨跌方向
        self.target_trend = (df['total_change'] > 0).astype(int)
        # 趋势反转预测（修改为预测未来反转）
        self.target_reversal = df['will_reverse'].fillna(0).astype(int)  # 1表示即将反转，0表示延续
        # 趋势持续时长（AI自主预测的核心）
        self.target_duration = df['duration_minutes']
        # 价格变动幅度（绝对值，用于止盈）
        self.target_amplitude = abs(df['total_change'])

        # 6. 特征筛选（只保留数值特征用于训练）
        feature_cols = [
            'hour', 'weekday', 'session_asia', 'session_europe', 'session_us',
            'subsession_asia_open', 'subsession_asia_main',
            'subsession_europe_open', 'subsession_europe_main',
            'subsession_us_open', 'subsession_us_main',
            'is_monday', 'is_friday',
            'start_price', 'price_round', 'amplitude_ratio', 'rolling_amplitude',
            'consecutive_same_trend', 'trend_duration_ma', 'trend_strength',
            'price_deviation', 'rsi', 'bb_position', 'volatility',
            'euro_breaks_asian_high', 'euro_breaks_asian_low', 
            'us_breaks_asian_high', 'us_breaks_asian_low',
            'new_high', 'new_low', 'breaks_resistance', 'breaks_support', 
            'break_validity', 'high_volume_on_breakout', 'rsi_confirmation',
            'price_retrace_ratio', 'deep_retrace', 'shallow_retrace', 
            'momentum_divergence', 'macd_histogram', 'macd_hist_change',
            'price_ma_distance', 'rsi_oscillation',
            # ATR相关特征
            'atr_value', 'atr_multiple', 'normalized_change_by_atr',
            # ATR倍数类别特征（经过get_dummies处理后）
            'atr_mult_微幅(0-1倍)', 'atr_mult_小幅(1-2倍)', 'atr_mult_中幅(2-3倍)', 
            'atr_mult_大幅(3-5倍)', 'atr_mult_巨幅(>5倍)'
        ]
        
        # 添加成交量相关特征
        if 'start_volume' in df.columns:
            feature_cols.append('start_volume')
        if 'volume_ma_ratio' in df.columns:
            feature_cols.append('volume_ma_ratio')
        
        # 确保所有特征列存在
        existing_features = [col for col in feature_cols if col in df.columns]
        self.features = df[existing_features].fillna(0)

        # 标准化特征
        self.features = self.scaler.fit_transform(self.features)
        return True

    def train_multi_task_models(self):
        """训练多任务AI模型：同时预测方向、反转、时长、幅度"""

        # 拆分训练集/测试集（7:3）
        X_train, X_test, y_train_trend, y_test_trend = train_test_split(
            self.features, self.target_trend, test_size=0.3, random_state=42
        )
        
        # 新增：训练趋势反转预测模型
        # 检查反转标签的分布，如果所有标签都相同，则跳过反转模型训练
        unique_reversal_labels = np.unique(self.target_reversal)
        if len(unique_reversal_labels) > 1:
            X_train_rev, X_test_rev, y_train_rev, y_test_rev = train_test_split(
                self.features, self.target_reversal, test_size=0.3, random_state=42
            )
            
            # 训练趋势反转预测模型
            self.reversal_model = RandomForestClassifier(
                n_estimators=150, random_state=42, max_depth=12, min_samples_split=8
            )
            self.reversal_model.fit(X_train_rev, y_train_rev)
            reversal_acc = accuracy_score(y_test_rev, self.reversal_model.predict(X_test_rev))
            # print(f"🔄 趋势反转模型准确率：{reversal_acc:.4f}")
        else:
            # 如果反转标签都相同，创建一个常量预测器
            self.reversal_model = None

            # print(f"   所有反转标签值为: {unique_reversal_labels[0]}")

        X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(
            self.features, self.target_duration, test_size=0.3, random_state=42
        )
        X_train_amp, X_test_amp, y_train_amp, y_test_amp = train_test_split(
            self.features, self.target_amplitude, test_size=0.3, random_state=42
        )

        # 1. 训练涨跌方向模型（分类）
        self.trend_model = RandomForestClassifier(
            n_estimators=150, random_state=42, max_depth=12, min_samples_split=8
        )
        self.trend_model.fit(X_train, y_train_trend)
        trend_acc = accuracy_score(y_test_trend, self.trend_model.predict(X_test))
        print(f"🎯 模型准确率：{trend_acc:.4f}")

        # 2. 训练趋势时长模型（回归，AI自主判断持仓时长）
        self.duration_model = RandomForestRegressor(
            n_estimators=150, random_state=42, max_depth=10, min_samples_split=8
        )
        self.duration_model.fit(X_train_dur, y_train_dur)
        duration_mae = mean_absolute_error(y_test_dur, self.duration_model.predict(X_test_dur))
        print(f"🎯 趋势时长模型MAE：{duration_mae:.2f} 分钟 (越小越准)")

        # 3. 训练价格幅度模型（回归，AI自主判断止盈幅度）
        self.amplitude_model = RandomForestRegressor(
            n_estimators=150, random_state=42, max_depth=10, min_samples_split=8
        )
        self.amplitude_model.fit(X_train_amp, y_train_amp)
        amplitude_mae = mean_absolute_error(y_test_amp, self.amplitude_model.predict(X_test_amp))
        print(f"🎯 价格幅度模型MAE：{amplitude_mae:.2f} 美元 (越小越准)")

        # 保存模型（实战中可直接加载，无需重复训练）
        model_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(self.trend_model, os.path.join(self.model_dir, f"trend_model_{model_ts}.pkl"))
        
        # 保存反转模型（如果存在）
        if self.reversal_model is not None:
            joblib.dump(self.reversal_model, os.path.join(self.model_dir, f"reversal_model_{model_ts}.pkl"))
        else:
            # 创建一个简单的反转模型，始终预测相同的值
            # 保存一个表示反转模型不存在的标记
            with open(os.path.join(self.model_dir, f"reversal_model_{model_ts}.txt"), 'w') as f:
                f.write(f"Constant prediction: {unique_reversal_labels[0] if len(unique_reversal_labels) > 0 else 0}")
        
        joblib.dump(self.duration_model, os.path.join(self.model_dir, f"duration_model_{model_ts}.pkl"))
        joblib.dump(self.amplitude_model, os.path.join(self.model_dir, f"amplitude_model_{model_ts}.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, f"scaler_{model_ts}.pkl"))

        # print(f"\n💾 实战模型已保存到 {self.model_dir} 目录")
        return True

    def get_latest_m1_time(self):
        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败: {mt5.last_error()}")
            return datetime.now()  # 如果连接失败，返回当前时间
        
        # 检查交易品种
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ 品种 {symbol} 不可用")
            mt5.shutdown()
            return datetime.now()

        if not symbol_info.visible:
            print(f"✅ 启用品种 {symbol}...")
            if not mt5.symbol_select(symbol, True):
                print(f"❌ 启用品种失败")
                mt5.shutdown()
                return datetime.now()

        # 获取最近的M1数据
        # 只需要获取最新的1根K线
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        
        if rates is None or len(rates) == 0:
            print(f"❌ 未获取到最新的M1数据")
            mt5.shutdown()
            return datetime.now()
        
        # 获取最新K线的时间
        latest_time = pd.to_datetime(rates[0]['time'], unit='s')
        
        # print(f"✅ 最新M1数据时间: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 断开MT5连接
        mt5.shutdown()
        
        return latest_time

    def generate_trading_signals(self):
        """生成可直接下单的实战交易信号"""
        print(
            f"{'开仓时间':<15} {'持仓时长':<10} {'方向':<8} {'反转概率':<8} {'止盈幅度':<8} {'止损幅度':<10} {'置信度':<8} {'风险收益比':<12} {'信号有效性':<8} {'交易时段':<8}")

        # 获取最新的M1数据时间
        latest_m1_time = self.get_latest_m1_time()
        
        # 取最后一条数据的特征，预测下一个交易信号
        last_feature = self.features[-1].reshape(1, -1)
        
        # 确保raw_data存在且不为空
        if len(self.raw_data) == 0:
            print("❌ 没有可用的历史数据，无法生成交易信号")
            return None
        
        # 获取最后一行数据
        raw_last_data = self.raw_data.iloc[-1]
        
        # 确保raw_last_data存在
        if raw_last_data is None:
            print("❌ 最后一行数据为空，无法生成交易信号")
            return None
        
        # 创建last_raw_data变量用于后续使用
        last_raw_data = raw_last_data

        # 1. AI预测核心参数（自主判断）
        trend_pred = self.trend_model.predict(last_feature)[0]  # 0=跌，1=涨
        trend_confidence = self.trend_model.predict_proba(last_feature)[0][trend_pred] * 100  # 置信度
        
        # ATR相关分析，用于调整信号置信度
        if 'atr_value' in raw_last_data and 'atr_multiple' in raw_last_data:
            current_atr = raw_last_data['atr_value']
            current_atr_multiple = raw_last_data['atr_multiple']
            
            # 根据ATR倍数调整信号置信度
            if current_atr_multiple > 3:
                # ATR倍数过高，可能面临回调风险，降低置信度
                trend_confidence *= 0.9
            elif current_atr_multiple < 0.5:
                # ATR倍数过低，可能缺乏足够动能，适度降低置信度
                trend_confidence *= 0.95
        
        # 利用时段反转特性辅助交易决策
        if hasattr(self, 'session_reversal_characteristics'):
            # 获取当前时段
            current_hour = raw_last_data['start_time'].hour
            if 0 <= current_hour <= 8:
                current_session = '亚盘'
            elif 9 <= current_hour <= 17:
                current_session = '欧盘'
            else:
                current_session = '美盘'
            
            # 获取当前时段的反转特性
            if current_session in self.session_reversal_characteristics:
                session_info = self.session_reversal_characteristics[current_session]
                reversal_count = session_info['reversal_count']
                avg_prev_atr_mult = session_info['avg_prev_atr_multiple']
                success_rate = session_info['success_rate']
                avg_reversal_strength = session_info['avg_reversal_strength']
                
                # 根据时段反转特性调整反转概率
                if reversal_count > 0:  # 如果该时段有反转记录
                    # 如果当前ATR倍数高于该时段平均反转前的ATR倍数，可能更容易出现反转
                    if 'atr_multiple' in raw_last_data and raw_last_data['atr_multiple'] > avg_prev_atr_mult * 1.2:
                        reversal_prob += 15  # 增加反转概率
                    
                    # 如果该时段反转成功率较低，也增加反转概率
                    if success_rate < 0.4:
                        reversal_prob += 10
                    
                    # 如果当前时段平均反转强度较高，也适当调整
                    if avg_reversal_strength > 2.0:
                        reversal_prob += 8
        
        # 利用震荡模式信息辅助交易决策
        if hasattr(self, 'session_oscillation_characteristics'):
            current_hour = raw_last_data['start_time'].hour
            if 0 <= current_hour <= 8:
                current_session = '亚盘'
            elif 9 <= current_hour <= 17:
                current_session = '欧盘'
            else:
                current_session = '美盘'
            
            # 检查当前时段的震荡特性
            if current_session in self.session_oscillation_characteristics:
                osc_info = self.session_oscillation_characteristics[current_session]
                osc_count = osc_info['count']
                avg_strength = osc_info['avg_strength']
                avg_duration = osc_info['avg_duration']
                
                # 如果当前时段震荡频繁，增加反转概率
                if osc_count > 5:  # 震荡模式较多
                    reversal_prob += 12
                
                # 如果震荡强度较高，也增加反转概率
                if avg_strength > 2.5:
                    reversal_prob += 10
                
                # 如果震荡持续时间较长，也适当调整
                if avg_duration > 10:
                    reversal_prob += 5
        
        # 根据震荡模式调整趋势预测置信度
        if hasattr(self, 'oscillation_patterns') and len(self.oscillation_patterns) > 0:
            # 检查最近是否有强烈的震荡模式
            recent_oscillations = [p for p in self.oscillation_patterns 
                                 if (raw_last_data['start_time'] - p['end_time']).total_seconds() / 60 < 60]  # 1小时内
            if len(recent_oscillations) > 0:
                avg_recent_strength = np.mean([p['strength'] for p in recent_oscillations])
                if avg_recent_strength > 3.0:  # 强烈震荡
                    # 在强烈震荡后，趋势信号的可靠性可能降低
                    trend_confidence *= 0.85
        
        # 利用价格波动接近10的倍数的模式辅助交易决策
        if hasattr(self, 'round_number_patterns'):
            # 检查最近的圆数模式
            recent_round_patterns = [p for p in self.round_number_patterns 
                                  if (raw_last_data['start_time'] - p['end_time']).total_seconds() / 3600 < 4]  # 4小时内
            
            if len(recent_round_patterns) > 0:
                # 如果最近有波动接近10的倍数，这可能意味着市场在遵循某种规律
                # 增加对趋势延续或反转的分析
                avg_recent_round_change = np.mean([p['actual_change'] for p in recent_round_patterns])
                
                # 如果最近的波动幅度接近10的倍数，可能表明市场有规律性
                if avg_recent_round_change > 0:
                    # 检查当前价格与最近的10的倍数的距离
                    current_price = raw_last_data['start_price']
                    rounded_price = round(current_price / 10) * 10
                    price_to_round = abs(current_price - rounded_price)
                    
                    # 如果价格接近10的倍数，可能有支撑或阻力
                    if price_to_round < 0.5:  # 接近10的倍数
                        # 增加反转概率，因为价格可能在10的倍数处遇到阻力或支撑
                        reversal_prob += 12
                    
                    # 如果当前ATR倍数与圆数模式相符，可以提高置信度
                    if 'atr_multiple' in raw_last_data:
                        current_atr_mult = raw_last_data['atr_multiple']
                        if 1.0 <= current_atr_mult <= 3.0:  # 中等ATR倍数，可能符合圆数模式
                            # 这种情况下，趋势可能更可靠
                            trend_confidence *= 1.05  # 小幅提升置信度
        
        # 新增：预测趋势反转概率（处理模型可能为None的情况）
        if self.reversal_model is not None:
            reversal_prob = self.reversal_model.predict_proba(last_feature)[0][1] * 100  # 反转概率
            reversal_pred = self.reversal_model.predict(last_feature)[0]  # 1=反转，0=延续
            
            # 增强反转检测逻辑：结合RSI、布林带等技术指标
            rsi_value = raw_last_data.get('rsi', None)
            bb_position = raw_last_data.get('bb_position', None)
            
            # 如果RSI超买或超卖，增加反转概率
            if rsi_value is not None:
                if rsi_value > 70:  # 超买区域
                    reversal_prob += 15  # 增加反转概率
                    reversal_prob = min(reversal_prob, 100)  # 限制最大值
                elif rsi_value < 30:  # 超卖区域
                    reversal_prob += 15  # 增加反转概率
                    reversal_prob = min(reversal_prob, 100)  # 限制最大值
            
            # 如果价格接近布林带上轨或下轨，增加反转概率
            if bb_position is not None:
                if bb_position > 0.8:  # 接近上轨
                    reversal_prob += 10  # 增加反转概率
                    reversal_prob = min(reversal_prob, 100)  # 限制最大值
                elif bb_position < 0.2:  # 接近下轨
                    reversal_prob += 10  # 增加反转概率
                    reversal_prob = min(reversal_prob, 100)  # 限制最大值
            
            # 新增：增强回调识别逻辑
            # 检查深度回调特征
            deep_retrace = raw_last_data.get('deep_retrace', 0)
            if deep_retrace == 1:
                reversal_prob += 20  # 深度回调增加反转概率
                reversal_prob = min(reversal_prob, 100)
            
            # 检查动量背离
            momentum_div = raw_last_data.get('momentum_divergence', 0)
            if momentum_div != 0:  # 存在动量背离
                reversal_prob += 25  # 动量背离显著增加反转概率
                reversal_prob = min(reversal_prob, 100)
            
            # 检查MACD柱状图变化
            macd_hist_change = raw_last_data.get('macd_hist_change', 0)
            current_trend = raw_last_data.get('trend_direction', 0)
            if current_trend == 1 and macd_hist_change < 0:  # 上升趋势中MACD柱状图下降
                reversal_prob += 10
                reversal_prob = min(reversal_prob, 100)
            elif current_trend == 0 and macd_hist_change > 0:  # 下降趋势中MACD柱状图上升
                reversal_prob += 10
                reversal_prob = min(reversal_prob, 100)
            
            # 检查RSI振荡特征
            rsi_osc = raw_last_data.get('rsi_oscillation', 0)
            if rsi_osc != 0:  # RSI从极端区域返回
                reversal_prob += 12
                reversal_prob = min(reversal_prob, 100)
            
            # 检查价格与移动平均线距离
            ma_dist = raw_last_data.get('price_ma_distance', 0)
            if (current_trend == 1 and ma_dist > 2) or (current_trend == 0 and ma_dist < -2):  # 远离移动平均线
                reversal_prob += 15  # 远离均线可能引发回调
                reversal_prob = min(reversal_prob, 100)
        else:
            # 如果反转模型不存在，使用默认值
            reversal_prob = 0  # 默认反转概率为0
            reversal_pred = 0  # 默认不反转
            
        duration_pred = self.duration_model.predict(last_feature)[0]  # 持仓时长（AI自主）
        amplitude_pred = self.amplitude_model.predict(last_feature)[0]  # 止盈幅度（AI自主）

        # 2. 风控过滤（不符合条件的信号直接丢弃）
        duration_pred = np.clip(duration_pred, MIN_TREND_DURATION, MAX_TREND_DURATION)  # 限制时长
        # 调整止盈止损幅度，使其更适合实际交易
        # 由于AI预测的幅度较小，我们将其放大以适应实际市场波动
        adjusted_amplitude_pred = amplitude_pred * 1.5  # 将预测幅度放大1.5倍
        stop_loss_amplitude = adjusted_amplitude_pred * STOP_LOSS_RATIO  # 止损幅度
        risk_reward = adjusted_amplitude_pred / stop_loss_amplitude  # 风险收益比

        # 3. 计算具体交易时间 - 使用最新的M1时间
        # 使用最新M1数据时间的下一分钟作为开仓时间
        open_time = latest_m1_time + timedelta(minutes=1)  # 开仓时间（下一分钟）
        close_time = open_time + timedelta(minutes=duration_pred)  # 平仓时间（AI自主时长）

        # 确定交易时段（根据XAUUSD时间，比北京时间晚6小时）
        # 将北京时间转换为XAUUSD时间（北京时间-6小时=XAUUSD时间）
        hour = open_time.hour
        # 确定交易时段 (按主要交易中心时间划分，UTC+2时区)
        if 1 <= hour < 2:  # 亚盘开盘时间 (07:00-08:00 UTC+2)
            session = "亚盘开盘"
        elif 2 <= hour < 11:  # 亚盘 (08:00-17:00 UTC+2)
            session = "亚盘"
        elif 11 <= hour < 12:  # 欧盘开盘时间 (17:00-18:00 UTC+2)
            session = "欧盘开盘"
        elif 12 <= hour < 16:  # 欧盘 (18:00-22:00 UTC+2)
            session = "欧盘"
        else:  # 美盘 (22:00-07:00 UTC+2)，包括隔夜时段
            if hour >= 16:
                session = "美盘"
            else:
                session = "美盘(隔夜)"
        

        # 4. 信号有效性判断
        signal_valid = False
        if trend_confidence >= CONFIDENCE_THRESHOLD and risk_reward >= RISK_REWARD_RATIO:
            signal_valid = True

        # 5. 格式化输出
        trend_str = "做多" if trend_pred == 1 else "做空"
        valid_str = "✅ 有效" if signal_valid else "❌ 无效"
        open_time_str = open_time.strftime("%Y-%m-%d %H:%M") if pd.notna(open_time) else "未知"

        # 保存交易信号
        self.trading_signals.append({
            "开仓时间": open_time_str,
            "平仓时间": close_time.strftime("%Y-%m-%d %H:%M") if pd.notna(close_time) else "未知",
            "持仓时长(分钟)": round(duration_pred, 0),
            "交易方向": trend_str,
            "趋势反转概率(%)": round(reversal_prob, 1),
            # 根据用户偏好，不显示入场价格信息
            # "开仓价格": round(last_raw_data['start_price'], 2),
            "止盈幅度(美元)": round(adjusted_amplitude_pred, 2),
            "止损幅度(美元)": round(stop_loss_amplitude, 2),
            "置信度(%)": round(trend_confidence, 1),
            "风险收益比": round(risk_reward, 2),
            "信号有效性": valid_str,
            "交易时段": session
        })

        # 打印单条信号（实战中可输出多条）
        print(
            f"{open_time_str:<20} {round(duration_pred, 0):<10} {trend_str:<8} {round(reversal_prob, 1):<12} {round(adjusted_amplitude_pred, 2):<12} {round(stop_loss_amplitude, 2):<12} {round(trend_confidence, 1):<10} {round(risk_reward, 2):<12} {valid_str:<10} {session:<10}")


        # 获取最后一条数据的特征值，分析关键突破特征
        raw_last_data = self.raw_data.iloc[-1]
        current_price = raw_last_data['start_price']
        
        # ATR相关分析
        if 'atr_value' in raw_last_data and 'atr_multiple' in raw_last_data:
            current_atr = raw_last_data['atr_value']
            current_atr_multiple = raw_last_data['atr_multiple']
            
            # 根据ATR倍数调整信号置信度
            if current_atr_multiple > 3:
                # ATR倍数过高，可能面临回调风险，降低置信度
                trend_confidence *= 0.9
            elif current_atr_multiple < 0.5:
                # ATR倍数过低，可能缺乏足够动能，适度降低置信度
                trend_confidence *= 0.95
        
        # 分析相似的ATR倍数模式
        if 'atr_multiple' in self.raw_data.columns:
            current_atr_mult = raw_last_data['atr_multiple']
            
            # 寻找具有相似ATR倍数的历史模式
            tolerance = 0.5  # ATR倍数容差
            similar_patterns = self.raw_data[
                (abs(self.raw_data['atr_multiple'] - current_atr_mult) <= tolerance) & 
                (self.raw_data.index != len(self.raw_data) - 1)  # 排除当前记录
            ]
            
            if len(similar_patterns) > 0:
                # 使用相似模式信息调整反转概率
                recent_similar = similar_patterns.tail(5)  # 获取最近的相似模式
                
                # 分析这些相似模式后续的表现
                future_directions = []
                for idx, row in recent_similar.iterrows():
                    current_idx = self.raw_data.index.get_loc(idx)
                    if current_idx + 1 < len(self.raw_data):
                        next_change = self.raw_data.iloc[current_idx + 1]['total_change']
                        future_directions.append(1 if next_change > 0 else 0)
                
                if future_directions:
                    up_count = sum(future_directions)
                    total_count = len(future_directions)
                    up_ratio = up_count / total_count if total_count > 0 else 0
                    
                    # 根据历史相似模式表现调整反转概率
                    if (trend_pred == 1 and up_ratio < 0.4) or (trend_pred == 0 and up_ratio > 0.6):
                        # 如果当前预测方向与历史相似模式相反，则增加反转概率
                        reversal_prob += 10
        
        # 计算支撑阻力位
        recent_prices = self.raw_data['start_price'].tail(20)
        resistance = recent_prices.max()
        support = recent_prices.min()
        
        # 判断是否接近支撑或阻力
        if abs(current_price - resistance) < (resistance - support) * 0.001:  # 接近阻力
            print(f"   📈 当前价格接近阻力位 {resistance:.2f}，可能存在反转压力")
        elif abs(current_price - support) < (resistance - support) * 0.001:  # 接近支撑
            print(f"   📉 当前价格接近支撑位 {support:.2f}，可能存在反弹支撑")
        
        # 分析是否突破亚盘高点低点（适用于欧盘和美盘）
        # 从原始数据中提取时间信息
        raw_data_with_time = self.raw_data.copy()
        raw_data_with_time['start_time'] = pd.to_datetime(raw_data_with_time['start_time'])
        raw_data_with_time['hour'] = raw_data_with_time['start_time'].dt.hour
        
        if session in ["欧盘", "美盘"]:
            # 计算亚盘时段的高点低点（假设最近亚盘数据）
            asian_data = raw_data_with_time[(raw_data_with_time['hour'] >= 8) & (raw_data_with_time['hour'] < 17)]
            if len(asian_data) > 0:
                asian_high = asian_data['start_price'].max()
                asian_low = asian_data['start_price'].min()
                
                if current_price > asian_high * 0.999:  # 接近突破亚盘高点
                    # 使用AI模型预测突破有效性
                    # 基于当前特征预测这是否是假突破
                    break_validity_prediction = 0
                    if hasattr(self, 'features') and len(self.features) > 0:
                        last_feature = self.features[-1].reshape(1, -1)
                        # 如果有训练好的突破有效性预测模型，这里可以加入预测逻辑
                        # 简化预测：基于RSI和价格位置
                        if 'rsi' in self.raw_data.columns and pd.notna(raw_last_data.get('rsi')):
                            rsi_value = raw_last_data['rsi']
                            if rsi_value > 70:  # 超买区域，可能是假突破
                                break_validity_prediction = -1  # 预测为假突破
                                print(f"   🚀 AI预测突破亚盘高点 {asian_high:.2f} - 疑似假突破 (RSI={rsi_value:.1f}，超买状态)")
                            elif rsi_value > 50:  # 确认突破
                                print(f"   🚀 AI确认突破亚盘高点 {asian_high:.2f} - 有效突破 (RSI={rsi_value:.1f})")
                            else:
                                print(f"   🚀 当前价格接近突破亚盘高点 {asian_high:.2f}，RSI={rsi_value:.1f}，需谨慎判断")
                        else:
                            print(f"   🚀 当前价格接近突破亚盘高点 {asian_high:.2f}，但缺少RSI指标确认")
                elif current_price < asian_low * 1.001:  # 接近突破亚盘低点
                    # 同样的逻辑应用于低点突破
                    if 'rsi' in self.raw_data.columns and pd.notna(raw_last_data.get('rsi')):
                        rsi_value = raw_last_data['rsi']
                        if rsi_value < 30:  # 超卖区域，可能是假突破
                            print(f"   📌 AI预测突破亚盘低点 {asian_low:.2f} - 疑似假突破 (RSI={rsi_value:.1f}，超卖状态)")
                        elif rsi_value < 50:  # 确认向下突破
                            print(f"   📌 AI确认突破亚盘低点 {asian_low:.2f} - 有效突破 (RSI={rsi_value:.1f})")
                        else:
                            print(f"   📌 当前价格接近突破亚盘低点 {asian_low:.2f}，RSI={rsi_value:.1f}，需谨慎判断")
                    else:
                        print(f"   📌 当前价格接近突破亚盘低点 {asian_low:.2f}，但缺少RSI指标确认")
        
        # 显示RSI和趋势强度
        if 'rsi' in self.raw_data.columns and pd.notna(raw_last_data.get('rsi')):
            rsi_value = raw_last_data['rsi']
            if rsi_value > 70:
                print(f"   ⚠️  RSI值为 {rsi_value:.2f}，市场可能超买")
            elif rsi_value < 30:
                print(f"   ⚠️  RSI值为 {rsi_value:.2f}，市场可能超卖")
        
        # 分析深度回调
        deep_retrace = raw_last_data.get('deep_retrace', 0)
        if deep_retrace == 1:
            print(f"   🔄 检测到深度回调模式，趋势反转可能性较高")
        
        # 分析动量背离
        momentum_div = raw_last_data.get('momentum_divergence', 0)
        if momentum_div == 1:
            print(f"   📊 检测到看涨动量背离，可能预示趋势底部")
        elif momentum_div == -1:
            print(f"   📊 检测到看跌动量背离，可能预示趋势顶部")
        
        # 分析MACD柱状图变化
        macd_hist_change = raw_last_data.get('macd_hist_change', 0)
        current_trend = raw_last_data.get('trend_direction', 0)
        if current_trend == 1 and macd_hist_change < 0:
            print(f"   📈 上升趋势中MACD柱状图收缩，上升动能减弱")
        elif current_trend == 0 and macd_hist_change > 0:
            print(f"   📉 下降趋势中MACD柱状图扩张，下降动能增强")
        
        # 分析RSI振荡
        rsi_osc = raw_last_data.get('rsi_oscillation', 0)
        if rsi_osc == 1:
            print(f"   📈 RSI从超卖区域回升，可能预示反弹")
        elif rsi_osc == -1:
            print(f"   📉 RSI从超买区域回落，可能预示回调")
        
        # 分析价格与移动平均线距离
        ma_dist = raw_last_data.get('price_ma_distance', 0)
        if abs(ma_dist) > 2:
            if ma_dist > 0:
                print(f"   📈 价格远离移动平均线上方，存在回调压力")
            else:
                print(f"   📉 价格远离移动平均线下方，存在反弹动力")
        
        # 市场状态总结
        market_state = ""
        if trend_pred == 1:
            market_state = "📈 顺势做多"
        else:
            market_state = "📉 顺势做空"
        
        # 检查是否存在来回波动模式（震荡行情）
        if hasattr(self, 'significant_opportunities') and len(self.significant_opportunities) > 5:
            # 分析最近的趋势变化频率
            recent_opps = self.significant_opportunities[-5:]  # 最近5个机会
            direction_changes = 0
            prev_direction = None
            
            for opp in recent_opps:
                current_direction = 1 if opp['total_change'] > 0 else 0
                if prev_direction is not None and current_direction != prev_direction:
                    direction_changes += 1
                prev_direction = current_direction
            
            if direction_changes >= 3:  # 在最近5个机会中有3次或以上的方向变化
                market_state += " (震荡行情)"
                # 在震荡行情下，增加反转概率
                reversal_prob += 15
            
        # 检查是否接近10的倍数价格水平
        if hasattr(self, 'round_number_moves'):
            current_price = raw_last_data['start_price']
            rounded_price = round(current_price / 10) * 10
            price_to_round = abs(current_price - rounded_price)
            
            if price_to_round < 0.5:  # 接近10的倍数
                market_state += " (接近整数位)"
                # 接近10的倍数时，可能有更强的支撑或阻力
                reversal_prob += 10
            
        # 检查最近的价格变动是否接近10的倍数
        if hasattr(self, 'significant_opportunities') and len(self.significant_opportunities) > 0:
            latest_opps = self.significant_opportunities[-3:]  # 最近3个机会
            for opp in latest_opps:
                total_change = abs(opp['total_change'])
                if total_change > 2:  # 有意义的变动
                    rounded_change = round(total_change / 10) * 10
                    difference = abs(total_change - rounded_change)
                    if difference <= 1.5:  # 变动接近10的倍数
                        market_state += " (符合圆数模式)"
                        # 当最近的变动符合圆数模式时，可能继续遵循此模式
                        break
            
        if reversal_prob > 70:
            market_state += " (警惕反转)"
        elif reversal_prob < 30:
            market_state += " (趋势强劲)"
        else:
            market_state += " (趋势不确定)"


        # 实战下单建议
        if signal_valid:
            print(f"\n📝 实战下单建议：开仓时间：{open_time_str} | 交易时段：{session} | 交易方向：{trend_str}黄金M1 | 开仓价格：{round(last_raw_data['start_price'], 2)}美元 | 止盈设置：{round(last_raw_data['start_price'] + (amplitude_pred if trend_pred == 1 else -amplitude_pred), 2)}美元 | 止损设置：{round(last_raw_data['start_price'] - (stop_loss_amplitude if trend_pred == 1 else -stop_loss_amplitude), 2)}美元 | 平仓时间：{close_time.strftime('%Y-%m-%d %H:%M')}（或达到止盈/止损立即平仓）")
            
            if reversal_pred == 1 and reversal_prob > 70:
                print(f"⚠️ 特别提醒：反转概率{reversal_prob}%，注意市场变化！")

        # 返回最新生成的信号
        return self.trading_signals[-1] if self.trading_signals else None

    def analyze_session_reversal_characteristics(self, data):
        """分析各时段反转点特性，用于辅助交易决策"""
        if 'ATR' not in data.columns or len(data) == 0:
            return

        # 为数据添加时段标识
        data_with_sessions = data.copy()
        data_with_sessions['hour'] = data_with_sessions['timestamp'].dt.hour
        data_with_sessions['minute'] = data_with_sessions['timestamp'].dt.minute
        data_with_sessions['datetime'] = data_with_sessions['timestamp']
        data_with_sessions['date'] = data_with_sessions['timestamp'].dt.date

        # 定义交易时段
        def get_session(hour):
            if 0 <= hour <= 8:  # 亚盘
                return '亚盘'
            elif 9 <= hour <= 17:  # 欧盘
                return '欧盘'
            else:  # 美盘
                return '美盘'

        data_with_sessions['session'] = data_with_sessions['hour'].apply(get_session)

        # 计算价格方向变化（识别潜在反转点）
        data_with_sessions['price_change'] = data_with_sessions['close'] - data_with_sessions['open']
        data_with_sessions['abs_price_change'] = abs(data_with_sessions['price_change'])
        data_with_sessions['direction'] = np.where(data_with_sessions['price_change'] > 0, 1, -1)
        data_with_sessions['direction_change'] = data_with_sessions['direction'].diff()

        # 识别反转点（方向发生变化的位置）
        reversal_points = data_with_sessions[
            (data_with_sessions['direction_change'] != 0) & 
            (data_with_sessions['ATR'] > 0)
        ]

        # 按时段统计反转特征
        session_reversal_stats = {}
        for session in ['亚盘', '欧盘', '美盘']:
            session_data = data_with_sessions[data_with_sessions['session'] == session]
            session_reversals = reversal_points[reversal_points['session'] == session]
            
            if len(session_reversals) > 0:
                # 计算反转前的价格波动强度（以ATR为单位）
                prev_atr_multiples = []
                for rev_idx in session_reversals.index:
                    prev_idx = rev_idx - 1
                    if prev_idx in data_with_sessions.index:
                        prev_row = data_with_sessions.loc[prev_idx]
                        if 'ATR' in prev_row and prev_row['ATR'] > 0:
                            atr_mult = prev_row['abs_price_change'] / prev_row['ATR']
                            prev_atr_multiples.append(atr_mult)
                
                avg_prev_atr_mult = np.mean(prev_atr_multiples) if prev_atr_multiples else 0
                
                # 计算反转后趋势的持续性
                post_trend_continuation = []
                for rev_idx in session_reversals.index:
                    next_idx = rev_idx + 1
                    if next_idx in data_with_sessions.index:
                        current_direction = data_with_sessions.loc[rev_idx, 'direction']
                        next_direction = data_with_sessions.loc[next_idx, 'direction']
                        # 如果下一个K线方向与反转方向相同，则认为反转成功延续
                        if current_direction == next_direction:
                            post_trend_continuation.append(1)
                        else:
                            post_trend_continuation.append(0)
                
                success_rate = np.mean(post_trend_continuation) if post_trend_continuation else 0
                
                session_reversal_stats[session] = {
                    'reversal_count': len(session_reversals),
                    'avg_prev_atr_multiple': avg_prev_atr_mult,
                    'success_rate': success_rate,
                    'avg_reversal_strength': session_reversals['atr_multiple'].mean() if len(session_reversals) > 0 and 'atr_multiple' in session_reversals.columns else 0,
                    'timestamps': session_reversals['datetime'].tolist()
                }
            else:
                session_reversal_stats[session] = {
                    'reversal_count': 0,
                    'avg_prev_atr_multiple': 0,
                    'success_rate': 0,
                    'avg_reversal_strength': 0,
                    'timestamps': []
                }

        # 将反转特征保存到实例变量，供后续分析使用
        self.session_reversal_characteristics = session_reversal_stats

    def analyze_oscillation_patterns(self, data):
        """分析来回波动模式（震荡行情），用于辅助交易决策"""
        if 'ATR' not in data.columns or len(data) < 10:
            return
        
        # 计算价格方向变化
        data_copy = data.copy()
        data_copy['price_change'] = data_copy['close'] - data_copy['open']
        data_copy['direction'] = np.where(data_copy['price_change'] > 0, 1, -1)
        data_copy['abs_change'] = abs(data_copy['price_change'])
        
        # 识别方向变化点（潜在的转折点）
        data_copy['direction_changed'] = data_copy['direction'].diff() != 0
        
        # 查找连续的来回波动模式
        oscillation_patterns = []
        consecutive_changes = 0
        start_idx = None
        
        for idx in data_copy.index:
            if data_copy.loc[idx, 'direction_changed']:
                if start_idx is None:
                    start_idx = idx
                    consecutive_changes = 1
                else:
                    consecutive_changes += 1
                    
                    # 检查是否形成了来回波动模式
                    if consecutive_changes >= 3:  # 至少3次方向变化
                        pattern_data = data_copy.loc[start_idx:idx]
                        if len(pattern_data) > 0:
                            avg_atr = pattern_data['ATR'].mean() if 'ATR' in pattern_data.columns else 0
                            total_range = pattern_data['high'].max() - pattern_data['low'].min()
                            pattern_duration = len(pattern_data)
                            
                            oscillation_patterns.append({
                                'start_time': data_copy.loc[start_idx, 'timestamp'],
                                'end_time': data_copy.loc[idx, 'timestamp'],
                                'start_idx': start_idx,
                                'end_idx': idx,
                                'changes_count': consecutive_changes,
                                'avg_atr': avg_atr,
                                'total_range': total_range,
                                'duration': pattern_duration,
                                'strength': total_range / avg_atr if avg_atr > 0 else 0  # 波动强度
                            })
                    
                    # 重置计数，但保留前一个点作为新模式的起点
                    start_idx = data_copy.index[max(0, data_copy.index.get_loc(idx) - 1)]
                    consecutive_changes = 1
            else:
                # 如果没有方向变化，重置计数
                start_idx = None
                consecutive_changes = 0
        
        # 保存震荡模式信息
        self.oscillation_patterns = oscillation_patterns
        
        # 按时段统计震荡模式
        if len(oscillation_patterns) > 0:
            data_copy['hour'] = data_copy['timestamp'].dt.hour
            def get_session(hour):
                if 0 <= hour <= 8:
                    return '亚盘'
                elif 9 <= hour <= 17:
                    return '欧盘'
                else:
                    return '美盘'
            
            data_copy['session'] = data_copy['hour'].apply(get_session)
            
            session_oscillations = {}
            for session in ['亚盘', '欧盘', '美盘']:
                session_patterns = [p for p in oscillation_patterns 
                                  if data_copy.loc[p['start_idx'], 'session'] == session]
                session_oscillations[session] = {
                    'count': len(session_patterns),
                    'avg_strength': np.mean([p['strength'] for p in session_patterns]) if session_patterns else 0,
                    'avg_duration': np.mean([p['duration'] for p in session_patterns]) if session_patterns else 0
                }
            
            self.session_oscillation_characteristics = session_oscillations

    def analyze_round_number_patterns(self, data):
        """分析价格波动接近10的倍数的模式，用于辅助交易决策"""
        if len(data) < 2:
            return
        
        # 计算价格变化
        data_copy = data.copy()
        data_copy['price_change'] = data_copy['close'] - data_copy['open']
        data_copy['abs_price_change'] = abs(data_copy['price_change'])
        
        # 识别接近10的倍数的价格变动
        round_number_moves = []
        for idx in data_copy.index:
            change = data_copy.loc[idx, 'abs_price_change']
            if change > 2:  # 只考虑有意义的变动
                # 检查是否接近10的倍数（如5、10、15、20、25、30等）
                rounded_to_10 = round(change / 10) * 10
                difference = abs(change - rounded_to_10)
                ratio = difference / change if change != 0 else float('inf')
                
                # 如果变动接近10的倍数（差异在10%以内）
                if ratio <= 0.1 or difference <= 1.5:
                    round_number_moves.append({
                        'timestamp': data_copy.loc[idx, 'timestamp'],
                        'actual_change': change,
                        'rounded_change': rounded_to_10,
                        'difference': difference,
                        'price_direction': 1 if data_copy.loc[idx, 'price_change'] > 0 else -1,
                        'high': data_copy.loc[idx, 'high'],
                        'low': data_copy.loc[idx, 'low'],
                        'close': data_copy.loc[idx, 'close']
                    })
        
        # 保存圆数模式信息
        self.round_number_moves = round_number_moves
        
        # 按时段统计圆数模式
        if len(round_number_moves) > 0:
            data_copy['hour'] = data_copy['timestamp'].dt.hour
            def get_session(hour):
                if 0 <= hour <= 8:
                    return '亚盘'
                elif 9 <= hour <= 17:
                    return '欧盘'
                else:
                    return '美盘'
            
            data_copy['session'] = data_copy['hour'].apply(get_session)
            
            session_round_moves = {}
            for session in ['亚盘', '欧盘', '美盘']:
                session_moves = [move for move in round_number_moves 
                               if data_copy[data_copy['timestamp'] == move['timestamp']]['session'].iloc[0] == session if len(data_copy[data_copy['timestamp'] == move['timestamp']]) > 0]
                # 修正上面的逻辑，使用更直接的方法
                session_moves = []
                for move in round_number_moves:
                    move_hour = move['timestamp'].hour
                    if (session == '亚盘' and 0 <= move_hour <= 8) or \
                       (session == '欧盘' and 9 <= move_hour <= 17) or \
                       (session == '美盘' and (move_hour >= 18 or move_hour <= 5)):
                        session_moves.append(move)
                
                session_round_moves[session] = {
                    'count': len(session_moves),
                    'avg_change': np.mean([m['actual_change'] for m in session_moves]) if session_moves else 0,
                    'common_round_numbers': list(set([m['rounded_change'] for m in session_moves])),
                    'avg_difference': np.mean([m['difference'] for m in session_moves]) if session_moves else 0
                }
            
            self.session_round_number_moves_characteristics = session_round_moves

    def run_full_analysis_and_training(self, days_back=60):
        """运行完整的分析和训练流程"""
        # 第一步：获取M1数据
        data = self.fetch_m1_data_for_period(days_back)
        if data is None:
            print(f"❌ 数据获取失败")
            return None

        # 第二步：分析趋势
        self.analyze_trends(data)

        # 第三步：加载并清洗数据（使用刚刚生成的CSV）
        if not self.load_and_clean_data():
            print(f"❌ 数据加载失败")
            return None

        # 第四步：构建交易特征
        if not self.build_trading_features():
            print(f"❌ 特征构建失败")
            return None

        # 第五步：训练多任务AI模型
        if not self.train_multi_task_models():
            print(f"❌ 模型训练失败")
            return None

        # 第六步：生成交易信号
        return self.generate_trading_signals()


def main():

    analyzer_trainer = M1DataAnalyzerAndTrainer()
    analyzer_trainer.run_full_analysis_and_training(60)  # 分析过去60天的数据

    
def run():
    """供外部调用的运行函数"""
    analyzer_trainer = M1DataAnalyzerAndTrainer()
    return analyzer_trainer.run_full_analysis_and_training(60)  # 分析过去60天的数据


if __name__ == "__main__":
    main()
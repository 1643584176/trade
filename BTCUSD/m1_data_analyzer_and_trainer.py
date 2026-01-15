"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留周一到周五(0到4)
        
        # 删除临时的weekday列
        df = df.drop(columns=['weekday'])


        print(f"📊 实际时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")

        # 检查数据的连续性
        time_diffs = df['timestamp'].diff().dropna()

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
```

```
"""
M1数据趋势分析与AI模型训练一体化工具
功能：获取最新M1数据 → 分析趋势 → 训练AI模型 → 生成交易信号
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import math
import os
import shutil
import joblib

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)

warnings.filterwarnings('ignore')

# 🔥 实战交易参数（可根据自己的交易规则调整）
CONFIDENCE_THRESHOLD = 80  # 置信度≥80%才输出信号（过滤低置信度）
STOP_LOSS_RATIO = 0.5  # 止损幅度=预测幅度的50%（风控）
MIN_TREND_DURATION = 5  # 最小持仓时长（分钟），避免太短的无效信号
MAX_TREND_DURATION = 120  # 最大持仓时长（分钟），避免持仓过久
RISK_REWARD_RATIO = 2  # 风险收益比≥2（止盈/止损≥2，符合交易风控）


class M1DataAnalyzerAndTrainer:
    """BTCUSD M1数据趋势分析与AI模型训练一体化类"""
    
    def __init__(self, symbol="BTCUSD"):
        self.symbol = symbol  # 支持多种交易品种
        # 创建输出目录
        self.output_dir = f"m1_trend_analysis_results_{self.symbol}"
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
        if os.path.exists(self.output_dir):
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
    def fetch_m1_data_for_period(self, days_back=7):
        """获取过去指定天数的M1数据，为了提升性能，默认只分析7天数据"""

        # 初始化MT5连接
        if not mt5.initialize():
            print(f"❌ MT5初始化失败，错误代码: {mt5.last_error()}")
            return None


        # 计算时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        # 获取M1数据
        rates = mt5.copy_rates_range(self.symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"❌ 错误: 未获取到过去 {days_back} 天的M1数据")
            print(f"💡 提示: 请检查MT5终端是否开启，以及{self.symbol}品种是否可用")
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
        
        # 过滤掉周六和周日的数据
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['weekday'] = df['timestamp'].dt.weekday  # 0是周一, 6是周日
        df = df[df['weekday'] < 5]  # 只保留
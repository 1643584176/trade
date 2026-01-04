#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD 模型训练脚本
此脚本专门用于训练 XAUUSD 货币对的 AI 交易模型
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所需模块
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入公共特征工程模块 - 使用动态路径添加
import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保项目根目录在 sys.path 中
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 尝试另一种导入方式
import importlib.util

common_feature_engineer_path = os.path.join(project_root, 'mlc', 'common', 'common_feature_engineer.py')
spec = importlib.util.spec_from_file_location("common_feature_engineer", common_feature_engineer_path)
common_feature_engineer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_feature_engineer)
CommonFeatureEngineer = common_feature_engineer.CommonFeatureEngineer


class OptimizedEvoAIModel:
    """
    优化的AI模型类，专注于高效训练和预测
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
            n_estimators=200,  # 减少树的数量以提高效率
            max_depth=15,  # 限制树的深度
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
            # 使用公共特征工程模块生成的特征列
            # 注意：需要先用CommonFeatureEngineer.generate_all_features处理数据
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'dayOfWeek_sin', 'dayOfWeek_cos',
                'asia_session', 'europe_session', 'us_session',
                'asia_europe_overlap', 'europe_us_overlap',
                'body', 'upper_shadow', 'lower_shadow', 'total_range',
                'bullish', 'bearish',
                'sma_5', 'sma_10', 'sma_20',
                'close_to_sma5', 'close_to_sma10', 'close_to_sma20',
                'sma5_above_sma10', 'sma10_above_sma20',
                'volatility_10', 'volatility_20',
                'returns', 'momentum_5',
                'rsi', 'macd', 'macd_signal',
                'price_change', 'price_spike',
                'bb_middle', 'bb_upper', 'bb_lower',
                'sma_short', 'sma_long', 'ma_cross', 'rsi_reversal', 'local_high', 'local_low',
                'sma_5_direction', 'sma_10_direction', 'sma_20_direction',
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency',
                'vol_cluster', 'sma20_slope'
            ]

            # 创建目标变量（未来1个M15周期的价格变动方向）
            df = df.copy()
            df['future_return'] = df['close'].shift(-4) / df['close'] - 1  # M15数据，预测下一个M15周期
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
            # 使用公共特征工程模块生成的特征列
            # 注意：需要先用CommonFeatureEngineer.generate_all_features处理数据
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'hour', 'day_of_week', 'hour_sin', 'hour_cos', 'dayOfWeek_sin', 'dayOfWeek_cos',
                'asia_session', 'europe_session', 'us_session',
                'asia_europe_overlap', 'europe_us_overlap',
                'body', 'upper_shadow', 'lower_shadow', 'total_range',
                'bullish', 'bearish',
                'sma_5', 'sma_10', 'sma_20',
                'close_to_sma5', 'close_to_sma10', 'close_to_sma20',
                'sma5_above_sma10', 'sma10_above_sma20',
                'volatility_10', 'volatility_20',
                'returns', 'momentum_5',
                'rsi', 'macd', 'macd_signal',
                'price_change', 'price_spike',
                'bb_middle', 'bb_upper', 'bb_lower',
                'sma_short', 'sma_long', 'ma_cross', 'rsi_reversal', 'local_high', 'local_low',
                'sma_5_direction', 'sma_10_direction', 'sma_20_direction',
                'rsi_direction', 'ma_direction_consistency', 'rsi_price_consistency',
                'vol_cluster', 'sma20_slope'
            ]

            X = df[feature_columns]
            # 填充NaN值以避免预测时出错
            X = X.fillna(0)
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


def get_mt5_data(symbol="XAUUSD", timeframe="TIMEFRAME_M15", days=40):
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


def train_xauusd_model():
    """
    训练 XAUUSD 模型
    """
    try:
        logger.info("开始训练 XAUUSD 模型...")

        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 定义模型文件路径
        model_file = os.path.join(current_dir, "xauusd_trained_model.pkl")

        # 删除已存在的旧模型文件
        if os.path.exists(model_file):
            os.remove(model_file)
            logger.info(f"已删除旧模型文件: {model_file}")

        # 获取数据
        logger.info("获取历史数据...")
        try:
            df = get_mt5_data("XAUUSD", "TIMEFRAME_M15", 365)
            logger.info(f"获取到 {len(df)} 条历史数据")
        except Exception as e:
            logger.error(f"无法获取MT5数据: {str(e)}")
            return False

        # 使用公共特征工程模块生成特征
        logger.info("生成特征数据...")
        df_with_features = CommonFeatureEngineer.generate_all_features(df)

        # 初始化模型
        logger.info("初始化模型...")
        model = OptimizedEvoAIModel()

        # 准备训练数据
        logger.info("准备训练数据...")
        X, y = model.prepare_data(df_with_features)

        if X is None or y is None or len(X) == 0:
            logger.error("训练数据准备失败")
            return False

        # 确保训练数据中没有NaN值
        X = X.fillna(0)

        # 训练模型
        logger.info("开始训练模型...")
        model.train(X, y)

        # 保存模型
        logger.info(f"保存模型到 {model_file}...")
        model.save_model(model_file)

        logger.info("XAUUSD 模型训练完成")
        return True

    except Exception as e:
        logger.error(f"训练 XAUUSD 模型时出错: {str(e)}")
        return False


def main():
    """
    主函数
    """
    logger.info("=== XAUUSD 模型训练程序启动 ===")

    success = train_xauusd_model()

    if success:
        logger.info("=== XAUUSD 模型训练成功完成 ===")
    else:
        logger.error("=== XAUUSD 模型训练失败 ===")


if __name__ == "__main__":
    main()

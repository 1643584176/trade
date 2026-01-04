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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
        self.scaler = StandardScaler()  # 标准化器
        self.core_features = []  # 核心特征列表
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
            n_estimators=300,  # 根据记忆信息，从200增至300
            max_depth=20,  # 根据记忆信息，从15增至20
            min_samples_split=5,  # 根据记忆信息，从10减至5
            min_samples_leaf=2,  # 根据记忆信息，从5减至2
            random_state=42,
            n_jobs=-1
        )
        logger.info("AI模型初始化完成")

    def remove_high_corr_features(self, X, threshold=0.85):
        """
        剔除高相关性特征
        """
        try:
            # 计算相关性矩阵
            corr_matrix = X.corr().abs()
            
            # 获取上三角矩阵（避免重复计算）
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # 找到相关系数高于阈值的特征
            high_corr_features = [column for column in upper_triangle.columns 
                                  if any(upper_triangle[column] > threshold)]
            
            logger.info(f"发现 {len(high_corr_features)} 个高相关性特征: {high_corr_features}")
            
            # 剔除高相关性特征
            features_to_keep = [col for col in X.columns if col not in high_corr_features]
            
            logger.info(f"剔除高相关性特征后，保留 {len(features_to_keep)} 个特征")
            return X[features_to_keep], features_to_keep
            
        except Exception as e:
            logger.error(f"剔除高相关性特征时异常: {str(e)}")
            return X, list(X.columns)

    def select_top_features_by_importance(self, X, y, top_n=30):
        """
        根据特征重要性选择Top N特征
        """
        try:
            # 使用随机森林计算特征重要性
            temp_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # 确保没有NaN值
            X_clean = X.fillna(X.median())  # 使用中位数填充NaN值
            
            # 训练临时模型获取特征重要性
            temp_model.fit(X_clean, y)
            
            # 获取特征重要性
            feature_importances = temp_model.feature_importances_
            feature_names = X_clean.columns.tolist()
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            }).sort_values(by='importance', ascending=False)
            
            # 选择Top N特征
            top_features = importance_df.head(top_n)['feature'].tolist()
            
            # 简化日志输出，只输出关键信息
            logger.info(f"选择Top {top_n} 重要特征完成")
            
            return X[top_features], top_features
            
        except Exception as e:
            logger.error(f"选择重要特征时异常: {str(e)}")
            # 如果出错，返回原始特征
            return X, list(X.columns)

    def robust_feature_preprocess(self, X, fit_scaler=True):
        """
        鲁棒性特征预处理
        """
        try:
            X = X.copy()
            
            # 1. 用中位数填充缺失值（更符合金融数据分布）
            for col in X.columns:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
            
            # 2. 3σ原则处理异常值
            for col in X.columns:
                mean_val = X[col].mean()
                std_val = X[col].std()
                
                # 定义异常值边界
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                # 截断异常值
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 3. 标准化特征（训练时拟合，预测时复用）
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # 转换回DataFrame以保留列名
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"鲁棒性特征预处理异常: {str(e)}")
            return X

    def create_tradeable_target(self, df, threshold=0.0005):  # 5个点的阈值
        """
        创建有效涨跌标签（仅当未来波动幅度超过阈值时才标记）
        """
        try:
            df = df.copy()
            
            # 计算未来4个M15周期的价格变动
            df['future_close'] = df['close'].shift(-1)
            df['future_return'] = (df['future_close'] - df['close']) / df['close']
            
            # 计算未来价格变动幅度（绝对值）
            df['future_change_abs'] = abs(df['future_return'])
            
            # 仅当未来变动幅度超过阈值时才标记涨跌
            # 波动幅度不足的样本标记为NaN并后续剔除
            df['target'] = np.nan
            significant_moves = df['future_change_abs'] >= threshold
            
            # 对显著变动的样本进行涨跌标记
            df.loc[significant_moves, 'target'] = (df['future_return'] > 0).astype(int)
            
            # 统计有效样本数量
            valid_samples = df['target'].notna().sum()
            total_samples = len(df)
            
            logger.info(f"有效涨跌标签: {valid_samples}/{total_samples} 样本超过阈值({threshold*10000:.0f}个点)")
            
            return df
            
        except Exception as e:
            logger.error(f"创建有效涨跌标签异常: {str(e)}")
            return df

    def add_xauusd_specific_features(self, df):
        """
        新增XAUUSD场景化特征
        """
        try:
            df = df.copy()
            
            # 1. 波动率聚类特征
            df['volatility_garch'] = df['close'].rolling(window=20).std().rolling(window=10).mean()
            df['volatility_trend'] = df['close'].rolling(window=20).std().diff()
            
            # 2. 相对价格水平特征
            # 价格相对日内高低点的距离
            df['dist_to_intraday_high'] = (df['high'].rolling(window=24).max() - df['close']) / df['close']
            df['dist_to_intraday_low'] = (df['close'] - df['low'].rolling(window=24).min()) / df['close']
            
            # 价格相对整数关口的距离
            # 找到最近的整数关口（10的倍数）
            df['nearest_round_level'] = (df['close'] // 10) * 10
            df['dist_to_round_level'] = abs(df['close'] - df['nearest_round_level']) / df['close']
            
            # 3. 事件窗口特征（简单模拟，实际可对接经济日历API）
            # 这里模拟一些关键时间点，实际应用中应从经济日历获取
            df['in_event_window'] = 0  # 默认非事件窗口
            # 示例：假设某些时段为事件窗口（可根据实际数据调整）
            # 如每周三、四的特定时段可能有重要数据发布
            
            logger.info("XAUUSD场景化特征添加完成")
            return df
            
        except Exception as e:
            logger.error(f"添加XAUUSD场景化特征异常: {str(e)}")
            return df

    def prepare_data(self, df):
        """
        准备训练数据

        Args:
            df (DataFrame): 包含特征的原始数据

        Returns:
            tuple: (X, y) 特征和标签
        """
        try:
            # 使用公共特征工程模块生成特征
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
                'vol_cluster', 'sma20_slope',
                # 保留的XAUUSD趋势幅度敏感性特征（M15周期优化版）
                'dist_to_10_trend', 'near_10_trend',
                # 保留的价格变动幅度敏感性特征（M15周期优化版）
                'dist_to_10_change', 'dist_to_5_change', 'near_10_change', 'near_5_change',
                # 保留的价格水平敏感性特征（M15周期优化版）
                'dist_to_10_multiple', 'dist_to_5_multiple', 'near_10_multiple', 'near_5_multiple',
                # 优化的XAUUSD M15周期辅助特征
                'price_dist_to_10_multiple', 'is_near_10_multiple',
                'recent_price_moves_near_10', 'high_10_dist', 'low_10_dist',
                'xauusd_m15_10_interval_pos', 'xauusd_m15_towards_10',
                'xauusd_m15_volatility_near_10'
            ]

            # 添加XAUUSD场景化特征
            df = self.add_xauusd_specific_features(df)
            
            # 创建有效涨跌标签
            df = self.create_tradeable_target(df, threshold=0.0005)  # 5个点的阈值

            # 获取特征数据
            X = df[feature_columns]
            y = df['target']

            # 删除含有NaN的行（仅在训练时使用，仅针对标签）
            if len(df) > 100:  # 训练数据需要足够的样本
                mask = y.notna()  # 只保留有有效标签的样本
                X = X[mask]
                y = y[mask]

            logger.info(f"数据准备完成，特征数量: {len(feature_columns)}, 样本数量: {len(X)}")
            
            # 1. 鲁棒性预处理（处理异常值和缺失值）
            X = self.robust_feature_preprocess(X, fit_scaler=True)
            
            # 2. 剔除高相关性特征
            X, selected_features = self.remove_high_corr_features(X, threshold=0.85)
            
            # 3. 根据特征重要性选择Top30特征
            X, self.core_features = self.select_top_features_by_importance(X, y, top_n=30)
            
            logger.info(f"最终特征数量: {len(self.core_features)}, 样本数量: {len(X)}")
            
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
                'vol_cluster', 'sma20_slope',
                # 保留的XAUUSD趋势幅度敏感性特征（M15周期优化版）
                'dist_to_10_trend', 'near_10_trend',
                # 保留的价格变动幅度敏感性特征（M15周期优化版）
                'dist_to_10_change', 'dist_to_5_change', 'near_10_change', 'near_5_change',
                # 保留的价格水平敏感性特征（M15周期优化版）
                'dist_to_10_multiple', 'dist_to_5_multiple', 'near_10_multiple', 'near_5_multiple',
                # 优化的XAUUSD M15周期辅助特征
                'price_dist_to_10_multiple', 'is_near_10_multiple',
                'recent_price_moves_near_10', 'high_10_dist', 'low_10_dist',
                'xauusd_m15_10_interval_pos', 'xauusd_m15_towards_10',
                'xauusd_m15_volatility_near_10'
            ]

            # 添加XAUUSD场景化特征
            df = self.add_xauusd_specific_features(df)
            
            X = df[feature_columns]
            
            # 使用与训练时相同的特征列（核心特征）
            if self.core_features:
                # 确保预测数据包含所有核心特征
                missing_features = set(self.core_features) - set(X.columns)
                if missing_features:
                    logger.warning(f"预测数据缺少核心特征: {missing_features}")
                
                # 只使用训练时选择的核心特征
                available_features = [f for f in self.core_features if f in X.columns]
                X = X[available_features]
            else:
                # 如果没有核心特征列表，使用原始特征
                X = X[feature_columns]
            
            # 鲁棒性预处理（使用训练时拟合的标准化器）
            X = self.robust_feature_preprocess(X, fit_scaler=False)
            
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
                    'generation': self.generation,
                    'scaler': self.scaler,  # 保存标准化器
                    'core_features': self.core_features  # 保存核心特征列表
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
                self.scaler = data['scaler']  # 加载标准化器
                self.core_features = data['core_features']  # 加载核心特征列表
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
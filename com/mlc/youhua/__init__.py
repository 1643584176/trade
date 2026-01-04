#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD 模型训练脚本（特征优化版）
此脚本专门用于训练 XAUUSD 货币对的 AI 交易模型
核心优化：特征冗余剔除、鲁棒标签、场景化特征增强
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')

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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入公共特征工程模块 - 使用动态路径添加
import importlib.util

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保项目根目录在 sys.path 中
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入公共特征工程模块
try:
    common_feature_engineer_path = os.path.join(project_root, 'mlc', 'common', 'common_feature_engineer.py')
    spec = importlib.util.spec_from_file_location("common_feature_engineer", common_feature_engineer_path)
    common_feature_engineer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common_feature_engineer)
    CommonFeatureEngineer = common_feature_engineer.CommonFeatureEngineer
except Exception as e:
    logger.warning(f"导入公共特征工程模块失败: {e}")


    # 定义空的特征工程类作为备用
    class CommonFeatureEngineer:
        @staticmethod
        def generate_all_features(df):
            logger.warning("使用备用特征工程逻辑")
            return df


class OptimizedEvoAIModel:
    """
    优化的AI模型类，专注于高效训练和预测（特征优化版）
    """

    def __init__(self, model_file=None):
        """
        初始化AI模型
        """
        self.model = None
        self.performance_history = []
        self.generation = 0
        self.scaler = None  # 特征标准化器
        self.core_features = None  # 筛选后的核心特征
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

    def remove_high_corr_features(self, X, corr_threshold=0.85):
        """
        剔除高相关特征，解决维度灾难
        """
        try:
            # 计算特征相关性矩阵
            corr_matrix = X.corr().abs()
            # 取上三角矩阵（避免重复计算）
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            # 找出相关系数超过阈值的特征列
            high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
            # 剔除高相关特征
            X_clean = X.drop(columns=high_corr_cols)
            logger.info(f"剔除高相关特征数量: {len(high_corr_cols)}, 特征列表: {high_corr_cols[:10]}...")
            return X_clean
        except Exception as e:
            logger.error(f"剔除高相关特征失败: {e}")
            return X

    def select_core_features_by_importance(self, X, y, top_k=30):
        """
        基于特征重要性筛选核心特征
        """
        try:
            # 训练临时模型获取特征重要性
            temp_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            temp_model.fit(X, y)

            # 生成特征重要性 Series
            importances = temp_model.feature_importances_
            feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

            # 保留 top_k 特征
            self.core_features = feat_importance.head(top_k).index.tolist()
            X_selected = X[self.core_features]

            # 可视化特征重要性（可选）
            plt.figure(figsize=(12, 6))
            feat_importance.head(top_k).plot(kind='bar')
            plt.title(f'Top {top_k} 核心特征重要性')
            plt.ylabel('重要性得分')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_importance.png'))
            plt.close()

            logger.info(f"筛选核心特征数量: {len(self.core_features)}, 前10个核心特征: {self.core_features[:10]}")
            return X_selected
        except Exception as e:
            logger.error(f"筛选核心特征失败: {e}")
            return X

    def robust_feature_preprocess(self, X, is_train=True):
        """
        鲁棒特征预处理：处理缺失值、异常值、标准化
        """
        try:
            X = X.copy()

            # 1. 缺失值处理：用中位数填充（比0更符合金融数据分布）
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)

            # 2. 异常值处理：3σ原则截断（处理价格跳空、极端波动）
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = X[col].mean()
                std = X[col].std()
                X[col] = np.clip(X[col], mean - 3 * std, mean + 3 * std)

            # 3. 特征标准化（消除量纲影响）
            if is_train:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X.values

            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            logger.info(f"特征预处理完成，处理后特征形状: {X_scaled.shape}")
            return X_scaled
        except Exception as e:
            logger.error(f"特征预处理失败: {e}")
            return X

    def create_tradeable_target(self, df, threshold=0.0005):
        """
        创建鲁棒标签：考虑交易成本（点差+手续费），仅标记有效涨跌
        threshold: XAUUSD 约 5 个点波动（覆盖点差+手续费）
        """
        try:
            df = df.copy()
            # 预测未来4个M15周期（1小时）的收益率
            df['future_return'] = df['close'].shift(-4) / df['close'] - 1

            # 只有波动幅度超过阈值才标记为有效涨跌，否则为NaN（过滤无效样本）
            df['target'] = np.nan
            df.loc[df['future_return'] > threshold, 'target'] = 1  # 有效上涨
            df.loc[df['future_return'] < -threshold, 'target'] = 0  # 有效下跌

            # 剔除无效样本
            valid_samples = len(df.dropna(subset=['target']))
            total_samples = len(df)
            df = df.dropna(subset=['target'])

            logger.info(f"有效交易样本数: {valid_samples}/{total_samples} ({valid_samples / total_samples:.2%})")
            return df
        except Exception as e:
            logger.error(f"创建鲁棒标签失败: {e}")
            # 降级使用原标签逻辑
            df['future_return'] = df['close'].shift(-4) / df['close'] - 1
            df['target'] = (df['future_return'] > 0).astype(int)
            return df

    def add_enhanced_features(self, df):
        """
        新增XAUUSD场景化特征：波动率聚类、相对价格水平、事件窗口
        """
        try:
            df = df.copy()

            # 1. 波动率聚类特征（简化版GARCH）
            df['returns'] = df['close'].pct_change()
            df['volatility_garch'] = df['returns'].rolling(window=20).var()
            df['volatility_trend'] = df['volatility_garch'].diff().apply(lambda x: 1 if x > 0 else 0)

            # 2. 相对价格水平特征
            # 日内高低点（滚动24根M15 K线 = 6小时）
            df['intraday_high'] = df['high'].rolling(window=24).max()
            df['intraday_low'] = df['low'].rolling(window=24).min()
            # 相对距离
            df['dist_to_intraday_high'] = (df['intraday_high'] - df['close']) / df['close']
            df['dist_to_intraday_low'] = (df['close'] - df['intraday_low']) / df['close']

            # 整数关口距离（XAUUSD 整数关口如 2000、2010、2020）
            df['round_level'] = (df['close'] // 10) * 10
            df['dist_to_round_level'] = abs(df['close'] - df['round_level']) / df['close']

            # 3. 交易时段精细化特征（事件窗口标记，需结合经济日历）
            df['in_event_window'] = 0  # 默认非事件窗口
            # 示例：可从经济日历API获取真实事件时间，此处仅为演示
            event_hours = [8, 15, 21]  # 非农、CPI等关键数据发布时段
            df['hour'] = df['time'].dt.hour
            df.loc[df['hour'].isin(event_hours), 'in_event_window'] = 1

            logger.info("新增场景化特征完成")
            return df
        except Exception as e:
            logger.warning(f"新增场景化特征失败: {e}，使用原始特征")
            return df

    def prepare_data(self, df):
        """
        准备训练数据（特征优化版）
        """
        try:
            # 基础特征列定义
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

            # 创建鲁棒目标变量
            df = self.create_tradeable_target(df, threshold=0.0005)

            # 删除含有NaN的行（仅在训练时使用）
            if len(df) > 100:  # 训练数据需要足够的样本
                df = df.dropna()

            # 新增场景化特征
            df = self.add_enhanced_features(df)

            # 筛选存在的特征列（避免KeyError）
            feature_columns = [col for col in feature_columns if col in df.columns]
            X = df[feature_columns]
            y = df['target']

            logger.info(f"原始特征数量: {len(feature_columns)}, 样本数量: {len(X)}")

            # 步骤1：剔除高相关特征
            X = self.remove_high_corr_features(X)

            # 步骤2：特征预处理（缺失值+异常值+标准化）
            X = self.robust_feature_preprocess(X, is_train=True)

            # 步骤3：筛选核心特征
            X = self.select_core_features_by_importance(X, y, top_k=30)

            logger.info(f"最终特征数量: {X.shape[1]}, 最终样本数量: {X.shape[0]}")
            return X, y

        except Exception as e:
            logger.error(f"数据准备异常: {str(e)}")
            return None, None

    def prepare_prediction_data(self, df):
        """
        准备预测数据（不移除NaN值，使用训练好的scaler和核心特征）
        """
        try:
            # 新增场景化特征
            df = self.add_enhanced_features(df)

            # 使用训练时筛选的核心特征
            if self.core_features:
                # 筛选存在的特征列
                feature_columns = [col for col in self.core_features if col in df.columns]
                X = df[feature_columns]
            else:
                # 降级使用基础特征
                feature_columns = [
                    'open', 'high', 'low', 'close', 'tick_volume', 'hour', 'day_of_week',
                    'sma_5', 'sma_10', 'sma_20', 'rsi', 'macd', 'volatility_10'
                ]
                feature_columns = [col for col in feature_columns if col in df.columns]
                X = df[feature_columns]

            # 特征预处理（使用训练好的scaler）
            X = self.robust_feature_preprocess(X, is_train=False)

            # 填充NaN值以避免预测时出错
            X = X.fillna(0)
            return X

        except Exception as e:
            logger.error(f"预测数据准备异常: {str(e)}")
            return None

    def evaluate_trading_model(self, y_test, y_pred, y_pred_proba):
        """
        交易模型专用评估：准确率+混淆矩阵+AUC+盈亏比
        """
        try:
            # 基础指标
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # 交易场景指标：胜率、盈亏比（假设止盈0.001，止损0.0005）
            tp = cm[1, 1]  # 正确预测上涨
            tn = cm[0, 0]  # 正确预测下跌
            fp = cm[0, 1]  # 错误预测上涨（假多）
            fn = cm[1, 0]  # 错误预测下跌（假空）

            win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0  # 做多胜率
            profit_loss_ratio = (tp * 0.001 - fp * 0.0005) / (fp * 0.0005) if fp > 0 else float('inf')

            logger.info(f"===== 模型评估结果 =====")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"AUC: {auc:.4f}")
            logger.info(f"做多胜率: {win_rate:.4f}")
            logger.info(f"盈亏比: {profit_loss_ratio:.4f}")
            logger.info(f"混淆矩阵:\n{cm}")
            logger.info(f"分类报告:\n{report}")

            return {
                'accuracy': accuracy, 'auc': auc,
                'win_rate': win_rate, 'profit_loss_ratio': profit_loss_ratio
            }
        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            return {'accuracy': 0, 'auc': 0, 'win_rate': 0, 'profit_loss_ratio': 0}

    def train(self, X, y):
        """
        训练模型（带交易场景评估）
        """
        try:
            # 时间分层划分训练集和测试集（避免数据泄露）
            X_train = X.iloc[:-int(len(X) * 0.2)]
            X_test = X.iloc[-int(len(X) * 0.2):]
            y_train = y.iloc[:-int(len(y) * 0.2)]
            y_test = y.iloc[-int(len(y) * 0.2):]

            # 训练模型
            self.model.fit(X_train, y_train)

            # 评估模型
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            eval_metrics = self.evaluate_trading_model(y_test, y_pred, y_pred_proba)

            # 记录性能
            self.performance_history.append({
                'generation': self.generation,
                'accuracy': eval_metrics['accuracy'],
                'auc': eval_metrics['auc'],
                'win_rate': eval_metrics['win_rate'],
                'profit_loss_ratio': eval_metrics['profit_loss_ratio'],
                'samples': len(X)
            })

            logger.info(f"模型训练完成，代数: {self.generation}")

        except Exception as e:
            logger.error(f"模型训练异常: {str(e)}")

    def predict(self, X):
        """
        预测信号
        """
        try:
            predictions = self.model.predict_proba(X)
            return predictions

        except Exception as e:
            logger.error(f"预测异常: {str(e)}")
            return None

    def save_model(self, filename):
        """
        保存模型（包含scaler和核心特征）
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'performance_history': self.performance_history,
                    'generation': self.generation,
                    'scaler': self.scaler,
                    'core_features': self.core_features
                }, f)
            logger.info(f"模型已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存模型异常: {str(e)}")

    def load_model(self, filename):
        """
        加载模型（包含scaler和核心特征）
        """
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.performance_history = data['performance_history']
                self.generation = data['generation']
                self.scaler = data.get('scaler')
                self.core_features = data.get('core_features')
            logger.info(f"模型已加载自 {filename}")
        except Exception as e:
            logger.error(f"加载模型异常: {str(e)}")


def get_mt5_data(symbol="XAUUSD", timeframe="TIMEFRAME_M15", days=40):
    """
    从MT5获取历史数据
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
    训练 XAUUSD 模型（特征优化版）
    """
    try:
        logger.info("开始训练 XAUUSD 模型（特征优化版）...")

        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 定义模型文件路径
        model_file = os.path.join(current_dir, "xauusd_trained_model_optimized.pkl")

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
            # 生成模拟数据用于测试（无MT5时）
            logger.warning("使用模拟数据进行测试")
            date_range = pd.date_range(start='2024-01-01', end='2025-01-01', freq='15T')
            df = pd.DataFrame({
                'time': date_range,
                'open': np.random.uniform(1900, 2100, len(date_range)),
                'high': np.random.uniform(1900, 2100, len(date_range)),
                'low': np.random.uniform(1900, 2100, len(date_range)),
                'close': np.random.uniform(1900, 2100, len(date_range)),
                'tick_volume': np.random.randint(100, 1000, len(date_range))
            })

        # 使用公共特征工程模块生成特征
        logger.info("生成特征数据...")
        df_with_features = CommonFeatureEngineer.generate_all_features(df)

        # 初始化模型
        logger.info("初始化模型...")
        model = OptimizedEvoAIModel()

        # 准备训练数据
        logger.info("准备训练数据（特征优化版）...")
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

        logger.info("XAUUSD 模型训练完成（特征优化版）")
        return True

    except Exception as e:
        logger.error(f"训练 XAUUSD 模型时出错: {str(e)}")
        return False


def main():
    """
    主函数
    """
    logger.info("=== XAUUSD 模型训练程序启动（特征优化版） ===")

    success = train_xauusd_model()

    if success:
        logger.info("=== XAUUSD 模型训练成功完成（特征优化版） ===")
    else:
        logger.error("=== XAUUSD 模型训练失败（特征优化版） ===")


if __name__ == "__main__":
    main()
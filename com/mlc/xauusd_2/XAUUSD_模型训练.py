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
import logging
from datetime import datetime, timedelta
import time
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def evaluate_model_during_training(model, X, y, sample_indices):
    """
    在训练过程中评估模型预测
    
    参数:
        model (EvoAIModel): 模型对象
        X (DataFrame): 特征数据
        y (Series): 标签数据
        sample_indices (list): 采样索引列表
    """
    try:
        # 选取部分样本进行预测（避免输出过多信息）
        sampled_X = X.iloc[sample_indices]
        sampled_y = y.iloc[sample_indices]
        
        # 进行预测
        predictions = model.model.predict_proba(sampled_X)
        
        # 输出预测结果
        logger.info("=== 模型预测示例 ===")
        for i in range(min(5, len(predictions))):
            actual_label = sampled_y.iloc[i]
            predicted_prob = predictions[i][1]  # 上涨概率
            
            direction = "上涨" if predicted_prob > 0.5 else "下跌"
            confidence = predicted_prob if predicted_prob > 0.5 else (1 - predicted_prob)
            
            logger.info(f"实际方向: {'上涨' if actual_label == 1 else '下跌'}, "
                       f"预测方向: {direction}, "
                       f"置信度: {confidence:.4f} ({predicted_prob:.4f})")
        
        return True
    except Exception as e:
        logger.error(f"训练过程中评估模型异常: {str(e)}")
        return False


def train_xauusd_model():
    """
    训练 XAUUSD 模型
    """
    try:
        logger.info("开始训练 XAUUSD 模型...")
        
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取当前包名
        package_name = os.path.basename(current_dir)
        
        # 添加当前目录到系统路径
        sys.path.append(current_dir)
        
        # 动态导入回测文件中的类
        backtest_file = os.path.join(current_dir, "XAUUSD_Backtest_M15.py")
        if not os.path.exists(backtest_file):
            logger.error(f"找不到回测文件: {backtest_file}")
            return False
        
        # 导入必要的类
        from XAUUSD_Backtest_M15 import FeatureEngineer, EvoAIModel
        
        # 获取数据
        logger.info("获取历史数据...")
        try:
            df = get_mt5_data("XAUUSD", "TIMEFRAME_M15", 365)
            logger.info(f"获取到 {len(df)} 条历史数据")
        except Exception as e:
            logger.error(f"无法获取MT5数据: {str(e)}")
            return False
        
        # 初始化特征工程和模型
        logger.info("初始化特征工程和模型...")
        feature_engineer = FeatureEngineer()
        model = EvoAIModel()
        
        # 生成特征
        logger.info("生成特征数据...")
        df_with_features = feature_engineer.generate_features(df)
        
        # 准备训练数据
        logger.info("准备训练数据...")
        X, y = model.prepare_data(df_with_features)
        
        if X is None or y is None or len(X) == 0:
            logger.error("训练数据准备失败")
            return False
        
        # 分割训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 选取一些样本用于周期性预测展示
        sample_indices = list(range(0, min(20, len(X_test)), 1))  # 选取前20个样本中的部分
        
        # 记录开始时间
        start_time = time.time()
        last_evaluation_time = start_time
        
        # 手动训练模型以便进行中间评估
        logger.info("开始训练模型...")
        from sklearn.metrics import accuracy_score
        
        # 模拟增量训练过程（这里简化处理，实际可以采用更复杂的在线学习方法）
        # 先用部分数据训练
        batch_size = max(100, len(X_train) // 10)  # 每批大约10%的数据
        total_batches = len(X_train) // batch_size + (1 if len(X_train) % batch_size > 0 else 0)
        
        for batch_idx in range(total_batches):
            # 获取当前批次数据
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train))
            X_batch = X_train.iloc[start_idx:end_idx]
            y_batch = y_train.iloc[start_idx:end_idx]
            
            # 增量训练模型（对于RandomForest，实际上是重新训练）
            # 注意：sklearn的RandomForest不支持真正的增量学习，这里仅作演示
            model.model.fit(X_batch, y_batch)
            
            # 检查是否需要进行评估输出（每分钟一次）
            current_time = time.time()
            if current_time - last_evaluation_time >= 60:  # 每隔60秒输出一次
                logger.info(f"训练进度: {min((batch_idx+1)*100/total_batches, 100):.1f}%")
                evaluate_model_during_training(model, X_test, y_test, sample_indices)
                last_evaluation_time = current_time
        
        # 最终评估
        y_pred = model.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 记录性能
        model.performance_history.append({
            'generation': model.generation,
            'accuracy': accuracy,
            'samples': len(X)
        })
        
        logger.info(f"模型训练完成，准确率: {accuracy:.4f}，代数: {model.generation}")
        
        # 删除旧的模型文件
        model_filename = f"{package_name}_trained_model.pkl"
        model_file = os.path.join(current_dir, model_filename)
        if os.path.exists(model_file):
            logger.info(f"删除旧的模型文件: {model_file}")
            os.remove(model_file)
        
        # 保存模型
        logger.info(f"保存模型到 {model_file}...")
        model.save_model(model_file)
        
        # 训练用于预测最高价和最低价的回归模型
        logger.info("开始训练最高价和最低价预测模型...")
        train_high_low_models(df_with_features, current_dir, package_name)
        
        logger.info("XAUUSD 模型训练完成")
        return True
        
    except Exception as e:
        logger.error(f"训练 XAUUSD 模型时出错: {str(e)}")
        return False


def train_high_low_models(df, current_dir, package_name):
    """
    训练用于预测最高价和最低价的回归模型
    
    参数:
        df (DataFrame): 包含特征的数据
        current_dir (str): 当前目录路径
        package_name (str): 包名
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        # 导入必要的类
        from XAUUSD_Backtest_M15 import FeatureEngineer
        
        # 初始化特征工程
        feature_engineer = FeatureEngineer()
        
        # 生成特征
        df_with_features = feature_engineer.generate_features(df)
        
        # 选择特征列（与分类模型相同的特征）
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
            'bb_position', 'trend_strength', 'reversal_position', 
            'historical_returns', 'recent_high', 'recent_low',
            'direction_streak', 'direction_persistence', 'price_direction',
            'recent_trade_performance', 'consecutive_wins', 'consecutive_losses', 'win_rate',
            'intraday_return', 'position_to_high', 'position_to_low',
            'upper_shadow_ratio', 'lower_shadow_ratio', 'high_drawdown', 'low_bounce'
        ]
        
        # 创建目标变量（未来1个周期的最高价和最低价）
        df_with_features = df_with_features.copy()
        df_with_features['future_high'] = df_with_features['high'].shift(-1)
        df_with_features['future_low'] = df_with_features['low'].shift(-1)
        
        # 删除含有NaN的行
        df_clean = df_with_features.dropna()
        
        if len(df_clean) < 100:
            logger.error("数据不足，无法训练最高价和最低价预测模型")
            return False
        
        # 准备特征和目标变量
        X = df_clean[feature_columns]
        y_high = df_clean['future_high']
        y_low = df_clean['future_low']
        
        # 分割训练集和测试集
        X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(
            X, y_high, test_size=0.2, random_state=42
        )
        
        X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(
            X, y_low, test_size=0.2, random_state=42
        )
        
        # 训练最高价预测模型
        logger.info("训练最高价预测模型...")
        high_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        high_model.fit(X_train_high, y_train_high)
        
        # 评估最高价预测模型
        y_pred_high = high_model.predict(X_test_high)
        mse_high = mean_squared_error(y_test_high, y_pred_high)
        mae_high = mean_absolute_error(y_test_high, y_pred_high)
        logger.info(f"最高价预测模型 - MSE: {mse_high:.6f}, MAE: {mae_high:.6f}")
        
        # 训练最低价预测模型
        logger.info("训练最低价预测模型...")
        low_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        low_model.fit(X_train_low, y_train_low)
        
        # 评估最低价预测模型
        y_pred_low = low_model.predict(X_test_low)
        mse_low = mean_squared_error(y_test_low, y_pred_low)
        mae_low = mean_absolute_error(y_test_low, y_pred_low)
        logger.info(f"最低价预测模型 - MSE: {mse_low:.6f}, MAE: {mae_low:.6f}")
        
        # 保存最高价和最低价预测模型
        high_model_filename = f"{package_name}_high_predictor.pkl"
        high_model_file = os.path.join(current_dir, high_model_filename)
        low_model_filename = f"{package_name}_low_predictor.pkl"
        low_model_file = os.path.join(current_dir, low_model_filename)
        
        # 删除旧的模型文件
        if os.path.exists(high_model_file):
            logger.info(f"删除旧的最高价预测模型文件: {high_model_file}")
            os.remove(high_model_file)
            
        if os.path.exists(low_model_file):
            logger.info(f"删除旧的最低价预测模型文件: {low_model_file}")
            os.remove(low_model_file)
        
        # 保存模型
        with open(high_model_file, 'wb') as f:
            pickle.dump(high_model, f)
        logger.info(f"最高价预测模型已保存到: {high_model_file}")
        
        with open(low_model_file, 'wb') as f:
            pickle.dump(low_model, f)
        logger.info(f"最低价预测模型已保存到: {low_model_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练最高价和最低价预测模型时出错: {str(e)}")
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
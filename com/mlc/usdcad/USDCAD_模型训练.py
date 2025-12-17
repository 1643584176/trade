#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USDCAD 模型训练脚本
此脚本专门用于训练 USDCAD 货币对的 AI 交易模型
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所需模块
import pandas as pd
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_mt5_data(symbol="USDCAD", timeframe="TIMEFRAME_M15", days=365):
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


def train_usdcad_model():
    """
    训练 USDCAD 模型
    """
    try:
        logger.info("开始训练 USDCAD 模型...")
        
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 添加当前目录到系统路径
        sys.path.append(current_dir)
        
        # 动态导入回测文件中的类
        backtest_file = os.path.join(current_dir, "USDCAD_Backtest_M15.py")
        if not os.path.exists(backtest_file):
            logger.error(f"找不到回测文件: {backtest_file}")
            return False
        
        # 导入必要的类
        from USDCAD_Backtest_M15 import FeatureEngineer, EvoAIModel
        
        # 获取数据
        logger.info("获取历史数据...")
        try:
            df = get_mt5_data("USDCAD", "TIMEFRAME_M15", 365)
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
        
        # 训练模型
        logger.info("开始训练模型...")
        model.train(X, y)
        
        # 保存模型
        model_file = os.path.join(current_dir, "usdcad_trained_model.pkl")
        logger.info(f"保存模型到 {model_file}...")
        model.save_model(model_file)
        
        logger.info("USDCAD 模型训练完成")
        return True
        
    except Exception as e:
        logger.error(f"训练 USDCAD 模型时出错: {str(e)}")
        return False

def main():
    """
    主函数
    """
    logger.info("=== USDCAD 模型训练程序启动 ===")
    
    success = train_usdcad_model()
    
    if success:
        logger.info("=== USDCAD 模型训练成功完成 ===")
    else:
        logger.error("=== USDCAD 模型训练失败 ===")

if __name__ == "__main__":
    main()
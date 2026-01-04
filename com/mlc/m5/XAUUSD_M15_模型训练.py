import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import sys
import os
from datetime import datetime, timedelta, timezone
import warnings
import importlib.util
warnings.filterwarnings('ignore')

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

# 动态导入基类
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_trainer_base_path = os.path.join(project_root, 'mlc', 'm5', 'model_trainer_base.py')
spec = importlib.util.spec_from_file_location("model_trainer_base", model_trainer_base_path)
model_trainer_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_trainer_base)
BaseModelTrainer = model_trainer_base.BaseModelTrainer

# 配置参数
class M15ModelConfig:
    SYMBOL = "XAUUSD"
    M15_TIMEFRAME = mt5.TIMEFRAME_M15
    HISTORY_M15_BARS = 200  # 用于预测的M15 K线数量（200根）
    PREDICT_FUTURE_BARS = 4  # 预测未来K线数量（1-4根）
    TRAIN_TEST_SPLIT = 0.8
    MODEL_SAVE_PATH = "xauusd_m15_model.json"  # XGBoost模型保存路径
    SCALER_SAVE_PATH = "m15_scaler.pkl"
    UTC_TZ = timezone.utc

class M15ModelTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__()
        self.config = M15ModelConfig()
    
    def get_m15_historical_data(self, bars_count: int = 365*24*4):  # 一年的M15数据
        """获取MT5真实历史M15数据"""
        self.initialize_mt5()
        
        # 获取当前时间
        current_utc = datetime.now(self.config.UTC_TZ)
        start_time = current_utc - timedelta(minutes=15*bars_count)  # M15数据，每根K线15分钟
        
        # 获取历史数据
        m15_rates = mt5.copy_rates_range(
            self.config.SYMBOL,
            self.config.M15_TIMEFRAME,
            int(start_time.timestamp()),
            int(current_utc.timestamp())
        )
        
        if m15_rates is None or len(m15_rates) == 0:
            raise Exception(f"获取M15历史数据失败：{mt5.last_error()}")
        
        # 准备数据和特征
        df = self.prepare_data_with_features(m15_rates, "M15")
        
        # 创建目标变量：预测未来1-4根K线的涨跌 (1=涨, 0=跌, -1=平)
        df['future_close_1'] = df['close'].shift(-1)  # 预测1根K线后
        df['future_close_2'] = df['close'].shift(-2)  # 预测2根K线后
        df['future_close_3'] = df['close'].shift(-3)  # 预测3根K线后
        df['future_close_4'] = df['close'].shift(-4)  # 预测4根K线后
        
        # 使用预测1根K线的涨跌作为目标
        df['price_change_pct'] = (df['future_close_1'] - df['close']) / df['close']
        
        # 定义目标变量 - M15周期可能波动更大，使用更大阈值
        df['target'] = np.where(df['price_change_pct'] > 0.002, 1,  # 涨（0.2%阈值，适应M15的较大波动）
                               np.where(df['price_change_pct'] < -0.002, -1, 0))  # 跌和平
        
        return df

    def train_model(self):
        """训练M15模型"""
        print("开始获取M15历史数据...")
        df = self.get_m15_historical_data(bars_count=365*24*4)  # 获取一年的M15数据
        
        print(f"获取到 {len(df)} 条历史数据")
        
        # 准备特征和目标变量
        X, y, feature_names = self.prepare_features_and_target(df, "M15")
        
        # 分割训练测试集
        split_idx = int(len(X) * self.config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 训练XGBoost模型
        print("开始训练XGBoost模型...")
        model, train_score, test_score = self.train_xgboost_model(
            X_train, X_test, y_train, y_test,
            model_params={
                'n_estimators': 250,  # M15模型，中等数量的估计器
                'max_depth': 18,
                'learning_rate': 0.1,
                'min_child_weight': 2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
        )
        
        # 特征重要性
        feature_importance = model.feature_importances_
        print("\n前10个最重要特征:")
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # 保存模型
        model.save_model(self.config.MODEL_SAVE_PATH)
        print(f"模型已保存至: {self.config.MODEL_SAVE_PATH}")
        
        return model, feature_names

def main():
    """主函数"""
    print("开始训练XAUUSD M15周期XGBoost模型")
    try:
        trainer = M15ModelTrainer()
        model, features = trainer.train_model()
        print("M15模型训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
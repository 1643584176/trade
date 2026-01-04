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
class M1ModelConfig:
    SYMBOL = "XAUUSD"
    M1_TIMEFRAME = mt5.TIMEFRAME_M1
    M5_TIMEFRAME = mt5.TIMEFRAME_M5
    M15_TIMEFRAME = mt5.TIMEFRAME_M15
    HISTORY_M1_BARS = 50  # 用于预测的M1 K线数量（30-50根）
    PREDICT_FUTURE_BARS = 3  # 预测未来K线数量
    TRAIN_TEST_SPLIT = 0.8
    MODEL_SAVE_PATH = "xauusd_m1_model.json"  # XGBoost模型保存路径
    SCALER_SAVE_PATH = "m1_scaler.pkl"
    UTC_TZ = timezone.utc

class M1ModelTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__()
        self.config = M1ModelConfig()
    
    def get_m1_historical_data(self, bars_count: int = 60*24*60):  # 60天的M1数据
        """获取MT5真实历史M1数据"""
        self.initialize_mt5()
        
        # 获取当前时间
        current_utc = datetime.now(self.config.UTC_TZ)
        start_time = current_utc - timedelta(minutes=bars_count)
        
        # 获取历史数据
        m1_rates = mt5.copy_rates_range(
            self.config.SYMBOL,
            self.config.M1_TIMEFRAME,
            int(start_time.timestamp()),
            int(current_utc.timestamp())
        )
        
        if m1_rates is None or len(m1_rates) == 0:
            raise Exception(f"获取M1历史数据失败：{mt5.last_error()}")
        
        # 转换为DataFrame
        df = pd.DataFrame(m1_rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        
        # 准备数据和特征
        df = self.prepare_data_with_features(m1_rates, "M1")
        
        # 创建目标变量：预测未来1-3根K线的涨跌 (1=涨, 0=跌, -1=平)
        df['future_close_1'] = df['close'].shift(-1)  # 预测1根K线后
        df['future_close_2'] = df['close'].shift(-2)  # 预测2根K线后
        df['future_close_3'] = df['close'].shift(-3)  # 预测3根K线后
        
        # 使用预测1根K线的涨跌作为目标
        df['price_change_pct'] = (df['future_close_1'] - df['close']) / df['close']
        
        # 定义目标变量
        df['target'] = np.where(df['price_change_pct'] > 0.0005, 1,  # 涨（0.05%阈值，适应M1的微小波动）
                               np.where(df['price_change_pct'] < -0.0005, -1, 0))  # 跌和平
        
        return df

    def train_model(self):
        """训练M1模型"""
        print("开始获取M1历史数据...")
        df = self.get_m1_historical_data(bars_count=60*24*60)  # 获取60天的M1数据
        
        print(f"获取到 {len(df)} 条历史数据")
        
        # 准备特征和目标变量
        X, y, feature_names = self.prepare_features_and_target(df, "M1")
        
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
                'n_estimators': 200,  # M1模型可能需要较少的估计器，因为数据量大
                'max_depth': 15,
                'learning_rate': 0.1,
                'min_child_weight': 3,
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
    print("开始训练XAUUSD M1周期XGBoost模型")
    try:
        trainer = M1ModelTrainer()
        model, features = trainer.train_model()
        print("M1模型训练完成！")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
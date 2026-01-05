import threading
import time
import sys
import os
import importlib.util
from datetime import datetime

# 添加公共模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "common"))

def train_m1_model():
    """训练M1模型"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练M1模型...")
    
    try:
        # 动态导入M1模型训练器并执行
        m1_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XAUUSD_M1_模型训练.py')
        
        # 使用importlib导入模块并执行
        spec = importlib.util.spec_from_file_location("XAUUSD_M1_模型训练", m1_trainer_path)
        m1_module = importlib.util.module_from_spec(spec)
        
        # 执行模块
        spec.loader.exec_module(m1_module)
        
        # 获取训练器类并执行
        if hasattr(m1_module, 'M1ModelTrainer'):
            trainer = m1_module.M1ModelTrainer()
            model, features = trainer.train_model()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M1模型训练完成！")
            return model, features
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M1模型训练器类不存在")
            return None, None
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M1模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_m5_model():
    """训练M5模型"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练M5模型...")
    
    try:
        # 动态导入M5模型训练器并执行
        m5_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XAUUSD_M5_模型训练.py')
        
        # 使用importlib导入模块并执行
        spec = importlib.util.spec_from_file_location("XAUUSD_M5_模型训练", m5_trainer_path)
        m5_module = importlib.util.module_from_spec(spec)
        
        # 执行模块
        spec.loader.exec_module(m5_module)
        
        # 获取训练器类并执行
        if hasattr(m5_module, 'M5ModelTrainer'):
            trainer = m5_module.M5ModelTrainer()
            model, features = trainer.train_model()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M5模型训练完成！")
            return model, features
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M5模型训练器类不存在")
            return None, None
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M5模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_m15_model():
    """训练M15模型"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始训练M15模型...")
    
    try:
        # 动态导入M15模型训练器并执行
        m15_trainer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XAUUSD_M15_模型训练.py')
        
        # 使用importlib导入模块并执行
        spec = importlib.util.spec_from_file_location("XAUUSD_M15_模型训练", m15_trainer_path)
        m15_module = importlib.util.module_from_spec(spec)
        
        # 执行模块
        spec.loader.exec_module(m15_module)
        
        # 获取训练器类并执行
        if hasattr(m15_module, 'M15ModelTrainer'):
            trainer = m15_module.M15ModelTrainer()
            model, features = trainer.train_model()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M15模型训练完成！")
            return model, features
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M15模型训练器类不存在")
            return None, None
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] M15模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数 - 并行训练三个模型"""
    print("="*60)
    print("开始并行训练M1、M5和M15模型")
    print("="*60)
    
    # 创建线程
    m1_thread = threading.Thread(target=train_m1_model, name="M1-Thread")
    m5_thread = threading.Thread(target=train_m5_model, name="M5-Thread")
    m15_thread = threading.Thread(target=train_m15_model, name="M15-Thread")
    
    # 启动线程
    start_time = time.time()
    
    m1_thread.start()
    m5_thread.start()
    m15_thread.start()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有模型训练线程已启动")
    
    # 等待所有线程完成
    m1_thread.join()
    m5_thread.join()
    m15_thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("="*60)
    print(f"所有模型训练完成！总耗时: {total_time:.2f}秒")
    print("="*60)


if __name__ == "__main__":
    main()
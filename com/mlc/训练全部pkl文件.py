import os
import sys
import importlib.util
import logging
from pathlib import Path
import traceback
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_currency_pairs():
    """
    获取所有货币对目录
    """
    base_path = Path("D:/newProject/Trader/com/mlc")
    currency_dirs = []
    
    # 遍历目录查找货币对文件夹
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in ['test', 'templates']:
            # 检查目录中是否有回测文件
            backtest_files = list(item.glob("*_Backtest_M15.py"))
            if backtest_files:
                currency_dirs.append((item.name, str(backtest_files[0])))
    
    return currency_dirs

def train_single_model(currency_pair, backtest_file_path):
    """
    训练单个货币对的模型
    
    Args:
        currency_pair (str): 货币对名称
        backtest_file_path (str): 回测文件路径
    """
    try:
        logger.info(f"开始训练 {currency_pair} 模型...")
        
        # 获取货币对目录
        backtest_dir = os.path.dirname(backtest_file_path)
        
        # 构造训练文件路径
        training_file_name = f"{currency_pair.upper()}_模型训练.py"
        training_file_path = os.path.join(backtest_dir, training_file_name)
        
        # 检查训练文件是否存在
        if not os.path.exists(training_file_path):
            logger.error(f"找不到训练文件: {training_file_path}")
            return False
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        # 切换到训练文件所在目录
        os.chdir(backtest_dir)
        
        try:
            # 动态导入并执行训练文件中的main函数
            spec = importlib.util.spec_from_file_location(f"{currency_pair}_training", training_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{currency_pair}_training"] = module  # 添加到sys.modules防止循环导入
            spec.loader.exec_module(module)
            
            # 执行main函数进行训练
            if hasattr(module, 'main'):
                module.main()
                logger.info(f"{currency_pair} 模型训练完成")
                return True
            else:
                logger.error(f"{currency_pair} 训练文件中没有找到main函数")
                return False
        finally:
            # 恢复原来的工作目录
            os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"训练 {currency_pair} 模型时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def train_all_models():
    """
    训练所有货币对的模型
    """
    logger.info("开始批量训练所有模型...")
    
    # 获取所有货币对
    currency_pairs = get_currency_pairs()
    
    if not currency_pairs:
        logger.warning("未找到任何货币对目录")
        return
    
    logger.info(f"找到 {len(currency_pairs)} 个货币对: {[pair[0] for pair in currency_pairs]}")
    
    # 记录成功和失败的训练
    successful_trainings = []
    failed_trainings = []
    
    # 逐个训练模型
    for currency_pair, backtest_file_path in currency_pairs:
        if train_single_model(currency_pair, backtest_file_path):
            successful_trainings.append(currency_pair)
        else:
            failed_trainings.append(currency_pair)
        
        # 稍微延时以避免系统资源占用过高
        time.sleep(1)
    
    # 输出总结
    logger.info("=" * 50)
    logger.info("批量训练完成")
    logger.info("=" * 50)
    logger.info(f"成功训练: {len(successful_trainings)} 个")
    if successful_trainings:
        logger.info(f"成功模型: {', '.join(successful_trainings)}")
    
    logger.info(f"训练失败: {len(failed_trainings)} 个")
    if failed_trainings:
        logger.info(f"失败模型: {', '.join(failed_trainings)}")

def main():
    """
    主函数
    """
    try:
        train_all_models()
    except KeyboardInterrupt:
        logger.info("用户中断了训练过程")
    except Exception as e:
        logger.error(f"批量训练过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
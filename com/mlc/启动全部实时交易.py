import os
import sys
import subprocess
import logging
from pathlib import Path
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_currency_pairs():
    """
    获取所有包含实时交易文件的货币对目录
    """
    base_path = Path("D:/newProject/Trader/com/mlc")
    currency_dirs = []
    
    # 遍历目录查找货币对文件夹
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in ['test', 'templates']:
            # 棜查目录中是否有实时交易文件
            trader_files = list(item.glob("*_trader_m15.py"))
            if trader_files:
                currency_dirs.append((item.name, str(trader_files[0])))
    
    return currency_dirs

def start_single_trader(currency_pair, trader_file_path):
    """
    启动单个货币对的实时交易
    
    Args:
        currency_pair (str): 货币对名称
        trader_file_path (str): 实时交易文件路径
    """
    try:
        logger.info(f"启动 {currency_pair} 实时交易...")
        
        # 检查实时交易文件是否存在
        if not os.path.exists(trader_file_path):
            logger.error(f"找不到实时交易文件: {trader_file_path}")
            return False
        
        # 在后台启动实时交易进程
        # 使用subprocess.Popen启动独立进程
        process = subprocess.Popen(
            [sys.executable, trader_file_path],
            cwd=os.path.dirname(trader_file_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"{currency_pair} 实时交易已启动，进程ID: {process.pid}")
        return process
            
    except Exception as e:
        logger.error(f"启动 {currency_pair} 实时交易时出错: {str(e)}")
        return None

def start_all_traders():
    """
    启动所有货币对的实时交易
    """
    logger.info("开始启动所有实时交易...")
    
    # 获取所有货币对
    currency_pairs = get_currency_pairs()
    
    if not currency_pairs:
        logger.warning("未找到任何货币对的实时交易文件")
        return
    
    logger.info(f"找到 {len(currency_pairs)} 个货币对的实时交易文件: {[pair[0] for pair in currency_pairs]}")
    
    # 记录启动成功的交易和失败的交易
    started_processes = []
    failed_starts = []
    
    # 逐个启动实时交易
    for currency_pair, trader_file_path in currency_pairs:
        process = start_single_trader(currency_pair, trader_file_path)
        if process:
            started_processes.append((currency_pair, process))
        else:
            failed_starts.append(currency_pair)
        
        # 稍微延时以避免同时启动过多进程
        time.sleep(2)
    
    # 输出总结
    logger.info("=" * 50)
    logger.info("实时交易启动完成")
    logger.info("=" * 50)
    logger.info(f"成功启动: {len(started_processes)} 个")
    if started_processes:
        for currency_pair, process in started_processes:
            logger.info(f"  - {currency_pair}: PID {process.pid}")
    
    logger.info(f"启动失败: {len(failed_starts)} 个")
    if failed_starts:
        logger.info(f"  - 失败的货币对: {', '.join(failed_starts)}")
        
    return started_processes, failed_starts

def monitor_processes(processes):
    """
    监控运行中的进程
    
    Args:
        processes (list): (currency_pair, process) 元组列表
    """
    logger.info("开始监控实时交易进程...")
    
    while True:
        try:
            running_processes = []
            for currency_pair, process in processes:
                if process.poll() is None:  # 进程仍在运行
                    running_processes.append((currency_pair, process))
                else:
                    logger.warning(f"{currency_pair} 实时交易进程已退出，返回码: {process.returncode}")
            
            if not running_processes:
                logger.info("所有实时交易进程已退出")
                break
                
            time.sleep(10)  # 每10秒检查一次
            
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号，停止监控...")
            # 终止所有进程
            for currency_pair, process in processes:
                try:
                    process.terminate()
                    logger.info(f"已终止 {currency_pair} 实时交易进程")
                except Exception as e:
                    logger.error(f"终止 {currency_pair} 实时交易进程时出错: {str(e)}")
            break

def main():
    """
    主函数
    """
    try:
        # 启动所有实时交易
        started_processes, failed_starts = start_all_traders()
        
        if started_processes:
            logger.info("所有实时交易已启动，开始监控进程...")
            # 监控运行中的进程
            monitor_processes(started_processes)
        else:
            logger.error("没有成功启动任何实时交易进程")
            
    except Exception as e:
        logger.error(f"启动全部实时交易过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
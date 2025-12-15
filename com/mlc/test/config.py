# 交易系统配置文件

# MT5配置
MT5_CONFIG = {
    "server": "MetaQuotes-Demo",  # MT5服务器
    "login": 123456,              # 登录账号
    "password": "password",       # 密码
    "path": ""                    # MT5安装路径（如果需要）
}

# 交易配置
TRADING_CONFIG = {
    "symbol": "XAUUSD",           # 交易品种
    "timeframe": "TIMEFRAME_M15", # 时间周期
    "lot_size": 1.0,              # 固定手数
    "max_positions": 1,           # 最大持仓数
    "stop_loss_pips": 600,        # 止损点数（对应600美元）
    "take_profit_pips": 1200      # 止盈点数
}

# 模型配置
MODEL_CONFIG = {
    "model_save_path": "trained_model.pkl",  # 模型保存路径
    "confidence_threshold": 0.55,            # 置信度阈值
    "evolution_generations": 100             # 最大进化代数
}

# 数据配置
DATA_CONFIG = {
    "start_year": 2025,           # 数据开始年份
    "end_year": 2025,             # 数据结束年份
    "training_split": 0.8         # 训练集比例
}

# 回测配置
BACKTEST_CONFIG = {
    "initial_balance": 100000,    # 初始资金
    "commission_per_trade": 0,    # 每笔交易手续费
    "slippage_points": 0          # 滑点点数
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",              # 日志级别
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}
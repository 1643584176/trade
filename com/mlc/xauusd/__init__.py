import datetime
import pytz

# 定义北京时间时区（东八区，Asia/Shanghai 是标准的北京时间时区标识）
beijing_timezone = pytz.timezone('Asia/Shanghai')
# 获取当前北京时间（精确到微秒，后续格式化到秒）
current_beijing_time = datetime.datetime.now(beijing_timezone)
# 格式化输出：年-月-日 时:分:秒
formatted_time = current_beijing_time.strftime('%Y-%m-%d %H:%M:%S')
print(f"当前北京时间（精确到秒）：{formatted_time}")
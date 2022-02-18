# 处理窗口
import numpy as np

# 字符串不足时追加参数再截取,使得文本更加整齐
def add_append_str(s,begin,end,append_s='0'):
    s += end * append_s
    return s[begin:end]

# 修正概率,合一
def choice(arr):
    arr = arr + abs(arr.min())
    sum_ = arr.sum()
    for i in range(len(arr)):
        v = arr[i]
        arr[i] = v / sum_
    return np.random.choice(len(arr), p=arr)

# 追加日志
def appand_log(path, log_str):
    with open(path, "a+", encoding="utf-8") as log_file:
        log_file.write(log_str)

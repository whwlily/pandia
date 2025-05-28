import numpy as np
from collections import deque

def index_of(value, array):
    return array.index(min(array, key=lambda x:abs(x - value)))


def sample(val):
    if type(val) is list:
        return np.random.uniform(val[0], val[1])
    else:
        return val


def deep_update(d1, d2):
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_update(d1[k], v)
        else:
            d1[k] = v
    return d1


def divide(a: np.ndarray, b: np.ndarray):
    return np.divide(a.astype(np.float32), b.astype(np.float32), 
                     out=np.zeros_like(a, dtype=np.float32), where=b != 0)


class SlidingWindowQueue:
    def __init__(self, max_len=50, alpha=0.9):
        self.queue = deque([], maxlen=50)
        self.alpha = alpha
        self.smooth_values = []

    def add_value(self, value):
        """
        向队列中添加值并更新平滑值
        :param value: 新增值
        """
        self.queue.append(value)
        if len(self.smooth_values) == 0:
            # 初始化第一个平滑值
            self.smooth_values.append(value)
        else:
            # 计算指数平滑值
            new_smooth_value = self.alpha * value + (1 - self.alpha) * self.smooth_values[-1]
            self.smooth_values.append(new_smooth_value)

    def get_smoothed_values(self):
        """
        获取当前所有的平滑值
        :return: 平滑值列表
        """
        return list(self.smooth_values)[-1]
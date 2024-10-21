from collections import deque
from typing import Tuple, Union

from pandia.context.frame_context import FrameContext
from pandia.context.packet_context import PacketContext

import math
import statistics

class MonitorBlockData(object):
    def __init__(self, ts_fn, val_fn, duration=1, val_checker=lambda v: v >= 0, ts_offset=0) -> None:
        self.duration = duration
        self.data = deque()
        self.sum = 0
        self.ts_fn = ts_fn
        self.val_fn = val_fn
        self.val_checker = val_checker
        self.ts_offset = ts_offset
        self.mini = 3

    def ts(self, val: Union[FrameContext, PacketContext]):
        return self.ts_fn(val) - self.ts_offset

    def append(self, val: Union[FrameContext, PacketContext, Tuple], ts):
        if self.val_checker(self.val_fn(val)):
            self.data.append(val)
            self.sum += self.val_fn(val) 
            self.mini = min(self.mini, self.val_fn(val))
        self.update_ts(ts)

    def non_empty(self):
        return len(self.data) > 0

    @property
    def num(self):
        return len(self.data)

    def avg(self, default_value=0):
        if self.non_empty():
            return self.sum / len(self.data) if len(self.data) > 0 else 0
        return default_value

    def update_ts(self, ts):
        while len(self.data) > 0 and self.ts(self.data[0]) < ts - self.duration:
            val = self.data.popleft()
            self.sum -= self.val_fn(val)
            # 重置mini
            if self.val_fn(val) == self.mini:
                self.mini = 3
                
        if self.mini == 3:
            for new_val in list(self.data):
                self.mini = min(self.mini, self.val_fn(new_val))
    
    # def mini(self):
    #     val_list = []
    #     if self.non_empty():
    #         for val in list(self.data):
    #             if self.val_checker(self.val_fn(val)):
    #                 val_list.append(self.val_fn(val))
    #     return min(val_list) if len(val_list) > 0 else math.inf
    
    @property
    def std(self):
        val_list = []
        if self.non_empty():
            for val in list(self.data):
                if self.val_checker(self.val_fn(val)):
                    val_list.append(self.val_fn(val))
        return statistics.stdev(val_list) if len(val_list) > 1 else 0
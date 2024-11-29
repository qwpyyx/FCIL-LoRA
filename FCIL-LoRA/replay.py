import random
from update import DatasetSplit

class ExperienceReplay:
    def __init__(self, max_size, old_data_ratio=0.3):
        # 缓存的最大大小
        self.max_size = max_size
        # 存储经验的缓冲区
        self.buffer = []
        # 控制每次训练时旧数据的比例
        self.old_data_ratio = old_data_ratio

    def add(self, data):
        # 如果 data 是 DatasetSplit 类型，提取其中的数据样本
        if isinstance(data, DatasetSplit):
            for idx in range(len(data)):
                example = data[idx]  # 获取每个数据样本（字典形式）
                self.buffer.append(example)  # 将字典添加到缓冲区
        else:
            # 如果传入的不是 DatasetSplit 类型，直接添加
            self.buffer.append(data)
        # print(f"Buffer size after adding: {len(self.buffer)}")

    def sample(self, batch_size):
        old_data_size = int(batch_size * self.old_data_ratio)

        # 确保采样的数量不会超过当前缓冲区的大小
        old_data_size = min(old_data_size, len(self.buffer))

        # 打印调试信息，查看采样数量和缓冲区大小
        # print(f"Sampling {old_data_size} items from buffer of size {len(self.buffer)}")

        if old_data_size == 0:
            raise ValueError("Cannot sample zero data points.")

        # 从缓冲区中随机采样数据字典（input_ids, attention_mask, label）
        old_data = random.sample(self.buffer, old_data_size)

        # 返回采样的数据
        return old_data

    def size(self):
        return len(self.buffer)
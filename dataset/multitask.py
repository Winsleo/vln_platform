import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler
import numpy as np
import random
from collections import defaultdict

# 1. 创建示例数据集
class TaskDataset(Dataset):
    def __init__(self, task_id, num_samples=100):
        self.task_id = task_id
        self.num_samples = num_samples
        self.data = np.random.randn(num_samples, 10)
        self.labels = np.random.randint(0, 2, num_samples)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'task_id': self.task_id,
            'data': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 2. 创建多个任务的数据集
task_datasets = {
    'task_a': TaskDataset('task_a', 100),
    'task_b': TaskDataset('task_b', 80),
    'task_c': TaskDataset('task_c', 120),
}

# 3. 创建组合数据集
class CombinedDataset(Dataset):
    def __init__(self, task_datasets):
        self.task_datasets = task_datasets
        self.total_samples = sum(len(ds) for ds in task_datasets.values())
        
        # 创建任务到索引范围的映射
        self.task_ranges = {}
        start_idx = 0
        for task_name, dataset in task_datasets.items():
            end_idx = start_idx + len(dataset)
            self.task_ranges[task_name] = (start_idx, end_idx)
            start_idx = end_idx
            
        # 创建全局索引到任务和局部索引的映射
        self.index_mapping = []
        for task_name, dataset in task_datasets.items():
            for i in range(len(dataset)):
                self.index_mapping.append((task_name, i))
                
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        task_name, task_idx = self.index_mapping[idx]
        return self.task_datasets[task_name][task_idx]

# 4. 创建自定义 BatchSampler
class MultiTaskBatchSampler(Sampler):
    def __init__(self, task_datasets, batch_size):
        self.task_datasets = task_datasets
        self.batch_size = batch_size
        
        # 计算每个任务需要的批次数量
        self.task_batch_counts = {}
        self.task_indices = {}
        
        for task_name, dataset in task_datasets.items():
            indices = list(range(len(dataset)))
            random.shuffle(indices)  # 打乱每个任务的索引
            
            # 计算这个任务需要多少个完整批次
            num_batches = len(dataset) // batch_size
            remainder = len(dataset) % batch_size
            
            # 存储每个任务的批次索引
            task_batches = []
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                task_batches.append(indices[start:end])
            
            # 处理剩余样本
            if remainder > 0:
                task_batches.append(indices[-remainder:])
            
            self.task_batch_counts[task_name] = len(task_batches)
            self.task_indices[task_name] = task_batches
        
        # 创建批次的调度计划
        self.batch_plan = []
        max_batches = max(self.task_batch_counts.values())
        
        # 为每个批次位置轮流选择任务
        for batch_idx in range(max_batches):
            for task_name in task_datasets.keys():
                if batch_idx < self.task_batch_counts[task_name]:
                    # 添加任务和批次索引
                    self.batch_plan.append((task_name, batch_idx))
        
        # 打乱批次顺序，但保持每个任务内的样本顺序不变
        random.shuffle(self.batch_plan)
        
    def __iter__(self):
        for task_name, batch_idx in self.batch_plan:
            # 获取该任务的这个批次索引
            batch_indices = self.task_indices[task_name][batch_idx]
            
            # 转换为全局索引
            start_idx = sum(len(self.task_datasets[name]) for name in list(self.task_datasets.keys())[:list(self.task_datasets.keys()).index(task_name)])
            global_indices = [start_idx + idx for idx in batch_indices]
            
            yield global_indices
            
    def __len__(self):
        return len(self.batch_plan)

# 5. 创建数据加载器
combined_dataset = CombinedDataset(task_datasets)
batch_sampler = MultiTaskBatchSampler(task_datasets, batch_size=16)

dataloader = DataLoader(
    combined_dataset,
    batch_sampler=batch_sampler,
    num_workers=2
)

# 6. 验证每个epoch是否遍历所有数据
all_samples = set()
for i, batch in enumerate(dataloader):
    # 记录这个批次中的所有样本
    batch_size = len(batch['data'])
    task_name = batch['task_id'][0]
    
    # 获取这个批次对应的全局索引
    start_idx = sum(len(task_datasets[name]) for name in list(task_datasets.keys())[:list(task_datasets.keys()).index(task_name)])
    local_indices = batch_sampler.task_indices[task_name][batch_sampler.batch_plan[i][1]]
    global_indices = [start_idx + idx for idx in local_indices]
    
    all_samples.update(global_indices)
    
    print(f"Batch {i}: Task {task_name}, Size {batch_size}")
    
    if i >= len(batch_sampler) - 1:
        break

print(f"\nTotal samples in dataset: {len(combined_dataset)}")
print(f"Unique samples visited: {len(all_samples)}")
print(f"All samples visited: {len(all_samples) == len(combined_dataset)}")
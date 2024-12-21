import math
import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size_list):
        self.dataset = dataset
        self.batch_size_list = batch_size_list
        self.number_of_datasets = len(dataset.datasets)  # 多少个数据集
        self.dataset_size = [len(cur_dataset) for cur_dataset in dataset.datasets]  # 数据集数量
        self.largest_dataset_size = max(self.dataset_size)  # 数据集中最大样本数

    def __len__(self):
        # 一个epoch内总共有多少样本（这里用最大的数据集，较小的数据集迭代完则进行重新采样）
        self.epoch_sample = sum(self.batch_size_list) * math.ceil(self.largest_dataset_size / self.batch_size_list[self.dataset_size.index(self.largest_dataset_size)])
        return self.epoch_sample

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        # 对每个数据集操作，获取具体数据（sampler）和数据迭代器（sampler_iterator）
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        # 这里是获取数据集合并后，每个数据集起始的索引值。假如三个数据集分别为10、50、40，则cumulative_sizes是一个[10,60,100]的列表(按照数据集的数量累计递增)，然后取每个数据集起始的那个索引，因此为[0,10,60]
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]

        step = sum(self.batch_size_list)  # 每次索引的步长
        epoch_samples = self.epoch_sample  # 有多少样本

        final_samples_list = []  # 存放数据集合并后的所有索引列表
        for __ in range(0, epoch_samples, step):
            # 从每个数据集里抽取
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]  # 获取当前数据集的sample-iterators
                cur_samples = []  # 当前数据集的sample

                # 获取当前step的样本
                samples_to_grab = self.batch_size_list[i]  # 该数据集要抓取的样本数
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()  # 获取到当前数据集一个样本的索引值
                        cur_sample = cur_sample_org + push_index_val[i]  # 加上当前数据集的起始索引值，得到最终的索引值
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # 到达迭代器的末尾-重新启动迭代器并继续获取该数据集的样本
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


if __name__ == '__main__':

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self,number,data_val,label_val):
            self.samples = torch.cat((torch.ones(number) * data_val, torch.ones(number) * -data_val))
            self.label = torch.zeros_like(self.samples) + label_val

        def __getitem__(self, index):
            labels_out = torch.zeros((1,2))
            labels_out[:,1] = self.label[index]
            return self.samples[index],labels_out

        def __len__(self):
            return self.samples.shape[0]

        @staticmethod
        def collate_fn(batch):
            batch_size=[3,10,5]
            index=[x for y in batch_size for x in range(y)]
            img, label = zip(*batch)
            for i, lb in enumerate(label):
                lb[:, 0]=index[i]  # 为label添加图像索引
            return torch.stack(img, 0), torch.cat(label, 0)

    batch_size = [3,10,5]  # 每个batch-szie里的数据按照定义的数量到对应的数据集中获取

    first_dataset = MyDataset(5,6,1)
    print("first_dataset",len(first_dataset))
    print("first_dataset: ",first_dataset.samples)
    second_dataset = MyDataset(30,8,2)
    print("second_dataset",len(second_dataset))
    print("second_dataset: ",second_dataset.samples)
    third_dataset = MyDataset(20,10,3)
    print("third_dataset",len(third_dataset))
    print("third_dataset: ",third_dataset.samples)
    concat_dataset = ConcatDataset([first_dataset, second_dataset, third_dataset])
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                             sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                           batch_size_list=batch_size),
                                             num_workers=0,
                                             batch_size=sum(batch_size),
                                             shuffle=False,
                                             collate_fn=MyDataset.collate_fn)
    print("----------------------------------")
    print("dataloader number:",len(dataloader))
    for inputs,labels in dataloader:
        print(inputs)
        print(labels)
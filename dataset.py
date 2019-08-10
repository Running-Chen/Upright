'''
	ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''
import os
import os.path
import json
import numpy as np
import sys
import re


class UoDataset():
    def __init__(self,
                 data_path,
                 batch_size=8,
                 num_point=2048,
                 split='train',
                 cache_size=20000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_point = num_point
        self.shape_cat = {}
        self.shape_cat['train'] = [
            line.rstrip().split(' ') for line in open(
                os.path.join(self.data_path,
                             'datalist/off_10C_100O_train_list.txt'))
        ]
        self.shape_cat['test'] = [
            line.rstrip().split(' ') for line in open(
                os.path.join(self.data_path, 'datalist/uonet_test.txt'))
        ]
        assert (split == 'train' or split == 'test')

        self.datapath = [
            os.path.join(self.data_path, 'rotate_10C_100O_t', x[0]) + '.pts'
            for x in self.shape_cat[split]
        ]
        ################
        #self.uopath = [os.path.join(self.data_path, 'upright', x[0][:-6]) for x in self.shape_cat[split]]

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (pc, cls) tuple

    def __getitem__(self, index):
        pc_tmp = np.loadtxt(self.datapath[index],
                            delimiter=' ').astype(np.float32)
        pc = pc_tmp[:2048]
        uo = pc_tmp[2048]
        assert (len(uo) == 3)

        return pc, uo

    def __len__(self):
        return len(self.datapath)

    def get_batch(self, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_pc = np.zeros((bsize, self.num_point, 3))
        batch_uo = np.zeros((bsize, 3), dtype=np.float32)
        for i in range(bsize):
            pc, uo = self.__getitem__(idxs[i + start_idx])
            batch_pc[i, :, :] = pc

            batch_uo[i] = uo
            
        return batch_pc, batch_uo

    def get_name(self, idxs):
        # use to get corresponding names in testing
        return self.shape_cat['test'][idxs][0]


if __name__ == '__main__':
    d = UoDataset(
        data_path=os.path.join('data'),
        split='train')  # os.path.join('..','data','Motion'),split='train')
    print(len(d))

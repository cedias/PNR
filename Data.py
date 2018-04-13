from random import choice
from torch.utils.data.sampler import Sampler

class BucketSampler(Sampler):
    """
    Evenly sample from bucket for datalen
    """

    def __init__(self, dataset,field):
        self.dataset = dataset
        self.field = field
        self.index_buckets = self._build_index_buckets()
        self.len = min([len(x) for x in self.index_buckets.values()])//2

    def __iter__(self):
        return iter(self.bucket_iterator())

    def __len__(self):
        return self.len

    def bucket_iterator(self):
        cl = list(self.index_buckets.keys())
   
        for x in range(len(self)):
            yield choice(self.index_buckets[choice(cl)])

            
    def _build_index_buckets(self):
        class_index = {}
        for ind,t in enumerate(self.dataset):
            cl = t[self.field]
            if cl not in class_index:
                class_index[cl] = [ind]
            else:
                class_index[cl].append(ind)
        return class_index
        


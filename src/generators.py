# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import numpy as np
import h5py

class Generator_Group_Sampler:
    '''A generator that will sample uniformly across list of generators until ALL generators are exhausted.'''
    def __init__(self, l_gens):
        self.l_gens = l_gens
        self.l_iter = None
        self.len = len(self)
        if hasattr(self.l_gens[0], 'prng'):
            self.prng = self.l_gens[0].prng
        if hasattr(self.l_gens[0], 'shuffle'):
            self.shuffle = self.l_gens[0].shuffle
    def __len__(self):
        '''Generator expects that the passed generators all implement __len__'''
        return np.sum(len(gen) for gen in self.l_gens)
    def __iter__(self):
        '''Generator expects that the passed generators all implement __iter__'''
        self.l_iter = [iter(gen) for gen in self.l_gens]
        len_inds = [len(gen) for gen in self.l_gens]
        len_inds_max = np.max(len_inds)
        for i in range(len_inds_max):
            for i_gen, it in enumerate(self.l_iter):
                if i < len_inds[i_gen]:
                    s = next(it)
                    yield s
                    
class Generator_Paired_Input_Output:
    '''Samples (in_key, out_key) sample pairs from fname_h5 and applies transforms on in_key and transforms_target on out_key'''
    def __init__(self, fname_h5, in_key, out_key, inds=None, transforms=None, transforms_target=None, **kwargs):
        self.fname_h5 = fname_h5
        self.inds = inds
        self.in_key = in_key
        self.out_key = out_key
        self.transforms = transforms
        self.transforms_target = transforms_target
        self.prng = kwargs.get('prng', np.random.RandomState(42))
        self.shuffle = kwargs.get('shuffle', False)
        self.len = None #will be overwritten in check_data()
        self.check_data()

    def check_data(self,):
        len = None
        with h5py.File(self.fname_h5, 'r') as fh:
            for k in [self.in_key, self.out_key]:
                if len is None: 
                    len = fh[k].shape[0]
                if len != fh[k].shape[0]:
                    raise AssertionError('Length of datasets vary across keys. %d vs %d' % (len, fh[k].shape[0]))
        self.len = len
        if self.inds == None:
            self.inds = np.arange(len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx): 
        with h5py.File(self.fname_h5, 'r') as fh:
            x = fh[self.in_key][idx,...]
            if self.transforms is not None:
                x = self.transforms(x)

            y = fh[self.out_key][idx,...]
            if self.transforms_target is not None:
                y = self.transforms_target(y)
        return (x,y)

    def __iter__(self):
        inds = np.arange(self.len)
        if self.shuffle:
            self.prng.shuffle(inds)
        for i in inds:
            s = self.__getitem__(idx=i)
            yield s
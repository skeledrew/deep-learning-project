#! /usr/bin/env python3

'''

'''

from os.path import exists

import tflearn as tfl

from common import *


class DataSet():

    def __init__(self):
        self._ds = None

    def load_data(self, ds):
        pdb.set_trace()
        if ds['name'] == 'mnist':
            from tflearn.datasets import mnist
            self._X, self._Y, self._test_X, self._test_Y = mnist.load_data(one_hot=ds['one_hot'])
            del mnist
            if 'reshape' in ds: self.reshape(ds['reshape'])
        return self

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def test_X(self):
        return self._test_X

    @property
    def test_Y(self):
        return self._test_Y

    def reshape(self, shape):
        '''Reshape the data.
        '''
        if not hasattr(self._X, 'reshape') or not hasattr(self._test_X, 'reshape'): return False
        self._X = self._X.reshape(shape)
        self._test_X = self._test_X.reshape(shape)
        return True

def make_net(arch, mods=None, preprocess=None, augument=None):
    ''' Create DNN architecture.
    '''
    if not mods: raise ValueError('No modules given.')
    net = None
    colls = [list, dict, tuple]
    pdb.set_trace()

    for idx, layer in enumerate(arch):
        func_name = layer.pop(0)
        func = None
        args = kwargs = None

        for m in mods:
            # search the modules for the function
            if not hasattr(m, func_name): continue
            func = getattr(m, func_name)
            break
        if not func: raise ValueError('Unable to find "%s" in any of the given modules.' % func_name)

        if len(layer) in [1, 2]:
            # args and kwargs grouped separately
            args = layer.pop(0) if type(layer[0]) in [list, tuple] else []
            kwargs = layer.pop(0) if isinstance(layer[0], dict) else {}

        else:
            # flat list of all args TODO: implement
            pass
            args = [arg for arg in layer if not type(arg) in colls]
            kwargs = {key: value for key, value in zip()}
        if idx > 0: args.insert(0, net)
        net = func(**kwargs) if not args else func(*args) if not kwargs else func(*args, **kwargs)
    return net

def train(**kwargs):
    '''Train a classifier.
    '''
    ds = DataSet().load_data(kwargs.get('dataset'))
    net = make_net(kwargs.get('net_arch', []), mods=kwargs.get('mods', []))
    model = tfl.DNN(net, tensorboard_verbose=kwargs.get('tb_verbose', 0))
    model.fit(ds.X, ds.Y)
    return model

def main(config=None):
    '''Setup training session.
    '''
    cfg_default = 'config.yaml'
    if not config and exists(cfg_default) or exists(config): config = load_yaml(config or cfg_default)
    tfl_layers = [getattr(tfl.layers, mod) for mod in tfl.layers.__dict__ if mod in ['core', 'conv', 'estimator', 'normalization', 'recurrent']]
    mods = tfl_layers
    model = train(net_arch=config['net_arch'], mods=mods, tb_verbose=1, dataset=config['dataset'])
    pdb.set_trace()

if __name__ == '__main__':
    main()

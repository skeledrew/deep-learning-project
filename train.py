#! /usr/bin/env python3

'''

'''

from os.path import exists

import tflearn as tfl

from common import *
import custom


class DataSet():

    def __init__(self):
        self._ds = None

    def load_data(self, ds):
        _ds = None
        if ds['name'] == 'mnist':
            from tflearn.datasets import mnist as _ds
            self._X, self._Y, self._test_X, self._test_Y = _ds.load_data(one_hot=ds.get('one_hot', False))

        if ds['name'] == 'cifar10':
            from tflearn.datasets import cifar10 as _ds
            (self._X, self._Y), (self._test_X, self._test_Y) = _ds.load_data(one_hot=ds.get('one_hot', False))
        from tflearn.data_utils import shuffle, to_categorical
        del _ds  # discard
        if 'reshape' in ds: self.reshape(ds['reshape'])
        if ds.get('shuffle', False): self._X, self._Y = shuffle(self._X, self._Y)

        if ds.get('to_categorical', False):
            self._Y = to_categorical(self._Y, None)
            self._test_Y = to_categorical(self._test_Y, None)
        return self

    def _load_data_from_builtin(self, ds):
        pass

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
    TODO: implement non-seq net capability
    '''
    if not mods: raise ValueError('No modules given.')
    net = None
    colls = [list, dict, tuple]

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
            args = layer[0] if type(layer[0]) in [list, tuple] else []
            kwargs = layer[-1] if isinstance(layer[-1], dict) else {}

        else:
            # flat list of all args TODO: implement
            pass
            args = [arg for arg in layer if not type(arg) in colls]
            kwargs = {key: value for key, value in zip()}
        if idx > 0: args.insert(0, net)
        #if func_name == 'custom_layer': args[1] = getattr(custom, args[1])
        net = func(**kwargs) if not args else func(*args) if not kwargs else func(*args, **kwargs)
    return net

def train(**kwargs):
    '''Train a neural network.
    '''
    ds_params = kwargs.get('dataset_p')
    rest = kwargs.get('model_p')
    ds = None

    try:
        ds = DataSet().load_data(ds_params)

    except Exception as e:
        print('Failed to load dataset: %s' % repr(e))
        pdb.post_mortem()
    net = None
    mcb = custom.MultiCallback()

    try:
        net = make_net(kwargs.get('net_arch', []), mods=kwargs.get('mods', []))

    except Exception as e:
        print('NN creation failed: %s' % repr(e))
        pdb.post_mortem()
    model = tfl.DNN(net, tensorboard_verbose=kwargs.get('tb_verbose', 0))
    model.fit({'input': ds.X}, {'target': ds.Y}, validation_set=({'input': ds.test_X}, {'target': ds.test_Y}), callbacks=mcb, **rest)
    return model

def main(config=None):
    '''Setup training session.
    '''
    cfg_default = 'config.yaml'
    if (not config and exists(cfg_default)) or exists(config): config = load_yaml(config or cfg_default)
    tfl_layers = [getattr(tfl.layers, mod) for mod in tfl.layers.__dict__ if mod in ['core', 'conv', 'estimator', 'normalization', 'recurrent', 'merge_ops']]
    mods = tfl_layers
    model = train(net_arch=config['net_arch'], mods=mods, tb_verbose=3, dataset_p=config['dataset'], model_p=config['model'])
    if 'model_save' in config['options']: model.save(config['options']['model_save'])

if __name__ == '__main__':
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg)

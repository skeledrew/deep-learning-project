'''Custom stuff.'''


import pdb

import tensorflow as tf
import tflearn as tfl


class MultiCallback(tfl.callbacks.Callback):

    def __init__(self, model, **kwargs):
        self._model = model
        self._kwargs = kwargs
        self._cbs = {}
        if 'callbacks' in kwargs: self._cbs = kwargs['callbacks']
        self._funcs = {name[3:]: func for name, func in zip(globals().keys(), globals().values()) if name.startswith('cb_')}

    def on_train_begin(self, training_state):
        pass

    def on_epoch_begin(self, training_state):
        #set_trace(training_state)
        try:
            if self._cbs.get('on_epoch_begin', False): [self._funcs[f_name](training_state, model=self._model) for f_name in self._cbs['on_epoch_begin']]

        except Exception as e:
            print('Error running an on_epoch_begin callback: %s' % repr(e))
        print('"on_epoch_begin" callback triggered')
        return

    def on_batch_begin(self, training_state):
        pass

    def on_sub_batch_begin(self, training_state):
        pass

    def on_sub_batch_end(self, training_state, train_index=0):
        pass

    def on_batch_end(self, training_state, snapshot=False):
        pass

    def on_epoch_end(self, training_state):
        pass

    def on_train_end(self, training_state):
        pass

def cb_set_trace(state, **kwargs):
    '''Callback to break into the code'''
    pdb.set_trace()
    return state

def tfl_sgd(incoming, **kwargs):
    '''Create an SGD optimizer via custom_layer
    TODO: pass in and get ref name
    '''
    by_ref = kwargs.get('<<by_ref>>', False)
    if '<<by_ref>>' in kwargs: del kwargs['<<by_ref>>']
    by_ref['sgd'] = tfl.SGD(**kwargs)
    return incoming

def tfl_metrics_top_k(incoming, **kwargs):
    '''Create top_k metric via custom_layer'''
    by_ref = kwargs.get('<<by_ref>>', False)
    if '<<by_ref>>' in kwargs: del kwargs['<<by_ref>>']
    by_ref['top_k'] = tfl.metrics.Top_k(**kwargs)
    return incoming

def view_filters(incoming, **kwargs):
    '''Write filters in Tensor to image summary.'''
    prev_vars = tfl.variables.get_layer_variables_by_name(kwargs.get('prev_layer_name'))
    filter = None  # get from collection
    filt_summ = tf.summary.image(kwargs.get('name', 'view_filter'), filter)
    # setup for writer adding in callback
    return incoming

def cb_view_filters(state, **kwargs):
    '''Callback to trigger filter dump for viewing.'''
    m = kwargs.get('model')
    if not m: return
    graph_data = m.session.graph._collections
    pdb.set_trace()
    return state

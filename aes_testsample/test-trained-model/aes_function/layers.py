import keras.backend as K
import numpy as np
from keras.engine.topology import Layer
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import _Merge


class Conv1DMask(Conv1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DMask, self).__init__(**kwargs)

    def compute_mask(self, x, mask):
        return mask


class GatePositional(_Merge):
    def __init__(self, bias=0, **kwargs):
        self.supports_masking = True
        self.bias = bias
        super(GatePositional, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape
        self.v = K.variable(
            (np.random.rand(input_shape[0][1], 50) * 0.2 -
             0.1).astype('float32'),
            name='v')
        self.b = K.variable(value=self.bias, dtype='float32', name='b')
        self.trainable_weights = [self.v, self.b]

    def call(self, inputs, mask=None):
        char = inputs[0]
        word = inputs[1]
        g = K.sigmoid(word * self.v + self.b)
        return (1. - g) * word + g * char

    def compute_mask(self, input, mask):
        return None

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class MaxPooling1DMask(MaxPooling1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaxPooling1DMask, self).__init__(**kwargs)

    def compute_mask(self, x, mask):
        return mask


class MeanOverTime(Layer):
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.cast(
            x.sum(axis=1) / mask.sum(axis=1, keepdims=True),
            K.floatx()) if self.mask_zero and mask is not None else K.mean(
                x, axis=1)

    def compute_mask(self, x, mask):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = {'mask_zero': self.mask_zero}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

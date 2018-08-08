
import tensorflow as tf
from keras.layers import Layer
import keras.backend as kb

def _pixel_norm(x, epsilon=1e-8, channel_axis=-1):
  with tf.variable_scope('PixelNorm'):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=channel_axis, keepdims=True) + epsilon)

class PixelNorm(Layer):
  def __init__(self, channel_axis=-1, **kwargs):
    self.channel_axis = channel_axis
    super().__init__() 

  def call(self, x):
    return _pixel_norm(x, channel_axis=self.channel_axis)

  def compute_output_shape(self, input_shape):
    return input_shape
  
  def get_config(self):
    return {
        'channel_axis': self.channel_axis,
        **super().get_config()
    }
  

def _upscale2d(x, factor=2):
  # Channels last upscale
  assert isinstance(factor, int) and factor >= 1
  if factor == 1: return x
  with tf.variable_scope('Upscale2D'):
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, 1, factor, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x
  
class Upscale2D(Layer):

  def call(self, x):
    return _upscale2d(x)

  def compute_output_shape(self, input_shape):
    batch_size, h, w, c = input_shape
    output_shape = [batch_size, h*2, w*2, c]
    return tuple(output_shape)
  

class ToChannelsLast(Layer):

  def call(self, x):
    return kb.permute_dimensions(x, [0, 2, 3, 1])

  def compute_output_shape(self, input_shape):
    batch_size, c, h, w = input_shape
    output_shape = [batch_size, h, w, c]
    return tuple(output_shape)
    
custom_objects = {
  'Upscale2D': Upscale2D,
  'PixelNorm': PixelNorm,
  'ToChannelsLast': ToChannelsLast,
}
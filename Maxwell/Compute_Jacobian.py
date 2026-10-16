from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gradients_impl as gradient_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.util import nest

def jacobian(output, inputs, use_pfor=True, parallel_iterations=None):

  flat_inputs = nest.flatten(inputs)
  output_tensor_shape = output.shape
  output_shape = array_ops.shape(output)
  output = array_ops.reshape(output, [-1])

  def loop_fn(i):
    y = array_ops.gather(output, i)
    return gradient_ops.gradients(y, flat_inputs,  unconnected_gradients=tf.UnconnectedGradients.ZERO)

  try:
    output_size = int(output.shape[0])
  except TypeError:
    output_size = array_ops.shape(output)[0]

  if use_pfor:
    pfor_outputs = control_flow_ops.pfor(
        loop_fn, output_size, parallel_iterations=parallel_iterations)
  else:
    pfor_outputs = control_flow_ops.for_loop(
        loop_fn,
        [output.dtype] * len(flat_inputs),
        output_size,
        parallel_iterations=parallel_iterations)

  for i, out in enumerate(pfor_outputs):
    if isinstance(out, ops.Tensor):
      new_shape = array_ops.concat(
          [output_shape, array_ops.shape(out)[1:]], axis=0)
      out = array_ops.reshape(out, new_shape)
      out.set_shape(output_tensor_shape.concatenate(flat_inputs[i].shape))
      pfor_outputs[i] = out

  return nest.pack_sequence_as(inputs, pfor_outputs)
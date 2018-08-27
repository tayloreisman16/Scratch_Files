import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
complex_array = np.array([[1.0-1.0j, 2.0-2.0j], [3.0-3.0j, 4.0j]])

real_tensor = tf.constant([[-9], [12]])
print_real_tensor = tf.Print(real_tensor, [real_tensor], "Real Tensor: ")

# tf.complex(
#     real,
#     imag,
#     name=None
# )

print("Complex Tensors must be defined using tf.complex and take floats as inputs...")
real = tf.constant([1., -1., 2., -2.])
imag = tf.constant([1., -1., 2., -2.])
complex_tensor = tf.complex(real, imag)

print_complex_tensor = tf.Print(complex_tensor, [complex_tensor], "Corrected Complex Tensor: ")

# tf.abs(
#     x,
#     name=None
# )


absolute_value = tf.abs(complex_tensor)

# tf.angle(
#     input,
#     name=None
# )

phase_value = tf.angle(complex_tensor)

# tf.reshape(
#     tensor,
#     shape,
#     name=None
# )

reshape_complex_tensor = tf.reshape(complex_tensor, [2, 2])

# tf.convert_to_tensor(
#     value,
#     dtype=None,
#     name=None,
#     preferred_dtype=None
# )

array_to_tensor = tf.convert_to_tensor(complex_array, dtype=tf.complex64)

# sess.run(print_real_tensor)
# sess.run(print_complex_tensor)
print("Complex Tensor: ", complex_tensor.eval())
print("Magnitude of Complex Tensor: ", absolute_value.eval())
print("Phase of Complex Tensor: ", phase_value.eval())
print("Reshape of Complex Tensor: ", reshape_complex_tensor.eval())
print("Complex Numpy Array to Complex Tensor: ", array_to_tensor.eval())


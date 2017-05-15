import TensorFlowCode as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = tf.add(a, b)

sess = tf.Session()
binding = {a:11, b:22}
c = sess.run(add, feed_dict = binding)
print(c)


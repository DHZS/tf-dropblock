# DropBlock in TensorFlow

This is a TensorFlow implementation of the following paper:

>DropBlock: A regularization method for convolutional networks  
>arXiv. https://arxiv.org/abs/1810.12890


## Usage

Graph Execution
```python
import numpy as np
import tensorflow as tf
from nets.dropblock import DropBlock

# only support `channels_last` data format
a = tf.placeholder(tf.float32, [None, 10, 10, 3])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

drop_block = DropBlock(keep_prob=keep_prob, block_size=3)
b = drop_block(a, training)

sess = tf.Session()
feed_dict = {a: np.ones([2, 10, 10, 3]), keep_prob: 0.8, training: True}
c = sess.run(b, feed_dict=feed_dict)

print(c[0, :, :, 0])
```


Eager Execution
```python
import tensorflow as tf
from nets.dropblock import DropBlock

tf.enable_eager_execution()

# only support `channels_last` data format
a = tf.ones([2, 10, 10, 3])

drop_block = DropBlock(keep_prob=0.8, block_size=3)
b = drop_block(a, training=True)

print(b[0, :, :, 0])

# update keep probability
drop_block.set_keep_prob(0.1)
b = drop_block(a, training=True)

print(b[0, :, :, 0])
```

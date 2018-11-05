#### TF基础类型及操作函数
##### tf.train.Saver()
  - 构造<tf.train.Saver>的时候，也可以自己手动填入要保存的变量
  ```python
  w = tf.Variable(4, name='w_op')
  saver = tf.train.Saver({v.op.name: v for v in [w]})
  ```
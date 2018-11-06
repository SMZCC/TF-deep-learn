#### TF基础类型及操作函数
##### tf.train.Saver()
  - 构造`<tf.train.Saver>`的时候，也可以自己手动填入要保存的**变量**,构造函数中的**`max_to_keep`**可用来指明最多保存几个模型
  - 调用`saver.save()`方法时，要指明是保存哪个**会话(sess)**的内容，以及**保存的路径**和**[global_step]**
  ```python
  w = tf.Variable(4, name='w_op')
  # 构造
  saver = tf.train.Saver({v.op.name: v for v in [w]}, max_to_keep=10)
  save_path = "./save/xxx.ckpt"
  # 调用
  with tf.Session() as sess:
    saver.save(sess, save_path)
  ```

##### tf.summary.FileWriter()
  - 在sess中构造,构造时，需指定文件的写入位置，以及需要写入的图(PS:如果能直接得到默认的图的话，应该也可以在sess外构造吧)
  - 使用`merged_summary_op = tf.summary.merge_all()`合并所有的记录,然后通过`sess.run([merged_summary_op])`来获得对应的字符串
  - 使用`tf_writer.add_summary()`将`summary`写入文件
  ```python
  write_dir = "./logs"
  merged_summary_op = tf.summary.merge_all()
  # 构造
  with tf.Session() as sess:
    summary_str = sess.run([merged_summay_op])
    tf_writer = tf.summary.FileWriter(write_dir, sess.graph)
    tf_writer.add_summary(summary_str, global_step)
  ```
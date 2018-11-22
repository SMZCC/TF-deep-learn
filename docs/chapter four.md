#### TF基础类型及操作函数
##### tf.train.Saver()
  - 构造`<tf.train.Saver>`的时候，也可以自己手动填入要保存的**变量**,构造函数中的`max_to_keep`可用来指明最多保存几个模型
  - 调用`saver.save()`方法时，要指明是保存哪个**会话(sess)**的内容，以及**保存的路径**和[global_step]
    + **saver.save()中的[global_step]可以是一个`<Variable>`**,也可以是一个`<string>`, `<bytes-like>`,或直接一个数
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
    + 已尝试，可在sess外构造，可以使用`tf.get_defaut_graph()`将默认图获取到，然后用于构造
  - 使用`merged_summary_op = tf.summary.merge_all()`合并所有的记录,然后通过`sess.run([merged_summary_op])`来获得对应的字符串
  - 使用`tf_writer.add_summary()`将`summary`写入文件
    + **add_summary()中的`global_step`不可是一个`<Variable>`,需要是一个`<string>`或者`<bytes-like>`，或者一个数**
  ```python
  write_dir = "./logs"
  merged_summary_op = tf.summary.merge_all()
  # 构造
  with tf.Session() as sess:
    summary_str = sess.run([merged_summay_op])
    tf_writer = tf.summary.FileWriter(write_dir, sess.graph)
    tf_writer.add_summary(summary_str, global_step)
  ```

##### tensorboard
  - 在`windows`下这简直是个坑，同样的东西，在`Ubuntu`下显示没有任何问题，在`windows`下怎么都没有数据，原因如下：
  - `windows`下`tensorboard`必须要到`logs`文件夹的上级目录才行，然后直接在该目录下使用`tensorboard --logdir logs/`即可，否则没有数据
  - 例：记录文件保存在`J:/linear/logs`, 命令行中`cd`到`linear`下，然后使用`tensorboard --logdir logs/`
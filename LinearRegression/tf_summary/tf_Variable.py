# coding=utf-8
# date: 2018/12/10, 13:10
# name: smz

import tensorflow as tf

def demo_one():

    g = tf.get_default_graph()


    var1 = tf.Variable(1.0, name='firstvar')
    print("var1:", "var1.name:", var1.name, "var1.op.name:", var1.op.name)

    var1 = tf.Variable(2.0, name='firstvar')
    print("var1:", "var1.name:", var1.name, "var1.op.name:", var1.op.name)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        _ = sess.run([init])
        print("var1 value:", sess.run(var1))

    writer = tf.summary.FileWriter(logdir='./logs', graph=g)
    writer.close()


def demo_tow():
    get_var1 = tf.get_variable("get_var1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.3))
    print("get_var1:", "get_var1.name:", get_var1.name, "get_var1.op.name:", get_var1.op.name)


def demo_three():
    with tf.variable_scope("scope1"):
        with tf.name_scope("scope2"):
            get_var1 = tf.get_variable('firstvar', shape=[1], initializer=tf.constant_initializer(0.1))
            tensor2 = get_var1 + 1.
            tensor3 = tf.add(get_var1, 2.)

            print("get_var1:", get_var1)
            print("get_var1.name:", get_var1.name)
            print("get_var1.op.name:", get_var1.op.name)

            print("tensor2:", tensor2)
            print("tensor2.name:", tensor2.name)
            print("tensor2.op.name:", tensor2.op.name)

            print("tensor3:", tensor3)
            print("tensor3.name:", tensor3.name)
            print("tensor3.op.name:", tensor3.op.name)


def demo_four():
    g = tf.Graph()
    with g.as_default():
        a = tf.constant([[1.0, 2.0]])
        b = tf.constant([[1.0], [2.0]])

        tensor1 = tf.matmul(a, b, name="example_op")
        tensor1_op = g.get_operation_by_name("example_op")

    all_elements = g.get_operations()
    print("tensor1_op:", tensor1_op)
    print("all_elements:", all_elements)


def demo_five():
    g = tf.Graph()
    default_g1 = tf.get_default_graph()
    with g.as_default():
        default_g2 = tf.get_default_graph()

    print("default_g1:", default_g1)
    print("defualt_g2:", default_g2)
    print("g:", g)


if __name__ == "__main__":
    demo_five()
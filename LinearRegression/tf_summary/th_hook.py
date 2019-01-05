# coding=utf-8
# date: 2018/12/20, 14:23
# name: smz


import torch as th
import numpy as np
import torch.autograd.variable as Variable
import torch.nn as nn





def demo_one():
    grads = []

    def hook_grad(grad_in):
        grads.append(grad_in.data)

    # x = Variable(th.from_numpy(np.array([1, 2]).astype(np.float32)), requires_grad=True)
    x = th.tensor([1, 2], dtype=th.float32, requires_grad=True)
    y = x.pow(2) + 1
    z = th.mean(y, dim=0)  # z = 1/2 * y || z = 1/2 * (x1^2 + x2^2)  || dz/dy1 = 0.5, dz/dy2=0.5, dz/dx1 = 1 * x1, dz/dx2 = 1 * x2
    y.register_hook(hook_grad)

    z.backward()

    print("x:{}\n".format(x))
    print("z:{}\n".format(z))
    print("x.grad:{}\n".format(x.grad))
    print("y.grad:{}".format(y.grad))
    print("z.grad:{}\n".format(z.grad))
    print("grads:{}\n".format(grads))


def demo_two():
    x = th.tensor([-1, 0, 1, 2], dtype=th.float32, requires_grad=True)
    x = x.cuda()    # 这里不能放到GPU上，否则算出来的梯度为None
    relu_layer = nn.ReLU()
    res = relu_layer(x)
    loss = th.sum(res, dim=0)
    loss.backward()
    print("x:{}\n".format(x))
    print("x.grad:{}\n".format(x.grad))


if __name__ == "__main__":
    demo_two()
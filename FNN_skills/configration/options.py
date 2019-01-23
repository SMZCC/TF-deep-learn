# coding=utf-8
# date: 2019/1/12, 14:01
# name: smz

from collections import OrderedDict

opts = OrderedDict()
opts["model_name"] = "xor.ckpt"

opts["input_fields"] = 2
opts["label_fields"] = 4
opts["hidden_fields"] = [200]    # 一层隐层

opts["epochs"] = 30     # plan_a 30, plan_b 10000
opts["batch_size"] = 100  # plan_a 100, plan_b 40000， plan_c 100
opts["learning_rate"] = 1e-4

opts["max_to_keep"] = 3
opts["checkpoints_dir"] = "../checkpoints/"
opts["logs_dir"] = "../logs/"
opts["train_x"] = "../data/train_X.npy"
opts["train_y"] = "../data/train_Y.npy"
# coding=utf-8
# date: 2018/11/5, 18:17
# name: smz

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

if __name__ == "__main__":
    save_path = "J:\\TF-deep-learn\\LinearRegression\\saved\\linear_model.ckpt-20000"
    print_tensors_in_checkpoint_file(save_path, None, True)
    # tensor_name值为None, 表示不是打印某个特定的tensor，all_tensors为True，表示打印出所有的tensor值

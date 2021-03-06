import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys

def test_unit_checkpoint_lenet(cluster, exe, dirname):
    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=1,
        dir_name=dirname,
        data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd')
    return_code_nockpt = os.system(command)
    if return_code_nockpt != 0:
        sys.stderr.write('LeNet (no checkpoint) execution failed, exiting with error')
        sys.exit(1)
    os.system('mv ckpt ckpt_baseline')


    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=1,
        dir_name=dirname,
        data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=1, optimizer_name='sgd')   
    return_code_ckpt_1 = os.system(command)
    if return_code_ckpt_1 != 0:
        sys.stderr.write('LeNet (checkpoint) execution failed, exiting with error')
        sys.exit(1)

    command = tools.get_command(
        cluster=cluster, executable=exe, num_nodes=1, num_processes=1,
        dir_name=dirname,
        data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
        data_reader_name='mnist', model_folder='tests',
        model_name='lenet_mnist_ckpt', num_epochs=2, optimizer_name='sgd')
    return_code_ckpt_2 = os.system(command)
    if return_code_ckpt_2 != 0:
        sys.stderr.write('LeNet execution (restart from checkpoint) failed, exiting with error')
        sys.exit(1)

    diff_test = os.system('diff -rq ckpt ckpt_baseline')
    os.system('rm -rf ckpt*')
    assert diff_test == 0

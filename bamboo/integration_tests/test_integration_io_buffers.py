import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, sys
import common_code

def test_integration_partitioned_and_distributed_io_mnist(cluster, dirname, exes):
    max_mb = 300
    # Printing output from 6*6*2=72 runs of LBANN makes the logs too slow.
    # Output from run_lbann is still printed - if there is a failure.
    should_log = False
    partitioned = 'mnist_partitioned_io'
    distributed = 'mnist_distributed_io'
    model_names = [partitioned, distributed]
    accuracies = {}
    errors = []
    all_values = []
    for mini_batch_size in [300, 150, 100, 75, 60, 50]:
        num_models = max_mb / mini_batch_size
        for procs_per_model in [1, 2, 3, 4, 5, 6]:
            num_ranks = procs_per_model * num_models
            for model_name in model_names:
                output_file_name = '%s/bamboo/integration_tests/output/%s_%d_%d_output.txt' % (dirname, model_name, mini_batch_size, procs_per_model)
                command = tools.get_command(
                    cluster=cluster, executable=exes['default'], num_nodes=2,
                    num_processes=num_ranks, dir_name=dirname,
                    data_filedir_ray='/p/gscratchr/brainusr/datasets/MNIST',
                    data_reader_name='mnist', mini_batch_size=mini_batch_size,
                    model_folder='tests', model_name=model_name, num_epochs=5,
                    optimizer_name='adagrad',
                    processes_per_model=procs_per_model,
                    output_file_name=output_file_name)
                common_code.run_lbann(command, model_name, output_file_name, should_log)
                accuracy_dict = common_code.extract_data(output_file_name, ['test_accuracy'], should_log)
                accuracies[model_name] = accuracy_dict['test_accuracy']
            
            partitioned_num_models = len(accuracies[partitioned].keys())
            distributed_num_models = len(accuracies[distributed].keys())
            assert partitioned_num_models == distributed_num_models
            
            for model_num in sorted(accuracies[partitioned].keys()):
                partitioned_accuracy = accuracies[partitioned][model_num]['overall']
                distributed_accuracy = accuracies[distributed][model_num]['overall']
                tolerance = 0.05
                # Are we within tolerance * expected_value?
                if abs(partitioned_accuracy - distributed_accuracy) > abs(tolerance * min(partitioned_accuracy, distributed_accuracy)):
                    errors.append('partitioned = %f != %f = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
                all_values.append('partitioned = %f, %f = distributed; model_num=%s mini_batch_size=%d procs_per_model=%d' % (partitioned_accuracy, distributed_accuracy, model_num, mini_batch_size, procs_per_model))
    
    print('Errors for: partitioned_and_distributed (%d)' % len(errors))
    for error in errors:
        print(error)
    if should_log:
        print('All values: (%d)' % len(all_values))
        for value in all_values:
            print(value)
    assert errors == []

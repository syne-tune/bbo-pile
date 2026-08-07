from benchmark_definitions import BenchmarkDefinition, n_full_evals


def pd1_benchmark(dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=3600000000,
        max_num_evaluations=n_full_evals,
        n_workers=1,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error_rate",
        mode="min",
        blackbox_name='pd1',
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


tasks = ['imagenet_resnet_batch_size_512',
         'uniref50_transformer_batch_size_128',
         'translate_wmt_xformer_translate_batch_size_64',
         'lm1b_transformer_batch_size_2048',
         'imagenet_resnet_batch_size_256',
         'mnist_max_pooling_cnn_tanh_batch_size_2048',
         'mnist_max_pooling_cnn_tanh_batch_size_256',
         'mnist_max_pooling_cnn_relu_batch_size_2048',
         'mnist_max_pooling_cnn_relu_batch_size_256',
         'mnist_simple_cnn_batch_size_2048',
         'mnist_simple_cnn_batch_size_256',
         'fashion_mnist_max_pooling_cnn_tanh_batch_size_2048',
         'fashion_mnist_max_pooling_cnn_tanh_batch_size_256',
         'fashion_mnist_max_pooling_cnn_relu_batch_size_2048',
         'fashion_mnist_max_pooling_cnn_relu_batch_size_256',
         'fashion_mnist_simple_cnn_batch_size_2048',
         'fashion_mnist_simple_cnn_batch_size_256',
         'svhn_no_extra_wide_resnet_batch_size_1024',
         'svhn_no_extra_wide_resnet_batch_size_256',
         'cifar100_wide_resnet_batch_size_2048',
         'cifar100_wide_resnet_batch_size_256',
         'cifar10_wide_resnet_batch_size_2048',
         'cifar10_wide_resnet_batch_size_256']

pd1_benchmark_definitions = {}

for ds in tasks:
    pd1_benchmark_definitions["pd1_" + ds] = pd1_benchmark(ds)

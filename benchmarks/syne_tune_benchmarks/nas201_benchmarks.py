from benchmark_definitions import BenchmarkDefinition, n_full_evals

def nas201_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=72000 if dataset_name == "ImageNet16-120" else 36000,
        max_num_evaluations=n_full_evals,
        n_workers=1,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        use_surrogate=True,
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
    )

nas201_benchmark_definitions = {
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}
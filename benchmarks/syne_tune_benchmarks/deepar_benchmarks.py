from benchmark_definitions import BenchmarkDefinition, n_full_evals


def deepar_benchmark(dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=3600000000,
        max_num_evaluations=n_full_evals,
        n_workers=1,
        elapsed_time_attr="metric_time",
        metric="metric_RMSE",
        mode="min",
        blackbox_name='icml-deepar',
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


tasks = ['m4-Daily',
 'exchange-rate',
 'm4-Yearly',
 'solar',
 'm4-Monthly',
 'electricity',
 'traffic',
 'm4-Quarterly',
 'm4-Weekly',
 'm4-Hourly',
 'wiki-rolling'
]

deepar_benchmark_definitions = {}

for ds in tasks:
    deepar_benchmark_definitions["deepar_" + ds] = deepar_benchmark(ds)

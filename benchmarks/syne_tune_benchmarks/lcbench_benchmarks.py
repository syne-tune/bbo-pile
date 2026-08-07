from benchmark_definitions import BenchmarkDefinition, n_full_evals


def lcbench_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=n_full_evals,
        n_workers=1,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )


lcbench_benchmark_definitions = {}

datasets = ['APSFailure', 'Amazon_employee_access', 'Australian', 'Fashion-MNIST', 'KDDCup09_appetency', 'MiniBooNE', 'adult', 'airlines', 'albert', 'bank-marketing', 'blood-transfusion-service-center', 'car', 'christine', 'cnae-9', 'connect-4', 'covertype', 'credit-g', 'dionis', 'fabert', 'helena', 'higgs', 'jannis', 'jasmine', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kr-vs-kp', 'mfeat-factors', 'nomao', 'numerai28.6', 'phoneme', 'segment', 'shuttle', 'sylvine', 'vehicle', 'volkert']
for ds in list(datasets):
    lcbench_benchmark_definitions["lcbench_" + ds] = lcbench_benchmark(ds)

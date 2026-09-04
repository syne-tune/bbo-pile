from benchmark_definitions import BenchmarkDefinition, n_full_evals


def bbomix_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=72000,
        n_workers=1,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_avg_ml_task_performance",
        mode="max",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        max_num_evaluations=n_full_evals,
    )


bbomix_benchmark_definitions = {}

bbomix_schc_search_spaces = [
    "vanillix_schc",
    "varix_schc",
    "ontix_schc",
    "disentanglix_schc",
]

bbomix_tcga_search_spaces = [
    "vanillix_tcga",
    "varix_tcga",
    "ontix_tcga",
    "disentanglix_tcga",
]

bbomix_schc_tasks = [
    "schc_RNA_METH_CLIN",
    "schc_METH_CLIN",
    "schc_RNA_CLIN",
]

bbomix_tcga_tasks = [
    "tcga_RNA_CLIN",
    "tcga_METH_CLIN",
    "tcga_DNA_CLIN",
    "tcga_RNA_DNA_METH_CLIN",
]

for task in bbomix_schc_tasks:
    for search_space in bbomix_schc_search_spaces:
        if "ontix" in search_space:
            for ontology in ["reactome", "chromosome"]:
                bbomix_benchmark_definitions[
                    f"bbomix-{search_space}-"
                    + task.replace("_", "-").replace(".", "")
                    + "-"
                    + ontology
                ] = bbomix_benchmark(
                    blackbox_name=f"bbomix_{search_space}",
                    dataset_name=task + "_" + ontology,
                )
        else:
            bbomix_benchmark_definitions[
                f"bbomix-{search_space}-" + task.replace("_", "-").replace(".", "")
            ] = bbomix_benchmark(
                blackbox_name=f"bbomix_{search_space}",
                dataset_name=task,
            )

for task in bbomix_tcga_tasks:
    for search_space in bbomix_tcga_search_spaces:
        if "ontix" in search_space:
            for ontology in ["reactome", "chromosome"]:
                bbomix_benchmark_definitions[
                    f"bbomix-{search_space}-"
                    + task.replace("_", "-").replace(".", "")
                    + "-"
                    + ontology
                ] = bbomix_benchmark(
                    blackbox_name=f"bbomix_{search_space}",
                    dataset_name=task + "_" + ontology,
                )
        else:
            bbomix_benchmark_definitions[
                f"bbomix_{search_space}-" + task.replace("_", "-").replace(".", "")
            ] = bbomix_benchmark(
                blackbox_name=f"bbomix_{search_space}",
                dataset_name=task,
            )

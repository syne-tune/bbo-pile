from global_optimization_problems import global_optimization_problem_collection

benchmark_definitions = {
    f"global-optimization_{name}": problem
    for name, problem in global_optimization_problem_collection.items()
}

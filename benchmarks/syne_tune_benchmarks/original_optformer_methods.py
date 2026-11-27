from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)

from other_searchers.original_optformer_searcher import OriginalOptFormerSearcher
from baselines import Methods

original_optformer_methods = {

    Methods.OptFormerBBOB_HillClimb:  lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                   config_space=method_arguments.config_space,
                                   random_seed=method_arguments.random_seed,
                                   designer_name='designer_hill_climb',
                                   model="bbob"
                                   ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerBBOB_GP: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_recursive_gp',
                                           model="bbob"
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerBBOB_REGEVO: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_reg_evo',
                                           model="bbob"
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerBBOB_RS: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_random_search',
                                           model="bbob"
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ),
    Methods.OptFormerHPOB_GP: lambda method_arguments: SingleObjectiveScheduler(
        config_space=method_arguments.config_space,
        searcher=OriginalOptFormerSearcher(points_to_evaluate=method_arguments.points_to_evaluate,
                                           config_space=method_arguments.config_space,
                                           random_seed=method_arguments.random_seed,
                                           designer_name='designer_recursive_gp',
                                           model="hpob"
                                           ),
        metric=method_arguments.metric,
        do_minimize=method_arguments.mode == "min",
        random_seed=method_arguments.random_seed,
    ), 
}
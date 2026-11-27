from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class BenchmarkDefinition:
    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    use_surrogate: Optional[bool] = True
    surrogate_kwargs: Optional[Dict] = None


n_full_evals = 100
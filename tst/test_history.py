import os
import tempfile
import pytest

from syne_tune.experiments import load_experiment

from syne_tune.config_space import randint, uniform, choice, loguniform, lograndint, FiniteRange
from syne_tune.constants import SYNE_TUNE_ENV_FOLDER

from open_optformer.history import History, Trial, Converter, OptformerConverter, preprocess


# ============================================================================
# Converter.value_to_txt() Tests
# ============================================================================

def test_value_to_txt_different_q():
    """Test value_to_txt with different quantization values"""
    for q in [100, 500, 1000, 2000]:
        converter = Converter(q=q)
        assert converter.value_to_txt(0.5, hp=uniform(0, 1)) == str(int(0.5 * q))
        
        optformer_converter = OptformerConverter(q=q)
        assert optformer_converter.value_to_txt(0.5, hp=uniform(0, 1)) == f"<{int(0.5 * q)}>"


def test_value_to_txt_integer_hyperparameters():
    """Test value_to_txt with integer hyperparameters"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        hp = randint(0, 10)
        if isinstance(converter, OptformerConverter):
            assert converter.value_to_txt(0, hp=hp) == "<0>"
            assert converter.value_to_txt(5, hp=hp) == "<500>"
            assert converter.value_to_txt(10, hp=hp) == "<1000>"
        else:
            assert converter.value_to_txt(0, hp=hp) == "0"
            assert converter.value_to_txt(5, hp=hp) == "500"
            assert converter.value_to_txt(10, hp=hp) == "1000"


def test_value_to_txt_categorical():
    """Test value_to_txt with categorical hyperparameters"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        hp = choice(['a', 'b', 'c', 'd'])
        if isinstance(converter, OptformerConverter):
            assert converter.value_to_txt('a', hp=hp) == "<0>"
            assert converter.value_to_txt('b', hp=hp) == "<1>"
            assert converter.value_to_txt('c', hp=hp) == "<2>"
            assert converter.value_to_txt('d', hp=hp) == "<3>"
        else:
            assert converter.value_to_txt('a', hp=hp) == "0"
            assert converter.value_to_txt('b', hp=hp) == "1"
            assert converter.value_to_txt('c', hp=hp) == "2"
            assert converter.value_to_txt('d', hp=hp) == "3"



def test_value_to_txt_metrics():
    """Test value_to_txt for metrics (hp=None, x_min/x_max provided)"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        if isinstance(converter, OptformerConverter):
            assert converter.value_to_txt(0.5, hp=None, x_min=0.0, x_max=1.0) == "<500>"
            assert converter.value_to_txt(0.0, hp=None, x_min=0.0, x_max=1.0) == "<0>"
            assert converter.value_to_txt(1.0, hp=None, x_min=0.0, x_max=1.0) == "<1000>"
        else:
            assert converter.value_to_txt(0.5, hp=None, x_min=0.0, x_max=1.0) == "500"
            assert converter.value_to_txt(0.0, hp=None, x_min=0.0, x_max=1.0) == "0"
            assert converter.value_to_txt(1.0, hp=None, x_min=0.0, x_max=1.0) == "1000"


def test_value_to_txt_metrics_missing_bounds():
    """Test value_to_txt for metrics raises error when x_min/x_max not provided"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        with pytest.raises(ValueError, match="x_min and x_max must be provided"):
            converter.value_to_txt(0.5, hp=None)


def test_value_to_txt_log_scale():
    """Test value_to_txt with log-scale hyperparameters"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        # Test loguniform
        hp = loguniform(1e-3, 1.0)
        # Log scale should handle values correctly
        val = 0.1
        txt = converter.value_to_txt(val, hp=hp)
        assert txt is not None
        # Verify round-trip works
        recovered = converter.txt_to_value(txt, hp=hp)
        assert abs(val - recovered) < 0.01, f"Log scale round-trip failed: {val} -> {txt} -> {recovered}"
        
        # Test lograndint
        hp_int = lograndint(1, 100)
        val_int = 10
        txt_int = converter.value_to_txt(val_int, hp=hp_int)
        assert txt_int is not None
        recovered_int = converter.txt_to_value(txt_int, hp=hp_int)
        assert abs(val_int - recovered_int) < 1, f"Log scale integer round-trip failed: {val_int} -> {txt_int} -> {recovered_int}"


# ============================================================================
# Converter.txt_to_value() Tests
# ============================================================================

def test_txt_to_value_basic():
    """Test txt_to_value with basic conversions - exact values where quantization allows"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        hp = uniform(0, 1)
        if isinstance(converter, OptformerConverter):
            assert converter.txt_to_value("<500>", hp=hp) == 0.5
            assert converter.txt_to_value("<0>", hp=hp) == 0.0
            assert converter.txt_to_value("<1000>", hp=hp) == 1.0
        else:
            assert converter.txt_to_value("500", hp=hp) == 0.5
            assert converter.txt_to_value("0", hp=hp) == 0.0
            assert converter.txt_to_value("1000", hp=hp) == 1.0


def test_txt_to_value_categorical():
    """Test txt_to_value with categorical hyperparameters"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        hp = choice(['a', 'b', 'c', 'd'])
        if isinstance(converter, OptformerConverter):
            assert converter.txt_to_value("<0>", hp=hp) == 'a'
            assert converter.txt_to_value("<1>", hp=hp) == 'b'
            assert converter.txt_to_value("<2>", hp=hp) == 'c'
            assert converter.txt_to_value("<3>", hp=hp) == 'd'
        else:
            assert converter.txt_to_value("0", hp=hp) == 'a'
            assert converter.txt_to_value("1", hp=hp) == 'b'
            assert converter.txt_to_value("2", hp=hp) == 'c'
            assert converter.txt_to_value("3", hp=hp) == 'd'


def test_txt_to_value_metrics():
    """Test txt_to_value for metrics - exact values where quantization allows"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        if isinstance(converter, OptformerConverter):
            assert converter.txt_to_value("<500>", hp=None, x_min=0.0, x_max=1.0) == 0.5
            assert converter.txt_to_value("<0>", hp=None, x_min=0.0, x_max=1.0) == 0.0
            assert converter.txt_to_value("<1000>", hp=None, x_min=0.0, x_max=1.0) == 1.0
        else:
            assert converter.txt_to_value("500", hp=None, x_min=0.0, x_max=1.0) == 0.5
            assert converter.txt_to_value("0", hp=None, x_min=0.0, x_max=1.0) == 0.0
            assert converter.txt_to_value("1000", hp=None, x_min=0.0, x_max=1.0) == 1.0


def test_txt_to_value_metrics_missing_bounds():
    """Test txt_to_value for metrics raises error when x_min/x_max not provided"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        with pytest.raises(ValueError, match="x_min and x_max must be provided"):
            if isinstance(converter, OptformerConverter):
                converter.txt_to_value("<500>", hp=None)
            else:
                converter.txt_to_value("500", hp=None)


def test_round_trip_conversion():
    """Test round-trip conversion: value -> txt -> value - exact where quantization allows and with inexact decimal cases"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        hp = uniform(0, 1)
        val = 0.123456
        txt = converter.value_to_txt(val, hp=hp)
        recovered = converter.txt_to_value(txt, hp=hp)
        quantization_interval = 1.0 / converter.q
        assert abs(val - recovered) <= quantization_interval / 2, \
            f"Inexact quantization round-trip failed: {val} -> {txt} -> {recovered} (allowed tolerance {quantization_interval/2})"
        
        val = 0.999999
        txt = converter.value_to_txt(val, hp=hp)
        recovered = converter.txt_to_value(txt, hp=hp)
        quantization_interval = 1.0 / converter.q
        assert abs(val - recovered) <= quantization_interval, \
            f"Inexact quantization round-trip failed: {val} -> {txt} -> {recovered} (allowed tolerance {quantization_interval/2})"

        val = 1.0
        txt = converter.value_to_txt(val, hp=hp)
        recovered = converter.txt_to_value(txt, hp=hp)
        assert val == recovered, f"Round-trip failed for {val}: {txt} -> {recovered}"


def test_optformer_parse_token_invalid():
    """Test OptformerConverter._parse_token with invalid format"""
    converter = OptformerConverter(q=1000)
    with pytest.raises(AssertionError, match="Token string must start with"):
        converter._parse_token("500")  # Missing angle brackets
    
    with pytest.raises(AssertionError, match="Token string must start with"):
        converter._parse_token("500>")  # Missing opening bracket
    
    with pytest.raises(AssertionError, match="Token string must start with"):
        converter._parse_token("<500")  # Missing closing bracket


# ============================================================================
# History.get_prompt() Tests
# ============================================================================

def test_history_basic():
    """Test History.get_prompt() with basic configuration"""
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
        'z': choice(['a', 'b', 'c'])
    }
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        history = History(name='test', algorithm='test', config_space=config_space, converter=converter)
        history.add_trial({'x': 0.5, 'y': 5, 'z': 'a'}, 0.5)
        history.add_trial({'x': 0.6, 'y': 6, 'z': 'b'}, 0.6)
        prompt = history.get_prompt()
        assert isinstance(prompt, str)
        
        if isinstance(converter, OptformerConverter):
            assert (
                "benchmark:test,algorithm:test" \
                "&&name:x,type:UNI,min_value:0,max_value:1" \
                "*name:y,type:INT,min_value:0,max_value:10" \
                "*name:z,type:CAT,categories:['a','b','c']" \
                "&<500><500><0>*<0>|<600><600><1>*<1000>|"
                == prompt
            )
        elif isinstance(converter, Converter):
            assert (
                "benchmark:test,algorithm:test," \
                "parameter:{name:x,type:UNI,min_value:0,max_value:1,}" \
                "parameter:{name:y,type:INT,min_value:0,max_value:10,}" \
                "parameter:{name:z,type:CAT,categories:['a','b','c'],}" \
                "&500,500,0*0|600,600,1*1000|"
                == prompt
            )


def test_history_trial_count_from_prompt():
    """Test that the number of trials in the prompt matches the number added to History"""
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
    }
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        history = History(name='test', algorithm='test', config_space=config_space, converter=converter)

        for nb_trials in [0, 2, 10]:
            for i in range(nb_trials):
                history.add_trial({'x': config_space['x'].sample(), 'y': config_space['y'].sample()}, i)
        prompt = history.get_prompt()
        prompt_trials = prompt.split("&")[-1]
        prompt_trials = prompt_trials[:-1].split("|")
        assert len(prompt_trials) == len(history.trials)

def test_history_parameter_count_from_prompt():
    """Test that the number of parameters in the prompt matches the number of parameters in the config space"""
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        for nb_parameters in [1, 2, 10]:
            config_space = {f'x_{i}': uniform(0, 1) for i in range(nb_parameters)}
            history = History(name='test', algorithm='test', config_space=config_space, converter=converter)
            history.add_trial({f'x_{i}': 0.5 for i in range(nb_parameters)}, 0.5)
            prompt = history.get_prompt()
            if isinstance(converter, OptformerConverter):
                prompt_parameters = prompt.split("&")[-2].split("*")
            elif isinstance(converter, Converter):
                prompt_parameters = prompt.split("&")[-2].split("parameter:")[1:]
            assert len(prompt_parameters) == nb_parameters

def test_history_shuffle():
    """Test History.get_prompt() with shuffle parameter"""
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
        'z': choice(['a', 'b', 'c'])
    }
    for converter in [Converter(q=1000), OptformerConverter(q=1000)]:
        history = History(name='test', algorithm='test', config_space=config_space, converter=converter)
        history.add_trial({'x': 0.5, 'y': 5, 'z': 'a'}, 0.5)
        
        # Get prompts with and without shuffle
        prompt_no_shuffle = history.get_prompt(shuffle=False)
        prompt_shuffle1 = history.get_prompt(shuffle=True)
        prompt_shuffle2 = history.get_prompt(shuffle=True)
        
        # Without shuffle should be deterministic
        assert prompt_no_shuffle == history.get_prompt(shuffle=False)
        
        # With shuffle, problem definition may differ but should be valid
        assert isinstance(prompt_shuffle1, str)
        assert isinstance(prompt_shuffle2, str)
        assert 'benchmark:test' in prompt_shuffle1
        assert 'algorithm:test' in prompt_shuffle1

def test_trial():
    trial = Trial(config={'x': 0.5}, metric=0.5)
    assert trial.config == {'x': 0.5}
    assert trial.metric == 0.5

def test_preprocess():
    prompt = 'parameter "trial" '
    processed_prompt = preprocess(prompt)
    assert processed_prompt == ''


def test_from_syne_tune_experiment():
    from syne_tune import Tuner, StoppingCriterion
    from syne_tune.backend import PythonBackend
    from syne_tune.config_space import randint
    from syne_tune.optimizer.baselines import RandomSearch

    def train_height(steps: int, width: float, height: float):
        """
        The function to be tuned, note that import must be in PythonBackend and no global variable are allowed,
        more details on requirements of tuned functions can be found in
        :class:`~syne_tune.backend.PythonBackend`.
        """
        from syne_tune import Reporter

        reporter = Reporter()
        for step in range(steps):
            dummy_score = (0.1 + width * step / 100) ** (-1) + height * 0.1
            # Feed the score back to Syne Tune.
            reporter(step=step, mean_loss=dummy_score)
            #time.sleep(0.1)

    config_space = {
        "steps": 100,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }

    metric = "mean_loss"
    scheduler = RandomSearch(
        config_space,
        metrics=[metric],
    )


    stop_criterion = StoppingCriterion(
        max_num_trials_completed=1,
    )

    with tempfile.TemporaryDirectory() as local_path:
        os.environ[SYNE_TUNE_ENV_FOLDER] = local_path
        backend = PythonBackend(tune_function=train_height, config_space=config_space)
        backend.set_path(results_root=local_path)
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=1,
            save_tuner=False,
            
        )
        tuner.run()
        experiment = load_experiment(tuner_name=tuner.name, local_path=local_path)
        history = History.from_syne_tune_experiment(experiment)

        assert len(history.trials) >= 1


def test_optformer_get_user_defined_symbols():
    """Test OptformerConverter.get_user_defined_symbols() returns correct symbols"""
    converter = OptformerConverter(q=100)
    symbols = converter.get_user_defined_symbols()
    
    # Check base symbols
    assert 'name' in symbols
    assert 'algorithm' in symbols
    assert 'benchmark' in symbols
    assert 'type' in symbols
    assert 'CAT' in symbols
    assert 'UNI' in symbols
    assert 'INT' in symbols
    assert 'LOGUNI' in symbols
    assert 'LOGINT' in symbols
    assert '|' in symbols
    assert '&' in symbols
    assert '*' in symbols
    assert ',' in symbols
    assert '<' in symbols
    assert '>' in symbols
    
    # Check quantized tokens
    assert '<0>' in symbols
    assert '<50>' in symbols
    assert '<100>' in symbols
    quant_tokens = [s for s in symbols if s.startswith('<') and s.endswith('>')]
    assert len(quant_tokens) == 101  # 0 to 100


def test_optformer_from_syne_tune_experiment():
    """End-to-end integration test: OptformerConverter with real Syne Tune experiment"""
    from syne_tune import Tuner, StoppingCriterion
    from syne_tune.backend import PythonBackend
    from syne_tune.config_space import randint, uniform, choice, loguniform, lograndint, Categorical
    from syne_tune.optimizer.baselines import RandomSearch

    def train_function(width: float, height: int, category: str, log_param: float, log_int: int):
        """
        The function to be tuned with various hyperparameter types.
        """
        from syne_tune import Reporter

        reporter = Reporter()
        for step in range(100):
            # Create a dummy score that depends on all parameters
            score = (0.1 + width * step  / 100) ** (-1) + height * 0.1 + len(category) * 0.05 + log_param * 0.1 + log_int * 0.01
            reporter(step=step, mean_loss=score)

    # Config space with various hyperparameter types
    config_space = {
        "width": uniform(0.0, 1.0),
        "height": randint(0, 20),
        "category": choice(['a', 'b', 'c', 'd']),
        "log_param": loguniform(1e-3, 1.0),
        "log_int": lograndint(1, 100),
    }

    metric = "mean_loss"
    scheduler = RandomSearch(
        config_space,
        metrics=[metric],
    )

    stop_criterion = StoppingCriterion(
        max_num_trials_completed=5,
    )

    with tempfile.TemporaryDirectory() as local_path:
        os.environ[SYNE_TUNE_ENV_FOLDER] = local_path
        backend = PythonBackend(tune_function=train_function, config_space=config_space)
        backend.set_path(results_root=local_path)
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=1,
            save_tuner=False,
        )
        tuner.run()
        experiment = load_experiment(tuner_name=tuner.name, local_path=local_path)
        
        history = History.from_syne_tune_experiment(
            experiment, 
            converter=OptformerConverter(q=1000)
        )

        # Verify basic structure
        assert len(history.trials) >= 5
        assert history.converter is not None
        assert isinstance(history.converter, OptformerConverter)
        
        # Verify prompt format
        prompt = history.get_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Verify prompt contains expected components
        assert f"benchmark:{history.name}," in prompt
        assert f"algorithm:{history.algorithm}" in prompt
        assert "&&" in prompt
        
        # Verify hyperparameter types are in the prompt
        assert "type:UNI" in prompt
        assert "type:INT" in prompt
        assert "type:CAT" in prompt
        assert "type:LOGUNI" in prompt
        assert "type:LOGINT" in prompt
        
        # Verify trials section format
        assert "&" in prompt
        trials_section = prompt.split("&")[-1]
        assert "|" in trials_section
        
        # Verify all trials are formatted with <VALUE> tokens
        trial_parts = trials_section.split("|")
        # Filter out empty strings
        trial_parts = [t for t in trial_parts if t]
        assert len(trial_parts) >= 5
        
        # Verify each trial has <VALUE> format for tokens
        for trial_part in trial_parts:
            if trial_part:
                assert "<" in trial_part and ">" in trial_part
                assert "*" in trial_part
        
        sample_trial = history.trials[0]
        for hp_name, hp in history.config_space.items():
            value = sample_trial.config[hp_name]
            txt = history.converter.value_to_txt(value, hp=hp, hp_name=hp_name)
            recovered = history.converter.txt_to_value(txt, hp=hp, hp_name=hp_name)
            
            if isinstance(hp, Categorical):
                assert recovered == value
            else:
                quantization_interval = (hp.upper - hp.lower) / history.converter.q
                # Use full quantization interval as tolerance (floor quantization can have error up to full interval)
                # Add small epsilon for floating point precision
                tolerance = quantization_interval + 1e-6
                assert abs(value - recovered) <= tolerance, \
                    f"Round-trip failed for {hp_name}: {value} -> {txt} -> {recovered} (tolerance: {tolerance})"


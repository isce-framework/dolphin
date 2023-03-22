from dolphin.workflows._pge_runconfig import AlgorithmParameters, RunConfig


def test_algorithm_parameters_yaml():
    AlgorithmParameters.print_yaml_schema()


def test_run_config_yaml():
    RunConfig.print_yaml_schema()

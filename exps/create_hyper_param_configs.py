"""Creates several configs in a folder for specified hyperparameters.


"""

import fire
import os
import yaml
import copy


def run(
    hyperparams_config,
):
    """Creates hyperparameter configs.

    Parameters
    ----------
    hyperparams_config : str
        A YAML file specifies how the hyperparams should be changed
        For example, to search over different learning rates:

        learning_rate(hyper):
        [
          {initial_learning_rate: 1e-3, decay_rate: 0.96},
          {initial_learning_rate: 1e-4, decay_rate: 0.98},
        ]

        Whatever parameters is marked with "(hyper)" needs to be a
        list, and the values in the list would be expanded.

        A folder with the same path without the '.yaml' extension will
        be created to hold the output configs. For example, if input
        config path is '/path/to/hyper.yaml', then an output folder
        '/path/to/hyper' will be created.

    """
    output_folder = os.path.splitext(hyperparams_config)[0]
    assert not os.path.exists(output_folder)
    os.makedirs(output_folder)

    with open(hyperparams_config) as f:
        base_config = yaml.safe_load(f)

    # Go through the config and write to new config files until there
    # is no param marked with (hyper)
    # Each element is (config, hyperparams_set); starting with none of
    # the hyperparameters set.
    configs_to_scan = [(base_config, [])]
    while True:
        new_configs = []
        for config, cur_params in configs_to_scan:
            expanded_config, new_params = replace_first_hyperparam(
                config
            )

            # Add in the params that are already set
            for p in new_params:
                p.update(cur_params)

            new_configs.extend(zip(expanded_config, new_params))

        if len(new_configs) == 0:
            break

        configs_to_scan = new_configs

    # Write the configs into the folder
    for config_id, (config, _) in enumerate(configs_to_scan):
        path = os.path.join(output_folder, 'config{:03d}.yaml'.format(
            config_id
        ))
        with open(path, 'w') as f:
            yaml.dump(config, f)
        print("Written to {}.".format(path))

    # Write a summary for the hyperparams set
    path = os.path.join(output_folder, 'summary.txt')
    with open(path, 'w') as f:
        for config_id, (_, params) in enumerate(configs_to_scan):
            f.write('config{:03d}:\n'.format(config_id))
            params_str = yaml.dump(params)
            f.write(params_str)
            f.write('\n')
    print("Written to {}.".format(path))


def replace_first_hyperparam(config):
    """Replace the first hyperparameter encounter in the config.

    For example, the input config can be

    learning_rate(hyper):
        [{initial_learning_rate: 1e-3, decay_rate: 0.96},
         {initial_learning_rate: 1e-4, decay_rate: 0.98}]

    The output would be
    [
     {learning_rate: {initial_learning_rate: 1e-3, decay_rate: 0.96}},
     {learning_rate: {initial_learning_rate: 1e-4, decay_rate: 0.98}},
    ]

    Parameters
    ----------
    config : dict

    Returns
    -------
    expanded : [dict]
        A list of configs with the first hyperparameter changed to one
        of the desired values.
    hyperparams : [dict]
        Each dict records the hyperparameters that have been set for
        the corresponding config, not nested. Same length as
        `expanded`. This is used so that it's easier to see what are
        the hyperparameters set for each config.

    """
    hyper_str = '(hyper)'
    hyper_key = None
    hyperparams = []

    if type(config) == dict:
        for k, v in config.items():
            if k.endswith(hyper_str):
                assert type(v) == list
                hyper_key = k[:-len(hyper_str)]
                hyper_values = v
                config.pop(k)
                break

            elif type(v) in [dict, list]:
                replaced, hyperparams = replace_first_hyperparam(v)
                if len(replaced) > 0:
                    hyper_key = k
                    hyper_values = replaced
                    break

    elif type(config) == list:
        for idx, item in enumerate(config):
            replaced, hyperparams = replace_first_hyperparam(item)
            if len(replaced) > 0:
                hyper_key = idx
                hyper_values = replaced
                break

    if hyper_key is None:
        return [], []

    expanded = []
    params = []
    for v in hyper_values:
        config_copy = copy.deepcopy(config)
        config_copy[hyper_key] = v
        expanded.append(config_copy)
        params.append({hyper_key: v})

    # Only set to `params` if `hyperparams` is empty, meaning the
    # hyperparameters are set in the current level, not nested.
    if len(hyperparams) == 0:
        hyperparams = params

    return expanded, hyperparams


if __name__ == '__main__':
    fire.Fire(run)

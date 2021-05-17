import os
import fire
import subprocess


def run(hyperparams_folder, output_folder, exp_script_path):
    """Runs hyperparameter search.

    Parameters
    ----------
    hyperparams_folder : str
        A folder contains several configs of parameters to run
    output_folder : str
        Where to save the output model
    exp_script_path : str
        The path to which the experiment script lies. An experiment
        script is expected to take a config path and a model path

    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(hyperparams_folder)
    files.sort()  # run them in order

    # filter out summary file
    files = [f for f in files if f != 'summary.txt']

    for config_id, config_file in enumerate(files):
        config_path = os.path.join(hyperparams_folder, config_file)
        model_path = os.path.join(output_folder, 'model{:03d}'.format(
            config_id
        ))

        command = 'python {} --config-path={} --model-path={}'.format(
            exp_script_path,
            config_path,
            model_path,
        )

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        stdout_path = os.path.join(model_path, 'stdout')
        stderr_path = os.path.join(model_path, 'stderr')
        stdout = open(stdout_path, 'w')
        stderr = open(stderr_path, 'w')
        print('Running {}'.format(command))
        print('Output redirected to {} and {}'.format(
            stdout_path, stderr_path
        ))
        subprocess.call(command.split(), stdout=stdout, stderr=stderr)

        stdout.close()
        stderr.close()


if __name__ == '__main__':
    fire.Fire(run)

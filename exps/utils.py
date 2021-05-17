import os
import yaml
import shutil

from artistcritic.path import MODEL_ROOT


def load_and_save_config(config_path, model_path):
    """Loads the config and save a copy to the model folder."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = os.path.expanduser(model_path)

    # If `model_path` is absolute, os.path.join would return
    # `model_path` (!!)
    model_path = os.path.join(MODEL_ROOT, model_path)

    # Save the config
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    shutil.copyfile(
        src=config_path,
        dst=os.path.join(model_path, 'exp_config.yaml')
    )

    return config

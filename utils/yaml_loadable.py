"""Interface for classes that can be loaded from yaml file."""

import yaml


class YAMLLoadable:
    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()

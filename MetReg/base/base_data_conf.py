import six
import abc


@six.add_metaclass(abc.ABCMeta)
class base_data_conf():
    """base class for configuration of datasets.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> dict:
        return self._config

    def get_config(self):
        return

    def set_config(self):
        return

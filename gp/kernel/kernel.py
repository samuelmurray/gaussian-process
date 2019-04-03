from abc import ABC


class Kernel(ABC):
    """
    Base class for all kernels
    """

    def __init__(self, nparams):
        self._nparams = nparams

    def __call__(self, x1, x2):
        assert x1.shape[1] == x2.shape[1], "Vectors must be of matching dimension"

    def set_params(self, params):
        assert params.size == self.nparams, \
            f"Provided {params.size} params; must be {self.nparams}"

    def get_params(self):
        raise NotImplementedError

    def get_true_params(self):
        raise NotImplementedError

    def gradients(self, x):
        raise NotImplementedError

    def gradients_wrt_data(self, x, n=None, dim=None):
        raise NotImplementedError

    @property
    def nparams(self):
        return self._nparams

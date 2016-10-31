from chainer import cuda


class NamedWeightDecay(object):

    """Optimizer hook function for specific weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.
        name (str) : Name of layer to be regularized

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'WeightDecay'

    def __init__(self, rate, name):
        self.rate = rate
        self.name = name

    def __call__(self, opt):
        if cuda.available:
            kernel = cuda.elementwise(
                'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

        rate = self.rate
        for name, param in opt.target.namedparams():
            if self.name in name:
                p, g = param.data, param.grad
                with cuda.get_device(p) as dev:
                    if int(dev) == -1:
                        g += rate * p
                    else:
                        kernel(p, rate, g)

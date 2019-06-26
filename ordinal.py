import numpy as np
import tensorflow as tf

from gpflow import logdensities
from gpflow import priors
from gpflow import settings
from gpflow import transforms
from gpflow.decors import params_as_tensors
from gpflow.decors import params_as_tensors_for
from gpflow.params import ParamList
from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.quadrature import hermgauss
from gpflow.quadrature import ndiagquad, ndiag_mc

from gpflow.likelihoods import Likelihood

def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class Ordinal(Likelihood):
    """
    A likelihood for doing ordinal regression.
    The data are integer values from 0 to K, and the user must specify (K-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a_0, a_1, ... a_{K-1}], then the likelihood is
    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)
    where phi is the cumulative density function of a Gaussian (the inverse probit
    function) and sigma is a parameter to be learned. A reference is:
    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }
    """

    
    def __init__(self, bin_start, bin_width, num_edges,**kwargs):
        """
        bin_edges is a numpy array specifying at which function value the
        output label should switch. If the possible Y values are 0...K, then
        the size of bin_edges should be (K-1).
        """
        super().__init__(**kwargs)
        self.bin_start = Parameter(bin_start, dtype=settings.float_type)
        self.bin_width = Parameter(bin_width, transform=transforms.positive, dtype=settings.float_type)
        self.sigma = Parameter(1.0, transform=transforms.positive)
        self.num_edges = num_edges
        self.num_bins = self.num_edges + 1

    @params_as_tensors
    def logp(self, F, Y):     
        
        self.bin_edges = [self.bin_start+ x*self.bin_width for x in range(self.num_edges)]       
        Y = tf.cast(Y, tf.int64)
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        return tf.log(inv_probit(selected_bins_left - F / self.sigma) -
                      inv_probit(selected_bins_right - F / self.sigma) + 1e-6)

    @params_as_tensors
    def _make_phi(self, F):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.
        Note that a matrix of F values is flattened.
        """
        self.bin_edges = [self.bin_start+ x*self.bin_width for x in range(self.num_edges)]
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        return inv_probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma) \
               - inv_probit(scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma)

    def conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        return tf.reshape(tf.matmul(phi, Ys), tf.shape(F))

    def conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        E_y = tf.matmul(phi, Ys)
        E_y2 = tf.matmul(phi, tf.square(Ys))
        return tf.reshape(E_y2 - tf.square(E_y), tf.shape(F))

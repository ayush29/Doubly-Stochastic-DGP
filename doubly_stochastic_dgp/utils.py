# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from gpflow import settings
import numpy as np

def normal_sample(mean, var, full_cov=False):
    if full_cov is False:
        z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)
        z = tf.random_normal([S, D, N, 1], dtype=settings.float_type)
        f = mean + tf.matmul(chol, z)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


class PositiveTransform(object):
    eps = 1e-6
    def forward(self, x):
        NotImplementedError

    def backward(self, y):
        NotImplementedError

    def forward_np(self, x):
        NotImplementedError

    def backward_np(self, y):
        NotImplementedError


class PositiveSoftplus(PositiveTransform):
    def forward(self, x):
        result = tf.log(1. + tf.exp(x)) + self.eps
        return tf.where(x > 35., x + self.eps, result)

    def backward(self, y):
        result = tf.log(tf.exp(y - self.eps) - 1.)
        return tf.where(y > 35., y - self.eps, result)

    def forward_np(self, x):
        result = np.log(1. + np.exp(x)) + self.eps
        return np.where(x > 35., x + self.eps, result)

    def backward_np(self, y):
        result = np.log(np.exp(y - self.eps) - 1.)
        return np.where(y > 35., y - self.eps, result)


class PositiveExp(PositiveTransform):
    def forward(self, x):
        return tf.exp(x) + self.eps

    def backward(self, y):
        return tf.log(y - self.eps)

    def forward_np(self, x):
        return np.exp(x) + self.eps

    def backward_np(self, y):
        return np.log(y - self.eps)

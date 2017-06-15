import numpy as np
import tensorflow as tf

import elbow.util as util

from elbow.conditional_dist import ConditionalDistribution
from elbow.parameterization import unconstrained, positive_exp, simplex_constrained, unit_interval, psd_matrix_small, psd_diagonal
from elbow.transforms import Logit, Simplex1, Simplex, Exp, TransformedDistribution, RowNormalize

import scipy.stats

class ContinuousUniform(ConditionalDistribution):
    """
    Represents a uniform distribution on an axis-aligned region in R^n. 

    Note this class does not perform bound checking (i.e., return
    logp=-inf outside the region) because this creates weird behavior
    with Gaussian approximating distributions, which always place
    *some* mass outside of the bounds. This could be fixed by
    implementing a truncated Gaussian posterior approximation.

    """
    
    def __init__(self, min_range, max_range, **kwargs):
        super(ContinuousUniform, self).__init__(min_range=min_range, max_range=max_range, **kwargs)

    def inputs(self):
        return {"min_range": None, "max_range": None}

    def _compute_shape(self, min_range_shape, max_range_shape, **kwargs):
        assert min_range_shape == max_range_shape
        return min_range_shape

    def _sample(self, min_range, max_range):
        return tf.random_uniform(shape=self.shape) * (max_range - min_range) + min_range

    def _logp(self, result, min_range, max_range):
        log_area = tf.reduce_sum(tf.log(max_range - min_range))
        return -log_area 

    def default_q(self):
        # TODO should implement a truncated Gaussian Q dist
        return Gaussian(shape=self.shape, name="q_"+self.name)

class GammaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, **kwargs):
        super(GammaMatrix, self).__init__(alpha=alpha, beta=beta, **kwargs)
    
    def inputs(self):
        return {"alpha": positive_exp, "beta": positive_exp}
    
    def _sample(self, alpha, beta):
        gammas = tf.random_gamma(shape=self.shape, alpha=alpha, beta=beta)
        return gammas

    def _logp(self, result, alpha, beta):    
        lp = tf.reduce_sum(util.dists.gamma_log_density(result, alpha, beta))
        return lp

    def _compute_shape(self, alpha_shape, beta_shape):
        return util.broadcast_shape(alpha_shape, beta_shape)
        
    def default_q(self, **kwargs):
        q1 = Gaussian(shape=self.shape)
        return TransformedDistribution(q1, Exp, name="q_"+self.name)

    def reparameterized(self):
        return False
    
class BetaMatrix(ConditionalDistribution):
    
    def __init__(self, alpha, beta, **kwargs):
        super(BetaMatrix, self).__init__(alpha=alpha, beta=beta, **kwargs)
    
    def inputs(self):
        return {"alpha": positive_exp, "beta": positive_exp}
    
    def _sample(self, alpha, beta):
        X = tf.random_gamma(shape=self.shape, alpha=alpha, beta=1)
        Y = tf.random_gamma(shape=self.shape, alpha=beta, beta=1)
        Z = X/(X+Y)
        return Z

    def _logp(self, result, alpha, beta):    
        lp = tf.reduce_sum(util.dists.beta_log_density(result, alpha, beta))
        return lp

    def _compute_shape(self, alpha_shape, beta_shape):
        return util.broadcast_shape(alpha_shape, beta_shape)
        
    def default_q(self, **kwargs):
        q1 = Gaussian(shape=self.shape)
        return TransformedDistribution(q1, Logit, name="q_"+self.name)

    def reparameterized(self):
        return False

        
class DirichletMatrix(ConditionalDistribution):
    """
    Currently just describes a vector of shape (K,), though could be extended to 
    a (N, K) matrix of iid draws. 
    """
    
    def __init__(self, alpha, **kwargs):
        super(DirichletMatrix, self).__init__(alpha=alpha, **kwargs)
        self.N, self.K = self.shape
        
    def inputs(self):
        return {"alpha": positive_exp}
    
    def _sample(self, alpha):

        # broadcast alpha from scalar to vector, if necessary
        alpha = alpha * tf.ones(shape=self.shape, dtype=self.dtype)

        gammas = tf.squeeze(tf.random_gamma(shape=(1,), alpha=alpha, beta=1))
        sample = RowNormalize.transform(gammas)
        return sample

    def _logp(self, result, alpha):    
        lp = tf.reduce_sum(util.dists.dirichlet_log_density(result, alpha))
        return lp

    def _compute_shape(self, alpha_shape):        
        return alpha_shape
        
    def default_q(self, **kwargs):
        n, k = self.shape

        # TODO: should we prefer Simplex or Simplex1 transformation?
        # preliminarily: Simplex1 seems to yield degenerate
        # posteriors, can't represent some natural distributions on
        # the simplex. not entirely sure why.
        
        #q1 = Gaussian(shape=(n, k-1))
        #return TransformedDistribution(q1, Simplex1, name="q_"+self.name)

        q1 = Gaussian(shape=(n, k))
        return TransformedDistribution(q1, Simplex, name="q_"+self.name)
        
    def reparameterized(self):
        return False
    
class BernoulliMatrix(ConditionalDistribution):
    def __init__(self, p=None, **kwargs):
        super(BernoulliMatrix, self).__init__(p=p, **kwargs)        
        
    def inputs(self):
        return {"p": unit_interval}

    def _input_shape(self, param, **kwargs):
        assert (param in self.inputs().keys())
        return self.shape
                     
    def _sample(self, p):
        unif = tf.random_uniform(shape=self.shape, dtype=tf.float32)
        return tf.cast(unif < p, self.dtype)
    
    def _expected_logp(self, q_result, q_p):
        # compute E_q [log p(z)] for a given prior p(z) and an approximate posterior q(z).
        # note q_p represents our posterior uncertainty over the parameters p: if these are known and
        # fixed, q_p is just a delta function, otherwise we have to do Monte Carlo sampling.
        # Whereas q_z represents our posterior over the sampled (Bernoulli) values themselves,
        # and we assume this is in the form of a set of Bernoulli probabilities. 

        p_z = q_p._sampled

        try:
            q_z = q_result.p
            lp = -tf.reduce_sum(util.dists.bernoulli_entropy(q_z, cross_q = p_z))
        except:
            lp = self._logp(result=q_result._sampled, p=p_z)
        return lp

    def _logp(self, result, p):
        lps = util.dists.bernoulli_log_density(result, p)
        return tf.reduce_sum(lps)
    
    def _entropy(self, p):
        return tf.reduce_sum(util.dists.bernoulli_entropy(p))
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def default_q(self, **kwargs):
        return BernoulliMatrix(shape=self.shape, name="q_"+self.name)
    
    def reparameterized(self):
        return False
    
class MultinomialMatrix(ConditionalDistribution):
    # matrix in which each row contains a single 1, and 0s otherwise.
    # the location of the 1 is sampled from a multinomial(p) distribution,
    # where the parameter p is a (normalized) vector of probabilities.
    # matrix shape: N x K
    # parameter shape: (K,)
    
    def __init__(self, p, **kwargs):
        super(MultinomialMatrix, self).__init__(p=p, **kwargs) 
        
    def inputs(self):
        return ("p")

    def _sample(self, p):
        N, K = self.shape        
        choices = tf.multinomial(tf.log(p), num_samples=N)

        """
        M = np.zeros(N, K, dtype=self.dtype)
        r = np.arange(N)
        M[r, choices] = 1
        """

        M = tf.one_hot(choices, depth=K, axis=-1)
        return M
    
    def _expected_logp(self, q_result, q_p):
        p = q_p._sampled
        q = q_result.p

        lp = tf.reduce_sum(util.dists.multinomial_entropy(q, cross_q=p))
        return lp
    
    def _compute_shape(self, p_shape):
        return p_shape
        
    def reparameterized(self):
        return False


class Laplace(ConditionalDistribution):

    def __init__(self, loc=None, scale=None, **kwargs):
        super(Laplace, self).__init__(loc=loc, scale=scale, **kwargs)
    
    def inputs(self):
        return {"loc": unconstrained, "scale": positive_exp}

    def _sample(self, loc, scale):
        base = tf.random_uniform(shape=self.shape, dtype=tf.float32) - 0.5
        std_laplace = tf.sign(base) * tf.log(1-2*tf.abs(base))
        return loc + scale * std_laplace

    def _logp(self, result, loc, scale):
        return -tf.reduce_sum(tf.abs(result-loc)/scale - tf.log(2*scale))

    def _entropy(self, loc, scale):
        return 1 + tf.reduce_sum(tf.log(2*scale))

    def default_q(self, **kwargs):
        return Laplace(shape=self.shape, name="q_"+self.name)


class MVGaussian(ConditionalDistribution):

    def __init__(self, mean=None, cov=None, **kwargs):
        super(MVGaussian, self).__init__(mean=mean, cov=cov, **kwargs) 

    def inputs(self):
        return {"mean": unconstrained, "cov": psd_diagonal}

    def _input_shape(self, param, **kwargs):
        if param=="mean":
            return self.shape
        elif param=="cov":
            n, k = self.shape
            return (n, n)
        else:
            raise Exception("unrecognized param %s" % param)

    
    def outputs(self):
        return ("out",)

    def _sample(self, mean, cov):
        L = tf.cholesky(cov)
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        return tf.matmul(L, eps) + mean
    
    def _logp(self, result, mean, cov):
        lp = multivariate_gaussian_log_density(result, mu=mean, Sigma=cov)
        return lp

    def _entropy(self, cov, **kwargs):
        return util.dists.multivariate_gaussian_entropy(Sigma=cov)
    
    def _sample_and_entropy(self, mean, cov, **kwargs):
        L = tf.cholesky(cov)
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)
        sample = tf.matmul(L, eps) + mean
        entropy = util.dists.multivariate_gaussian_entropy(L=L)
        return sample, entropy
    
    def reparameterized(self):
        return True


def is_gaussian(dist):
    """
    Convenience method to identify Gaussian distributions via duck typing
    """
    try:
        dist.mean
        dist.variance
        return True
    except:
        return False

    
class Gaussian(ConditionalDistribution):
    
    def __init__(self, mean=None, std=None, **kwargs):
        super(Gaussian, self).__init__(mean=mean, std=std, **kwargs) 

    def inputs(self):
        return {"mean": unconstrained, "std": positive_exp}

    def outputs(self):
        return ("out",)

    def derived_parameters(self, mean, std, **kwargs):
        return {"variance": std**2}
    
    def _sample(self, mean, std):
        eps = tf.random_normal(shape=self.shape, dtype=self.dtype)        
        return eps * std + mean
    
    def _logp(self, result, mean, std):
        lp = tf.reduce_sum(util.dists.gaussian_log_density(result, mean=mean, stddev=std))
        return lp

    def _entropy(self, std, **kwargs):
        variance = tf.ones(self.shape) * std**2
        return tf.reduce_sum(util.dists.gaussian_entropy(variance=variance))

    def default_q(self, **kwargs):
        return Gaussian(shape=self.shape, name="q_"+self.name)

    def _expected_logp(self, q_result, q_mean=None, q_std=None):

        def get_sample(q, param):
            return self.inputs_nonrandom[param] if q is None else q._sampled
            
        std_sample = get_sample(q_std, 'std')
        mean_sample = get_sample(q_mean, 'mean')
        result_sample = q_result._sampled
        
        if is_gaussian(q_result) and not is_gaussian(q_mean):
            cross = util.dists.gaussian_cross_entropy(q_result.mean, q_result.variance, mean_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif not is_gaussian(q_result) and is_gaussian(q_mean):
            cross = util.dists.gaussian_cross_entropy(q_mean.mean, q_mean.variance, result_sample, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        elif is_gaussian(q_result) and is_gaussian(q_mean):
            cross = util.dists.gaussian_cross_entropy(q_mean.mean, q_mean.variance + q_result.variance, q_result.mean, tf.square(std_sample))
            elp = -tf.reduce_sum(cross)
        else:
            elp = self._logp(result=result_sample, mean=mean_sample, std=std_sample)
        return elp
            
    def reparameterized(self):
        return True

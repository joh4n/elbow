import numpy as np
import tensorflow as tf

from elbow.util.special_hacks import gammaln, betaln
from elbow.util.misc import extract_shape
import scipy.special


    
def gaussian_entropy(stddev=None, variance=None):
    """Creates a TensorFlow variable representing the sum of one or more
    Gaussian entropies.

    Args:v
        stddev (optional, vector or scalar): mutually exclusive with variance
        variance (optional, vector or scalar): mutually exclusive with stddev

    Note that the entropy of a Gaussian does not depend on the mean. 

    Returns:
        h: TF scalar containing the total entropy of this distribution
    """
    if variance is None:
        variance = stddev*stddev

    t = .5 * (1 + np.log(2*np.pi) + tf.log(variance))
    entropy = tf.reduce_sum(t)
    return entropy

def gaussian_cross_entropy(mean_p, variance_p, mean_q, variance_q):
    # E_p [ - log q(x) ]
    gaussian_lp = gaussian_log_density(mean_p, mean = mean_q, variance=variance_q)
    correction = -.5 * variance_p / variance_q
    return  - (gaussian_lp + correction)

def gaussian_kl(mu_p, sigma2_p, mu_q=None, sigma2_q=None):

    if mu_q is None and sigma2_q is None:
        # assume standard normal
        kl = .5 * (tf.square(mu_p) + sigma2_p - 1.0 - tf.log(sigma2_p))
    else:
        logsp = tf.log(sigma2_p, name="logsp")
        logsq =  tf.log(sigma2_q, name="logsq")
        kl = .5 * (tf.square(mu_p - mu_q)/sigma2_q + sigma2_p/sigma2_q - 1.0 - logsp + logsq)
        
    return kl


def gaussian_log_density(x, mean=None, stddev=None, variance=None):
    """Creates a TensorFlow variable representing the sum of one or more
    independent Gaussian log-densities.

    Args:
        x (scalar or vector): variable(s) being modeled as Gaussian
        mean (optional): defaults to zero if not specified
        stddev (optional): mutually exclusive with variance
        variance (optional): mutually exclusive with stddev

    Each of mean, stddev, variance must either be the same dimension
    as x, or a scalar (which will be broadcast as required). They may
    be TensorFlow variables or Python/numpy values.

    Returns:
        lp: TF scalar containing the log probability of x 
    """
    
    if variance is None:
        variance = stddev * stddev
    
    if mean is None:
        r = x
    else:
        r = x - mean

    z = r*r / variance
    
    lps = -0.5 * z   - .5 * tf.log(2*np.pi * variance)
    return lps

def multivariate_gaussian_log_density(x, mu,
                                      Sigma=None, L=None,
                                      prec=None, L_prec=None):
    """
    Assume X is a single vector described by a multivariate Gaussian
    distribution with x ~ N(mu, Sigma).

    We accept parameterization in terms of the covariance matrix or
    its cholesky decomposition L (more efficient if available), or the
    precision matrix or its cholesky decomposition L_prec.
    The latter is useful when representing a Gaussian in its natural 
    parameterization. Note that we still require the explicit mean mu
    (not the natural parameter prec*mu) since I'm too lazy to cover
    all the permutations of possible arguments (though this should be
    straightforward). 

    """
    s = extract_shape(x)
    try:
        n, = s
    except:
        n, m = s
        assert(m==1)

    if L is None and Sigma is not None:
        L = tf.cholesky(Sigma)        
    if L_prec is None and prec is not None:
        L_prec = tf.cholesky(prec)
        
    if L is not None:
        neg_half_logdet = -tf.reduce_sum(tf.log(tf.diag_part(L)))
    else:
        assert(L_prec is not None)
        neg_half_logdet = tf.reduce_sum(tf.log(tf.diag_part(L_prec)))
        
    d = tf.reshape(x - mu, (n,1))
    if L is not None:
        alpha = tf.matrix_triangular_solve(L, d, lower=True)
        exponential_part= tf.reduce_sum(tf.square(alpha))
    elif prec is not None:
        d = tf.reshape(d, (n, 1))
        exponential_part = tf.reduce_sum(d * tf.matmul(prec, d))
    else:
        assert(L_prec is not None)
        d = tf.reshape(d, (1, n))
        alpha = tf.matmul(d, L_prec)
        exponential_part= tf.reduce_sum(tf.square(alpha))

    n_log2pi = n * 1.83787706641
    logp =  -0.5 * n_log2pi
    logp += neg_half_logdet
    logp += -0.5 * exponential_part
        
    return logp


def multivariate_gaussian_entropy(Sigma=None, L=None, L_prec=None):

    if L is None and Sigma is not None:
        L = tf.cholesky(Sigma)
    
    if L is not None:
        half_logdet = tf.reduce_sum(tf.log(tf.diag_part(L)))
        n, _ = extract_shape(L)
    else:
        half_logdet = -tf.reduce_sum(tf.log(tf.diag_part(L_prec)))
        n, _ = extract_shape(L_prec)

    log_2pi = 1.83787706641
    entropy = .5*n*(1 + log_2pi) + half_logdet
    return entropy

def inv_gamma_log_density(x, alpha, beta):
    """Creates a TensorFlow variable representing the sum of one or more
    independent inverse Gamma log-densities.

    Args:
        x (scalar or vector): variable(s) being modeled as InvGamma
        alpha (scalar or vector)
        beta (scalar or vector)

    Each of alpha and beta must either be the same dimension
    as x, or a scalar (which will be broadcast as required). They may
    be TensorFlow variables or Python/numpy values.

    Returns:
        lp: TF scalar containing the log probability of x 
    """

    if isinstance(beta, tf.Tensor):
        log_beta = tf.log(beta)
    else:
        log_beta = np.log(beta)

    if isinstance(x, tf.Tensor):
        log_x = tf.log(x)
    else:
        log_x = np.log(x)


    if isinstance(alpha, tf.Tensor):
        gammaln_alpha = gammaln(alpha)
    else:
        gammaln_alpha = scipy.special.gammaln(alpha)

    lps = -beta / x + alpha * log_beta - (alpha+1) * tf.log(x) - gammaln_alpha
    return lps
    
def gamma_log_density(x, alpha, beta, parameterization=None):
    """Creates a TensorFlow variable representing the sum of one or more
    independent inverse Gamma densities.

    Args:
        x (scalar or vector): variable(s) being modeled as InvGamma
        alpha (scalar or vector):
        beta (scalar or vector):

    Each of alpha and beta must either be the same dimension
    as x, or a scalar (which will be broadcast as required). They may
    be TensorFlow variables or Python/numpy values.

    Returns:
        lp: TF scalar containing the log probability of x 
    """

    try:
        dtype = x.dtype
    except AttributeError:
        try:
            dtype = alpha.dtype
        except AttributeError:
            dtype = beta.dtype
        
    if isinstance(beta, tf.Tensor):
        log_beta = tf.log(beta)
    else:
        log_beta = np.log(beta)

    if isinstance(x, tf.Tensor):
        log_x = tf.log(x)
    else:
        log_x = np.log(x)
    
            
    l1 = -beta * x
    l2 = alpha * log_beta
    l3 = (alpha-1) * log_x

    if isinstance(alpha, tf.Tensor):
        l4 = gammaln(alpha)
    else:
        l4 = scipy.special.gammaln(alpha)

    lps = l1 + l2 + l3 - l4
    return lps

def dirichlet_log_density(x, alpha, clip_finite=True):
    """
    Assumes that x is either a vector or a right-stochastic matrix (i.e., each row is a probability dist), and alpha is broadcastable to the same shape as x. 

    WARNING: does not enforce sum(x)==1; bad things will happen if this is violated. 

    """

    if len(x.get_shape()) == 1:
        x = tf.reshape(x, (1, -1))
    
    if clip_finite:
        logx = tf.log(tf.clip_by_value(x, 1e-37, 1.0), name="dirichlet_logx")
    else:
        logx = tf.log(x, name="dirichlet_logx")
        
    # force broadcasting alpha 
    if alpha.get_shape() != x.get_shape():
        alpha = tf.zeros_like(x) + alpha
        
    log_z = tf.reduce_sum(gammaln(alpha), axis=1) - gammaln(tf.reduce_sum(alpha, axis=1))
    log_density = tf.reduce_sum((alpha - 1) * logx, axis=1) - log_z
    
    return log_density
    
def beta_log_density(x, alpha=1.0, beta=1.0):
    log_z = betaln(alpha, beta)
    log_density = (alpha - 1) * tf.log(x) + (beta-1) * tf.log(1-x) - log_z
    return log_density

def multinomial_entropy(p, cross_q=None, clip_finite=True):
    # compute the (cross) entropy of a probability vector vs another

    if cross_q is None:
        cross_q = p
    
    if clip_finite:
        lq = tf.log(tf.clip_by_value(cross_q, 1e-37, 1.0), name="multinomial_logq")
        lq1 = tf.log(tf.clip_by_value(1.0-cross_q, 1e-37, 1.0), name="multinomial_log1q")
    else:
        lq = tf.log(cross_q, name="multinomial_logq")
        lq1 = tf.log(1.0-cross_q, name="multinomial_log1q")

    entropy = -tf.reduce_sum(p * lq, axis=1)
    return entropy
    
def bernoulli_entropy(p, cross_q=None, clip_finite=True):
    # compute the entropy of a (tensor of independent) bernoulli distribution with probabilities p,
    # or optionally the cross-entropy with respect to another distribution with probabilities cross_q. 
    
    if cross_q is None:
        cross_q = p
    
    if clip_finite:
        lq = tf.log(tf.clip_by_value(cross_q, 1e-37, 1.0), name="bernoulli_logq")
        lq1 = tf.log(tf.clip_by_value(1.0-cross_q, 1e-37, 1.0), name="bernoulli_log1q")
    else:
        lq = tf.log(cross_q, name="bernoulli_logq")
        lq1 = tf.log(1.0-cross_q, name="bernoulli_log1q")

    entropy = -p * lq - (1.0-p) * lq1
    return entropy
    
def bernoulli_kl(p, q, clip_finite=True):

    if clip_finite:
        lp = tf.log(tf.clip_by_value(p, 1e-37, 1.0), name="bernoulli_logp")
        lp1 = tf.log(tf.clip_by_value(1.0-p, 1e-37, 1.0), name="bernoulli_log1p")
        lq = tf.log(tf.clip_by_value(q, 1e-37, 1.0), name="bernoulli_logq")
        lq1 = tf.log(tf.clip_by_value(1.0-q, 1e-37, 1.0), name="bernoulli_log1q")
    else:
        lp = tf.log(p, name="bernoulli_logp")
        lp1 = tf.log(1.0-p, name="bernoulli_log1p")
        lq = tf.log(p, name="bernoulli_logq")
        lq1 = tf.log(1.0-p, name="bernoulli_log1q")
        
    kl = p * (lp - lq) + (1.0-p) * (lp1 - lq1)
    return kl
    
def bernoulli_log_density(x, p, clip_finite=True):

    if clip_finite:
        # avoid taking log(0) for float32 inputs
        # TODO: adapt to float64, etc. 
        lp = tf.log(tf.clip_by_value(p, 1e-37, 1.0), name="bernoulli_logp")
        lp1 = tf.log(tf.clip_by_value(1.0-p, 1e-37, 1.0), name="bernoulli_log1p")
    else:
        lp = tf.log(p, name="bernoulli_logp")
        lp1 = tf.log(1.0-p, name="bernoulli_log1p")
        
    log_probs = tf.multiply(x, lp) + tf.multiply(1-x, lp1)
    return log_probs
    
    

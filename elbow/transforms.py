from __future__ import print_function
import numpy as np
import tensorflow as tf


# import elbow.util as util
from elbow.util import shapes_equal, shape_is_scalar, extract_shape
from elbow.conditional_dist import ConditionalDistribution


class DeterministicTransform(ConditionalDistribution):
    """
    Generic superclass for deterministic transforms.
    """
    
    def _logp(self, result, *args, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)

    def _entropy(self, *args, **kwargs):
        return tf.constant(0.0, dtype=tf.float32)
        
    def attach_q(self, qdist):
        raise Exception("cannot attach an explicit Q distribution to a deterministic transform. attach to the parent instead!")

    def observe(self, observed_val):
        raise Exception("cannot explicitly observe a deterministic transformed value. ")
    
    def default_q(self, **kwargs):
        """
        Automatically construct a Q distribution as a deterministic transform of parent Qs.
        """
        qs = {inp : n.q_distribution() for (inp, n) in self.inputs_random.items()}
        qs.update(self.inputs_nonrandom)
        qs.update(kwargs)
        return type(self)(name="q_"+self.name, **qs)
    

class UnaryTransform(DeterministicTransform):
    """
    Define a random variable as the deterministic transform of another RV 
    already present in the model. Variables of this type are treated as a 
    special case by the joint model: they cannot be given their own Q
    distributions, but are automatically given a Q distribution corresponding 
    to a deterministic transform of the parent's Q distribution. 
    """
    
    def __init__(self, A, transform, **kwargs):
        self.transform=transform
        assert(isinstance(A, ConditionalDistribution))
        super(UnaryTransform, self).__init__(A=A, **kwargs)

        if transform.is_structural():
            # pass through transformed elementwise params
            transformed = {}
            for inp_name, shape in A.input_shapes.items():
                inp = getattr(A, inp_name)
                if shapes_equal(shape, A.shape):
                # if util.shapes_equal(shape, A.shape):
                    transformed[inp_name] = transform.transform(inp)
                elif shape_is_scalar(shape):
                # elif util.shape_is_scalar(shape):
                    transformed[inp_name] = inp

            try:
                derived = A.derived_parameters(**transformed)
            except Exception as e:
                print("could not derive additional parameters for structural transform %s of %s: %s" % (transform, A, e))
                derived = {}

            self.__dict__.update(transformed)
            self.__dict__.update(derived)
            
    def inputs(self):
        d = {"A": None}
        return d

    def _sample(self, A):
        tA = self.transform.transform(A)
        return tA

    def _compute_shape(self, A_shape):
        return self.transform.output_shape(A_shape)

    def _compute_dtype(self, A_dtype):
        return A_dtype
    
    def observe(self, observed_val):
        transformed = self.transform.inverse(observed_val)
        self.inputs_random["A"].observe(transformed)

    def default_q(self):
        q_A = self.inputs_random["A"].q_distribution()    
        return UnaryTransform(q_A, self.transform, name="q_"+self.name)

    def is_gaussian(self):
        return self.transform.is_structural() and self.A.is_gaussian()


class TransformedDistribution(ConditionalDistribution):

    """
    Define a new distribution as a deterministic transform of a given 
    source distribution. Unlike the DeterministicTransform, which transforms
    a RV already present in the model, this creates a new random variable
    having the density of the transformed distribution. 

    """
    
    def __init__(self, dist, transform, **kwargs):

        # we assume dist is an *instance* of a ConditionalDist class that is
        # not currently part of any model, but will have some set of
        # (random and/or nonrandom) inputs that will be absorbed into the
        # TransformedDistribution.
        # As a convenience, we allow passing in an abstract class, e.g. Gaussian,
        # and instantiate the class ourselves with arguments passed into
        # the TransformedDistribution
        if isinstance(dist, type):
            if "shape" in kwargs:
                inp_shape = transform.input_shape(kwargs["shape"])
                del kwargs["shape"]
            else:
                inp_shape = None
            self.dist = dist(shape=inp_shape, **kwargs)
        else:
            self.dist = dist

        self.transform=transform
        super(TransformedDistribution, self).__init__(**kwargs)


        for inp_name, shape in self.dist.input_shapes.items():
            if transform.is_structural() and shape==self.dist.shape:
                # for structural transforms, pass through
                # transformed elementwise params
                inp = getattr(self.dist, inp_name)
                setattr(self, inp_name, transform.transform(inp))
            else:
                # don't pass through params in any other cases
                # (so we need to delete the values set by the constructor)
                delattr(self, inp_name)

        
    def _setup_inputs(self, **kwargs):
        self.inputs_random = self.dist.inputs_random
        self.inputs_nonrandom = self.dist.inputs_nonrandom
        return {}
        
    def _sample_and_entropy(self, **kwargs):
        sample, logjac = self.transform.transform(self.dist._sampled, return_log_jac=True)
        self._sampled_log_jacobian = logjac
        entropy = self.dist._sampled_entropy + logjac
        return sample, entropy
        
    def inputs(self):
        return self.dist.inputs()

    def _compute_shape(self, **kwargs):
        return self.transform.output_shape(self.dist.shape)

    def _compute_dtype(self, **kwargs):
        return self.dist.dtype
    
    def _sample(self, **kwargs):
        ds = self.dist._sample(**kwargs)
        return self.transform.transform(ds)

    def _logp(self, result, **kwargs):
        inverted, inverse_logjac = self.transform.inverse(result, return_log_jac=True)
        return self.dist._logp(inverted, **kwargs) + inverse_logjac
        
    def _entropy(self, *args, **kwargs):
        return self.dist._entropy(*args, **kwargs) + self._sampled_log_jacobian

    def default_q(self, **kwargs):
        dvm = self.dist.default_q()
        return TransformedDistribution(dvm, self.transform, name="q_"+self.name)

#############################################################################
    
class Transform(object):

    """
    Abstract class representing a deterministic transformation of a matrix. 
    Subclasses should implement the transform() method, including the option
    to return the log jacobian determinant, and the output_shape() method. 
    Invertible transforms should also implement inverse() and input_shape(). 
    
    Transform objects contain no state and are never instantiated, all methods
    are static only. (we use Python classmethods to allow a static method to 
    call other static methods of the same class (used, e.g., in implementing 
    SelfInverseTransform). 
    """
    
    @classmethod
    def transform(cls, x, return_log_jac=False, **kwargs):
        raise NotImplementedError

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        raise NotImplementedError

    @classmethod
    def output_shape(cls, input_shape):
        # default to assuming a pointwise transform
        return input_shape

    @classmethod
    def input_shape(cls, output_shape):
        # default to assuming a pointwise transform
        return output_shape

    @classmethod
    def is_structural(cls):
        # a 'structural' transform is one that simply slices, permutes, or otherwise
        # rearranges the elements of its input. For example, tranpose is structural, as
        # is 'get the first column', while 'add 2 to every element' is not.
        # The DeterministicTransform class propagates input parameters (mean, std, etc)
        # through structural transformations, but not otherwise. 
        return False

class SelfInverseTransform(Transform):

    """
    Base class for transforms that are their own inverse (transpose, reciprocal, etc.).
    """
    
    @classmethod
    def inverse(cls, *args, **kwargs):
        return cls.transform(*args, **kwargs)

    @classmethod
    def input_shape(cls, output_shape):
        return cls.output_shape(output_shape)
    
class Logit(Transform):

    """
    Map from the real line to the unit interval using the logistic sigmoid fn. 
    """
    
    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_logit_input")
        transformed = 1.0 / (1 + tf.exp(-x))
        
        if return_log_jac:
            jacobian = transformed * (1-transformed)
            if clip_finite:
                jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
            log_jacobian = tf.reduce_sum(tf.log(jacobian))
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        x = tf.log(1./transformed - 1.0)

        if return_log_jac:
            jacobian = transformed * (1-transformed)
            if clip_finite:
                jacobian = tf.clip_by_value(jacobian, 1e-45, 1e38, name="clipped_jacobian")
            log_jacobian = -tf.reduce_sum(tf.log(jacobian))
            return x, log_jacobian
        else:
            return x

class RowNormalize(Transform):

    @classmethod
    def transform(cls, x_positive, return_log_jac=False, **kwargs):
        try:
            n, k = extract_shape(x_positive)
            Z = tf.expand_dims(tf.reduce_sum(x_positive, axis=1), axis=1)
        except ValueError: # x is just a vector
            k, = extract_shape(x_positive)
            Z = tf.reduce_sum(x_positive)
            
        transformed = x_positive / Z
        if return_log_jac:
            log_jacobian = -k * tf.reduce_sum(tf.log(Z))
            return transformed, log_jacobian
        else:
            return transformed
        
class UnitColumn(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, **kwargs):
        try:
            n, k = x.get_shape()
            ones = tf.ones((n,1))
            transformed = tf.concat(1, (x, ones))
        except ValueError:
            k = x.get_shape()
            transformed = tf.concat(0, (x, 1))
        
        if return_log_jac:
            return transformed, 0
        else:
            return transformed

    @classmethod
    def output_shape(cls, input_shape):
        try:
            n, k = input_shape
            return (n, k+1)
        except ValueError:
            k, = input_shape
            return (k+1,)
        

class RowNormalize1(Transform):

    # equivalent to compose(UnitColumn, RowNormalize), but
    # defined as a single unit so we can specify an inverse
    
    @classmethod
    def transform(cls, x_positive, return_log_jac=False, **kwargs):
        try:
            n, k = extract_shape(x_positive)
            ones = tf.ones((n,1))
            expanded = tf.concat(1, (x_positive, ones))
            Z = tf.expand_dims(tf.reduce_sum(expanded, axis=1), axis=1)
        except ValueError: # x is just a vector
            k, = extract_shape(x_positive)
            expanded = tf.concat(0, (x, 1))
            Z = tf.reduce_sum(expanded)
            
        transformed = expanded / Z
        if return_log_jac:
            log_jacobian = -k * tf.reduce_sum(tf.log(Z))
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def output_shape(cls, input_shape):
        try:
            n, k = input_shape
            return (n, k+1)
        except ValueError:
            k, = input_shape
            return (k+1,)

        
    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):

        # first scale to have a unit column
        cols = tf.unstack(transformed, axis=1)
        k = len(cols)
        last_col = cols[-1]
        x = tf.stack( [col / last_col for col in cols[:-1]], axis=1)

        if return_log_jac:
            log_jacobian = -(k-1) * tf.reduce_sum(tf.log(last_col))
            return x, log_jacobian
        else:
            return x

    @classmethod
    def input_shape(cls, output_shape):
        try:
            n, k = output_shape
            return (n, k-1)
        except ValueError:
            k, = output_shape
            return (k-1,)

        
class Exp(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_exp_input")

        transformed = tf.exp(x)
        if return_log_jac:
            log_jacobian = tf.reduce_sum(x)
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            transformed = tf.clip_by_value(transformed, 1e-45, 1e38, name="clipped_log_input")
        x = tf.log(transformed)

        if return_log_jac:
            log_jacobian = -tf.reduce_sum(x)
            return x, log_jacobian
        else:
            return x


class Log1Exp(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            x = tf.clip_by_value(x, -88, 88, name="clipped_exp_input")

        # y = log(1 + exp(x))
        # dy/dx = exp(x) / (1 + exp(x))
        transformed = tf.log(1 + tf.exp(x))
        if return_log_jac:
            log_jacobian = tf.reduce_sum(x - transformed)
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            transformed = tf.clip_by_value(transformed, 1e-45, 1e38, name="clipped_log_input")

        # x = log(exp(y)-1)
        # dx/dy = exp(y)/(exp(y)-1)
        x = tf.log(tf.exp(transformed) - 1)
        if return_log_jac:
            log_jacobian = tf.reduce_sum(transformed - x)
            return x, log_jacobian
        else:
            return x

        
class Square(Transform):

    @classmethod
    def transform(cls, x, return_log_jac=False, **kwargs):
        transformed = tf.square(x)

        if return_log_jac:
            log_jacobian = tf.reduce_sum(tf.log(x)) + np.log(2)
            return transformed, log_jacobian
        else:
            return transformed

    @classmethod
    def inverse(cls, transformed, return_log_jac=False, **kwargs):
        x = tf.sqrt(transformed)
        if return_log_jac:
            jacobian = .5/x
            log_jacobian = tf.reduce_sum(tf.log(jacobian))
            return x, log_jacobian
        else:
            return x

        
class Reciprocal(SelfInverseTransform):

    @classmethod
    def transform(cls, x, return_log_jac=False, clip_finite=True, **kwargs):
        if clip_finite:
            # caution: assumes input is positive
            x = tf.clip_by_value(x, 1e-38, 1e38, name="clipped_reciprocal_input")
            
        nlogx = -tf.log(x)
        transformed = tf.exp(nlogx)
        
        if return_log_jac:
            log_jacobian = 2*tf.reduce_sum(nlogx)
            return transformed, log_jacobian
        else:
            return transformed


class Transpose(SelfInverseTransform):
    # todo should there be a special property for permutation
    # transforms, so that we also transform means, variances, etc?

    @classmethod
    def transform(cls, x, return_log_jac=False):
        transformed = tf.transpose(x)
        if return_log_jac:
            return transformed, 0.0
        else:
            return transformed

    @classmethod
    def output_shape(cls, input_shape):
        N, M = input_shape
        return (M, N)

    @classmethod
    def is_structural(cls):
        return True
    
    
def invert_transform(source):
    """
    Given a Transform class, return a class 
    with methods swapped to perform the inverse transform.
    """
    
    class Inverted(Transform):

        @classmethod
        def transform(cls, *args, **kwargs):
            return source.inverse(*args, **kwargs)

        @classmethod
        def inverse(cls, *args, **kwargs):
            return source.transform(*args, **kwargs)

        @classmethod
        def output_shape(cls, *args, **kwargs):
            return source.input_shape(*args, **kwargs)

        @classmethod
        def input_shape(cls, *args, **kwargs):
            return source.output_shape(*args, **kwargs)

        @classmethod
        def is_structural(cls):
            return source.is_structural()

    return Inverted

def chain_transforms(*transforms):
    
    class Chain(Transform):

        @classmethod
        def transform(cls, x, return_log_jac=False):
            log_jacs = []
            for transform in transforms:
                if return_log_jac:
                    x, lj = transform.transform(x, return_log_jac=return_log_jac)
                    log_jacs.append(lj)
                else:
                    x = transform.transform(x, return_log_jac=return_log_jac)
            if return_log_jac:
                return x, tf.reduce_sum(tf.stack(log_jacs))
            else:
                return x

        @classmethod
        def inverse(cls, transformed, return_log_jac=False):
            log_jacs = []
            for transform in transforms[::-1]:
                if return_log_jac:
                    transformed, lj = transform.inverse(transformed, return_log_jac=return_log_jac)
                    log_jacs.append(lj)
                else:
                    transformed = transform.inverse(transformed, return_log_jac=return_log_jac)

            if return_log_jac:
                return transformed, tf.reduce_sum(tf.pack(log_jacs))
            else:
                return transformed

        @classmethod
        def output_shape(cls, input_shape):
            for transform in transforms:
                input_shape = transform.output_shape(input_shape)
            return input_shape

        @classmethod
        def input_shape(cls, output_shape):
            for transform in transforms[::-1]:
                output_shape = transform.input_shape(output_shape)
            return output_shape

        @classmethod
        def is_structural(cls):
            is_structural = True
            for transform in transforms[::-1]:
                is_structural *= transform.is_structural()
            return is_structural

    return Chain

# define some common transforms by composing the base transforms defined above
Sqrt = invert_transform(Square)
Log = invert_transform(Exp)
Reciprocal_Sqrt = chain_transforms(Reciprocal, Sqrt)
Reciprocal_Square = chain_transforms(Reciprocal, Square)
Exp_Reciprocal = chain_transforms(Exp, Reciprocal)

ColNormalize1 = chain_transforms(Transpose, RowNormalize1, Transpose)
ColNormalize = chain_transforms(Transpose, RowNormalize, Transpose)

Simplex1 = chain_transforms(Exp, RowNormalize1)
Simplex1Col = chain_transforms(Exp, ColNormalize1)

Simplex_Raw = chain_transforms(Exp, RowNormalize)
Simplex_Raw_Col = chain_transforms(Exp, ColNormalize)


class Simplex(Transform):
    # result is invariant to shifting the (logspace) input,
    # so we choose a shift to avoid overflow

    @classmethod
    def transform(cls, x, **kwargs):
        xmax = tf.expand_dims(tf.reduce_max(x, axis=1), axis=1)
        return Simplex_Raw.transform(x-xmax, **kwargs)

class SimplexCol(Transform):
    # result is invariant to shifting the (logspace) input,
    # so we choose a shift to avoid overflow

    @classmethod
    def transform(cls, x, **kwargs):
        xmax = tf.expand_dims(tf.reduce_max(x, axis=0), axis=0)
        return Simplex_Raw_Col.transform(x-xmax, **kwargs)


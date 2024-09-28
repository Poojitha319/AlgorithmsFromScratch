import numpy as np
EPSILON = 1e-7


def normal(shape, scale=0.5):
    """
    Initializes weights with a normal (Gaussian) distribution.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        scale (float, optional): Standard deviation of the normal distribution. Defaults to 0.5.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    return np.random.normal(loc=0.0, scale=scale, size=shape)


def uniform(shape, scale=0.5):
    """
    Initializes weights with a uniform distribution.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        scale (float, optional): Scale factor for the uniform distribution range [-scale, scale]. Defaults to 0.5.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    return np.random.uniform(low=-scale, high=scale, size=shape)


def zero(shape, **kwargs):
    """
    Initializes weights with zeros.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    return np.zeros(shape)


def one(shape, **kwargs):
    """
    Initializes weights with ones.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    return np.ones(shape)


def orthogonal(shape, scale=0.5):
    """
    Initializes weights using orthogonal initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        scale (float, optional): Scaling factor for the orthogonal matrix. Defaults to 0.5.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initialization requires at least two dimensions")
    
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q * scale


def _glorot_fan(shape):
    """
    Computes the fan_in and fan_out for Glorot and He initializations.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        tuple: (fan_in, fan_out)
    """
    assert len(shape) >= 2, "Shape must have at least two dimensions"
    
    if len(shape) == 4:
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in, fan_out = shape[:2]
    return float(fan_in), float(fan_out)


def glorot_normal(shape, **kwargs):
    """
    Initializes weights using Glorot (Xavier) normal initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, fan_out = _glorot_fan(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return normal(shape, scale=std)


def glorot_uniform(shape, **kwargs):
    """
    Initializes weights using Glorot (Xavier) uniform initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, fan_out = _glorot_fan(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, scale=limit)


def he_normal(shape, **kwargs):
    """
    Initializes weights using He normal initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, _ = _glorot_fan(shape)
    std = np.sqrt(2.0 / fan_in)
    return normal(shape, scale=std)


def he_uniform(shape, **kwargs):
    """
    Initializes weights using He uniform initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, _ = _glorot_fan(shape)
    limit = np.sqrt(6.0 / fan_in)
    return uniform(shape, scale=limit)


def lecun_normal(shape, **kwargs):
    """
    Initializes weights using LeCun normal initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, _ = _glorot_fan(shape)
    std = np.sqrt(1.0 / fan_in)
    return normal(shape, scale=std)


def lecun_uniform(shape, **kwargs):
    """
    Initializes weights using LeCun uniform initialization.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, _ = _glorot_fan(shape)
    limit = np.sqrt(3.0 / fan_in)
    return uniform(shape, scale=limit)


def sparse(shape, sparsity=0.1, scale=0.01, **kwargs):
    """
    Initializes weights with sparse connections.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        sparsity (float, optional): Proportion of non-zero elements. Defaults to 0.1.
        scale (float, optional): Scale of the non-zero elements. Defaults to 0.01.
        
    Returns:
        np.ndarray: Initialized weights with specified sparsity.
    """
    weight = np.zeros(shape)
    num_elements = np.prod(shape)
    num_nonzeros = int(num_elements * sparsity)
    indices = np.unravel_index(
        np.random.choice(num_elements, num_nonzeros, replace=False),
        shape
    )
    weight[indices] = np.random.normal(0.0, scale, size=num_nonzeros)
    return weight


def identity(shape, **kwargs):
    """
    Initializes weights as identity matrices. Only applicable for 2D square matrices.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        
    Returns:
        np.ndarray: Initialized identity matrix.
    """
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Identity initialization requires a 2D square shape")
    return np.identity(shape[0])


def variance_scaling(shape, scale=1.0, mode='fan_in', distribution='normal', **kwargs):
    """
    Initializes weights using variance scaling.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        scale (float, optional): Scaling factor. Defaults to 1.0.
        mode (str, optional): One of 'fan_in', 'fan_out', 'fan_avg'. Defaults to 'fan_in'.
        distribution (str, optional): 'normal' or 'uniform'. Defaults to 'normal'.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    fan_in, fan_out = _glorot_fan(shape)
    
    if mode == 'fan_in':
        denominator = fan_in
    elif mode == 'fan_out':
        denominator = fan_out
    elif mode == 'fan_avg':
        denominator = (fan_in + fan_out) / 2
    else:
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'")
    
    if distribution == 'normal':
        std = np.sqrt(scale / denominator)
        return normal(shape, scale=std)
    elif distribution == 'uniform':
        limit = np.sqrt(3.0 * scale / denominator)
        return uniform(shape, scale=limit)
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")


def constant(shape, value=0.0, **kwargs):
    """
    Initializes weights with a constant value.
    
    Args:
        shape (tuple): Shape of the weights tensor.
        value (float, optional): Constant value to initialize. Defaults to 0.0.
        
    Returns:
        np.ndarray: Initialized weights.
    """
    return np.full(shape, fill_value=value)


def get_initializer(name):
    initializers = {
        'normal': normal,
        'uniform': uniform,
        'zero': zero,
        'one': one,
        'orthogonal': orthogonal,
        'glorot_normal': glorot_normal,
        'glorot_uniform': glorot_uniform,
        'he_normal': he_normal,
        'he_uniform': he_uniform,
        'lecun_normal': lecun_normal,
        'lecun_uniform': lecun_uniform,
        'sparse': sparse,
        'identity': identity,
        'variance_scaling': variance_scaling,
        'constant': constant,
    }
    initializer = initializers.get(name.lower())
    if initializer is None:
        raise ValueError(f"Invalid initialization function name: {name}")
    return initializer



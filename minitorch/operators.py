"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiply two floating point numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The same number x

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floating point numbers.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Negate a floating point number.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The negation of x

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: 1.0 if x < y, 0.0 otherwise

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floating point numbers.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The maximum of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two floating point numbers are close to each other.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: 1.0 if |x - y| < 1e-2, 0.0 otherwise

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The sigmoid of x

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: max(0, x)

    """
    return max(0.0, x)


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
    ----
        x (float): Input number (must be positive)

    Returns:
    -------
        float: The natural logarithm of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.`

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: e^x

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: The gradient of log(x) * d

    """
    return d / x


def inv(x: float) -> float:
    """Compute the inverse of a number.

    Args:
    ----
        x (float): Input number (must not be zero)

    Returns:
    -------
        float: 1 / x

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: The gradient of (1/x) * d

    """
    return -(d / (x * x))


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: The gradient of ReLU(x) * d

    """
    return d if x > 0 else 0.0


# TODO: Implement for Task 0.1.
# ... existing imports and comments ...

# ... rest of the file ...

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element in an iterable.

    Args:
    ----
        fn (Callable[[float], float]): Function to apply
        ls (Iterable[float]): Input iterable

    Returns:
    -------
        Iterable[float]: Iterable with function applied to each element

    """
    return (fn(x) for x in ls)


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Combine two iterables element-wise using a function.

    Args:
    ----
        fn (Callable[[float, float], float]): Function to combine elements
        ls1 (Iterable[float]): First input iterable
        ls2 (Iterable[float]): Second input iterable

    Returns:
    -------
        Iterable[float]: Iterable with elements combined using the function

    """
    return (fn(x, y) for x, y in zip(ls1, ls2))


def reduce(
    fn: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Reduce an iterable to a single value using a function.

    Args:
    ----
        fn (Callable[[float, float], float]): Function to reduce elements
        ls (Iterable[float]): Input iterable
        initial (float): Initial value for the reduction

    Returns:
    -------
        float: Reduced value

    Raises:
    ------
        ValueError: If the input iterable is empty

    """
    result = initial
    for element in ls:
        result = fn(result, element)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in an iterable.

    Args:
    ----
        ls (Iterable[float]): Input iterable

    Returns:
    -------
        Iterable[float]: Iterable with each element negated

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements in two iterables.

    Args:
    ----
        ls1 (Iterable[float]): First input iterable
        ls2 (Iterable[float]): Second input iterable

    Returns:
    -------
        Iterable[float]: Iterable with corresponding elements added

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in an iterable.

    Args:
    ----
        ls (Iterable[float]): Input iterable

    Returns:
    -------
        float: Sum of all elements

    """
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in an iterable.

    Args:
    ----
        ls (Iterable[float]): Input iterable

    Returns:
    -------
        float: Product of all elements

    """
    return reduce(mul, ls, 1.0)

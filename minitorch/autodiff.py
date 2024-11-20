from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    up_vals, low_vals = list(vals), list(vals)
    up_vals[arg] += epsilon
    low_vals[arg] -= epsilon

    slope = (f(*up_vals) - f(*low_vals)) / (2 * epsilon)
    return slope


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative with respect to this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Whether the variable is a leaf."""
        ...

    def is_constant(self) -> bool:
        """Whether the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Iterable of parent variables that created this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Iterable of tuples of parent variables and their derivative with respect to this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph."""

    # Task 1.4.
    def dfs(node: Variable) -> None:
        if node.unique_id in visited or node.is_constant():
            return
        visited.add(node.unique_id)
        for parent in node.parents:
            dfs(parent)
        order.append(node)

    visited = set()
    order = []
    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Performs backpropagation on the computation graph."""
    # Task 1.4.
    order = topological_sort(variable)
    derivatives = dict()
    derivatives[variable.unique_id] = deriv

    for node in order:
        local_deriv = derivatives[node.unique_id]

        if node.is_leaf():
            node.accumulate_derivative(local_deriv)
        else:
            parent_derivs = node.chain_rule(local_deriv)
            for parent, p_deriv in parent_derivs:
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += p_deriv
                else:
                    derivatives[parent.unique_id] = p_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Tuple of saved values that need to be used during backpropagation."""
        return self.saved_values

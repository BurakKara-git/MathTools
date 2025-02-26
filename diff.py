from numpy import cos, sin, exp, log, absolute, sign


class DualNumber:
    """Represents a dual number for automatic differentiation."""

    def __init__(self, real, dual):
        self.real = real  # Function value
        self.dual = dual  # Derivative value

    def __str__(self):
        return f"{self.real}, {self.dual}E"

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real, self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)

    __rsub__ = __sub__

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(
                self.real * other.real, self.dual * other.real + self.real * other.dual
            )
        else:
            return DualNumber(self.real * other, self.dual * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if other == 0 or (isinstance(other, DualNumber) and other.real == 0):
            raise ZeroDivisionError("Division by Zero")

        if isinstance(other, DualNumber):
            real = self.real / other.real
            dual = (self.dual * other.real - self.real * other.dual) / (other.real**2)
            return DualNumber(real, dual)
        else:
            return DualNumber(self.real / other, self.dual / other)

    def __rtruediv__(self, other):
        if self.real == 0:
            raise ZeroDivisionError("Division by Zero")

        if isinstance(other, DualNumber):
            real = other.real / self.real
            dual = (other.dual * self.real - other.real * self.dual) / (self.real**2)
            return DualNumber(real, dual)
        else:
            return DualNumber(other / self.real, (-other * self.dual) / (self.real**2))

    def __pow__(self, power: int):
        """Computes power of a dual number."""
        result = DualNumber(self.real, self.dual)
        for _ in range(power - 1):
            result *= self
        return result


class Dexp(DualNumber):
    """Computes e^x for a dual number."""

    def __init__(self, dual_number):
        super().__init__(
            exp(dual_number.real), dual_number.dual * exp(dual_number.real)
        )


class Dlog(DualNumber):
    """Computes log(x) for a dual number."""

    def __init__(self, dual_number):
        if isinstance(dual_number.real, complex):
            pass
        elif dual_number.real <= 0:
            raise ValueError("Logarithm is undefined for non-positive values.")
        super().__init__(log(dual_number.real), dual_number.dual / dual_number.real)


class Dcos(DualNumber):
    """Computes cos(x) for a dual number."""

    def __init__(self, dual_number):
        super().__init__(
            cos(dual_number.real), -dual_number.dual * sin(dual_number.real)
        )


class Dsin(DualNumber):
    """Computes sin(x) for a dual number."""

    def __init__(self, dual_number):
        super().__init__(
            sin(dual_number.real), dual_number.dual * cos(dual_number.real)
        )


class Dabs(DualNumber):
    """Computes absolute value |x| for a dual number."""

    def __init__(self, dual_number):
        if dual_number.real == 0:
            raise ValueError("Derivative of Absolute value is undefined at 0")
        if isinstance(dual_number.real, complex):
            raise ValueError("Complex Absolute Value is nowhere differentiable")
        else:
            super().__init__(
                absolute(dual_number.real), dual_number.dual * sign(dual_number.real)
            )


def diff(f, x_value):
    """
    Computes the derivative of function f at x = x_value using Dual Numbers.

    Args:
        f (function): Function to differentiate.
        x_value (float): The point at which to evaluate the derivative.

    Returns:
        float: The derivative of f at x_value.
    """
    return f(DualNumber(x_value, 1)).dual


if __name__ == "__main__":
    
    def foo1(x):
        """
        Example function: f(x) = e^(e^x * sin(x) / |x|) + x^2 - x + 5 + 1/x - log(x**2) + cos(x) - abs(x).

        Args:
            x (DualNumber): Dual number input.

        Returns:
            DualNumber: The result of f(x).
        """
        return Dexp(Dexp(x) * Dsin(x)) + x**2 - 5 * x + 5 + 1 / x - Dlog(x**2) + Dcos(x) - Dabs(x)

    def foo2(x):
        """
        Example function: f(x) = e^(e^x * sin(x) / |x|) + x^2 - x + 5 + 1/x - log(x**2) + cos(x).

        Args:
            x (DualNumber): Dual number input.

        Returns:
            DualNumber: The result of f(x).
        """
        return Dexp(Dexp(x) * Dsin(x)) + x**2 - 5 * x + 5 + 1 / x - Dlog(x**2) + Dcos(x)

    x = 10.0  # Example real value
    result = diff(foo1, x)
    print(f"Derivative at x = {x}: {result}")

    x = 10.0 + 5j  # Example complex value
    result = diff(foo2, x)
    print(f"Derivative at x = {x}: {result}")

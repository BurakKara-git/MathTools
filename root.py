from diff import diff


def bisection(f, a, b, epsilon):
    """
    Bisection method to find a root of f(x) in the interval [a, b].
    The function f(x) must be continuous and f(a) * f(b) ≤ 0.

    Args:
        f (function): The function for which the root is to be found.
        a (float): Left endpoint of the interval.
        b (float): Right endpoint of the interval.
        epsilon (float): Desired precision.

    Returns:
        float: Approximate root of f(x).
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function values at endpoints must have opposite signs.")

    while abs(b - a) > epsilon:
        p = (a + b) / 2  # Midpoint
        if abs(f(p)) <= epsilon:  # Found root
            return p
        elif f(a) * f(p) < 0:  # Root is in [a, p]
            b = p
        else:  # Root is in [p, b]
            a = p

    return (a + b) / 2  # Return best approximation


def newton(f, epsilon, x0=None, max_iter=100):
    """
    Newton's method to find a root of f(x).
    The function f(x) must be differentiable near its root.

    Args:
        f (function): The function whose root is to be found.
        epsilon (float): Desired precision.
        x0 (float, optional): Initial guess. Defaults to midpoint from bisection if None.
        max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        float: Approximate root of f(x).
    """
    if x0 is None:
        x0 = bisection(f, -1e10, 1e10, 1e-2)  # Estimate root if not given

    x = x0  # Initial guess
    for _ in range(max_iter):
        fx = f(x)
        dfx = diff(f, x)

        if abs(fx) <= epsilon:
            return x  # Root found
        if dfx == 0:  # Avoid division by zero
            x += epsilon
            continue

        x -= fx / dfx  # Newton's update

    raise RuntimeError("Newton's method did not converge.")


if __name__ == "__main__":

    def foo(x):
        return -1000 + x**2 - x**3 - 1  # Example function

    epsilon = 1e-10  # Precision
    x_min = -1e10  # Choose better ranges
    x_max = +1e10  # Choose better ranges

    try:
        newton_result = newton(foo, epsilon)
        print("Newton's method root:", newton_result)
    except RuntimeError as e:
        print("Newton's method error:", e)

    try:
        bisection_result = bisection(foo, x_min, x_max, epsilon)
        print("Bisection method root:", bisection_result)
    except ValueError as e:
        print("Bisection method error:", e)

    # To find local maximum or minimum values try:
    diff_foo = lambda x: diff(foo, x)

    try:
        newton_result_diff = newton(diff_foo, epsilon, 10)
        print("Newton's method local max/min:", newton_result_diff)
    except RuntimeError as e:
        print("Newton's method error:", e)

    try:
        bisection_result_diff = bisection(diff_foo, x_min, x_max, epsilon)
        print("Bisection method local max/min:", bisection_result_diff)
    except ValueError as e:
        print("Bisection method error:", e)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def halton_sequence(base: int, count: int, skip: int = 20) -> list[int]:
    result = []
    n, d = 0, 1
    for i in range(count + skip):
        x = d - n
        if x == 1:
            n = 1
            d *= base
        else:
            y = d // base
            while x <= y:
                y //= base
            n = (base + 1) * y - x
        if i > skip:
            result.append(n / d)
    return result


def nth_prime_number(n: int) -> int:
    # fmt: off
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
        83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
        179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
        271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373,
        379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
        479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
        599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691,
        701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821,
        823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
        941, 947, 953, 967, 971, 977, 983, 991, 997
    ]

    return primes[n]


def generate_quasi_random_numbers(
    dim: int, n: int, l_bounds: np.ndarray, u_bounds: np.ndarray
) -> np.ndarray:
    rng = np.random.default_rng()

    sample = np.array(
        [halton_sequence(base=nth_prime_number(i), count=n) for i in range(dim)]
    ).transpose()

    sample = rng.permuted(sample, axis=1)
    return sample * (u_bounds - l_bounds) + l_bounds


def rank_values(values: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(values, axis=1)
    return np.argsort(distance)


def monte_carlo_step(
    a: np.ndarray,
    b: np.ndarray,
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, float]:
    num_variables = b.shape[0]
    points = generate_quasi_random_numbers(
        num_variables, point_count, l_bounds, u_bounds
    )
    values = points @ a.T - b
    ranks = rank_values(values)
    points = points[ranks]
    best_value = values[ranks[0]]
    return points, best_value


def get_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def get_cube_polygon(
    xyz: np.ndarray,
    sizes: np.ndarray,
    face_colors: str = "green",
    edge_colors: str = "darkgreen",
    alpha: float = 0.25,
    label: str | None = None,
) -> Poly3DCollection:
    faces = np.array(
        [
            [[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]],
        ]
    )

    return Poly3DCollection(
        faces * sizes + xyz,
        facecolors=face_colors,
        edgecolors=edge_colors,
        alpha=alpha,
        label=label,
    )


def visualise_step(
    feasible: np.ndarray,
    discarded: np.ndarray,
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    iteration_no: int,
):
    fig = plt.figure(iteration_no)
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        feasible[:, 0],
        feasible[:, 1],
        feasible[:, 2],
        color="green",
        label="Лучшие точки",
    )
    ax.scatter(
        discarded[:, 0], discarded[:, 1], discarded[:, 2], color="red", alpha=0.1
    )

    feasible_space = get_cube_polygon(
        l_bounds, np.abs(u_bounds - l_bounds), label="Пр. поиска на след. шаге"
    )

    ax.add_collection3d(feasible_space)

    ax.set_xlabel("Ось X")
    ax.set_ylabel("Ось Y")
    ax.set_zlabel("Ось Z")

    ax.xaxis.labelpad = 12
    ax.yaxis.labelpad = 12
    ax.zaxis.labelpad = 12

    ax.set_title(f"Пространство поиска на {iteration_no + 1}-ом шаге")
    ax.legend()
    plt.show()


def visualise_best_values(best_values: np.ndarray) -> None:
    plt.plot(np.linalg.norm(best_values, axis=1))
    plt.title("Лучшее расстояние до нуля")
    plt.xlabel("Шаг")
    plt.ylabel("Расстояние")
    plt.show()


def monte_carlo_solve(
    a: np.ndarray,
    b: np.ndarray,
    l_bounds: np.ndarray,
    u_bounds: np.ndarray,
    epsilon: float = 0.001,
    max_iterations: int = 20,
    points_count: int = 200,
    feasible_count: int = 10,
) -> np.ndarray | None:
    best_values = []

    for i in range(max_iterations):
        step_points, best_value = monte_carlo_step(
            a, b, l_bounds, u_bounds, points_count
        )
        best_values.append(best_value)
        if np.all(np.abs(best_value) < epsilon):
            break

        feasible = step_points[:feasible_count]
        discarded = step_points[feasible_count:]

        l_bounds, u_bounds = get_bounds(feasible)

        visualise_step(feasible, discarded, l_bounds, u_bounds, iteration_no=i)
    else:
        return None

    visualise_best_values(np.array(best_values))

    return step_points[0]


def main():
    points_count = 1000
    epsilon = 0.1
    max_iterations = 20
    feasible_count = 10

    a = np.array([[1, -5, 3], [2, 4, 1], [-3, 3, -7]])
    b = np.array([-1, 6, -13])

    l_bounds = np.array([-100, -100, -100])
    u_bounds = np.array([100, 100, 100])

    solution = monte_carlo_solve(
        a,
        b,
        l_bounds,
        u_bounds,
        points_count=points_count,
        max_iterations=max_iterations,
        feasible_count=feasible_count,
        epsilon=epsilon,
    )

    if solution is None:
        print("Не удалось найти решение за указанное количество итераций!")
    else:
        print(f"Final solution: {solution}")


if __name__ == "__main__":
    main()

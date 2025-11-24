"""
Performance benchmark for non-dominated sorting implementations.
"""
import numpy as np
import time

from pymoo.functions.standard.non_dominated_sorting import fast_non_dominated_sort


def generate_test_data(n_points, n_objectives, seed=42):
    """Generate random test data for performance comparison"""
    np.random.seed(seed)
    return np.random.rand(n_points, n_objectives) * 100


def benchmark_function(func, F, runs=3):
    """Benchmark a function with multiple runs"""
    times = []
    for _ in range(runs):
        start_time = time.perf_counter()
        result = func(F)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return np.mean(times), result


def test_correctness(F):
    """Test if default and optimized produce the same result"""
    result1 = fast_non_dominated_sort(F)
    result2 = fast_non_dominated_sort(F, native_biobj_sorting=True)

    # Sort each front for comparison
    sorted_result1 = [sorted(front) for front in result1]
    sorted_result2 = [sorted(front) for front in result2]

    if len(sorted_result1) != len(sorted_result2):
        return False

    for f1, f2 in zip(sorted_result1, sorted_result2):
        if f1 != f2:
            return False

    return True


def run_benchmark():
    """Run performance benchmark comparing default vs native bi-objective sorting"""
    print("Bi-Objective Non-dominated Sorting Performance Benchmark")
    print("="*70)
    print(f"{'Size':<10} {'Default (ms)':<15} {'Native BiObj (ms)':<18} {'Speedup':<10} {'Correct':<10}")
    print("-"*70)

    sizes = [50, 100, 500, 1000, 2000]
    all_correct = True

    for size in sizes:
        F = generate_test_data(size, 2)

        # Benchmark default
        time_default, _ = benchmark_function(lambda x: fast_non_dominated_sort(x), F)

        # Benchmark native bi-objective
        time_native, _ = benchmark_function(
            lambda x: fast_non_dominated_sort(x, native_biobj_sorting=True), F
        )

        # Check correctness
        is_correct = test_correctness(F)
        all_correct = all_correct and is_correct

        # Calculate speedup
        speedup = time_default / time_native if time_native > 0 else float('inf')

        print(f"{size:<10} {time_default*1000:<15.3f} {time_native*1000:<18.3f} {speedup:<10.2f}x {'Yes' if is_correct else 'No':<10}")

    print("-"*70)
    print(f"All results correct: {'Yes' if all_correct else 'No'}")

    return all_correct


if __name__ == "__main__":
    run_benchmark()

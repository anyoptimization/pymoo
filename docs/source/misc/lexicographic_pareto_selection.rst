Lexicographic Selection from a Pareto Front
==========================================

In multi-objective optimization, the result is often not a single solution, but a set of Pareto-optimal solutions.
This set represents trade-offs between conflicting objectives. However, in practical decision-making, one final
solution is often required.

One simple way to choose a final solution is lexicographic ordering. In this approach, objectives are sorted by
importance. First, solutions are compared by the most important objective. If several solutions are equal or almost
equal, the next objective is used, and so on.

The example below shows a small helper function for selecting one solution from an already obtained Pareto front.
All objectives are assumed to be minimized.

.. code-block:: python

    import numpy as np

    def lexicographic_selection(F, priority_order=None):
        F = np.asarray(F)

        if priority_order is None:
            priority_order = list(range(F.shape[1]))

        order = np.lexsort(tuple(F[:, i] for i in reversed(priority_order)))
        return order[0]

    F = np.array([
        [0.10, 0.80],
        [0.20, 0.40],
        [0.15, 0.55],
        [0.10, 0.90],
    ])

    selected = lexicographic_selection(F, priority_order=[0, 1])

    print("Selected solution index:", selected)
    print("Selected objective values:", F[selected])

In this example, the first objective has the highest priority. The function first compares solutions by the first
objective and then uses the second objective to break ties. This can be useful when a user already has a Pareto front
and needs to select one final compromise solution according to predefined preferences.

If some objectives should be maximized, they can be multiplied by -1 before applying the function.

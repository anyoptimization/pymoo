"""Flowshop scheduling optimization problem."""

from typing import Any

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.util import default_random_state


class FlowshopScheduling(ElementwiseProblem):
    """Flowshop scheduling problem.

    Args:
        processing_times: Matrix where processing_times[i][j] is the processing time
            for job j on machine i.
    """

    def __init__(self, processing_times: np.ndarray, **kwargs: Any) -> None:
        n_machines, n_jobs = processing_times.shape
        self.records = processing_times

        super().__init__(n_var=n_jobs, n_obj=1, xl=0, xu=n_machines, vtype=int, **kwargs)

    def _evaluate(self, x: Any, out: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        out["F"] = self.makespan(x)

    def makespan(self, x: np.ndarray) -> float:
        """Calculate the makespan (total completion time) for a job sequence.

        Args:
            x: Job sequence array.

        Returns:
            The makespan value.
        """
        machine_times = self.get_machine_times(x)
        # The makespan is the difference between the starting time of the first job
        # and the latest finish time of any job. Minimizing the makespan amounts to
        # minimizing the total time it takes to process all jobs from start to finish.
        makespan = machine_times[-1][-1] + self.records[-1][x[-1]]  # finish time of the last job
        return makespan

    def get_machine_times(self, x: np.ndarray) -> list:
        """Calculate the machine processing times for a given job sequence.

        Args:
            x: Job sequence array.

        Returns:
            List of lists containing the starting time for each job on each machine.
        """
        n_machines, n_jobs = self.records.shape

        # A 2d array to store the starting time for each job on each machine
        # machine_times[i][j] --> starting time for the j-th job on machine i
        machine_times: list = [[] for _ in range(n_machines)]

        # Assign the initial job to the machines
        machine_times[0].append(0)
        for i in range(1, n_machines):
            # Start the next job when the previous one is finished
            machine_times[i].append(machine_times[i - 1][0] + self.records[i - 1][x[0]])

        # Assign the remaining jobs
        for j in range(1, n_jobs):
            # For the first machine, we can put a job when the previous one is finished
            machine_times[0].append(machine_times[0][j - 1] + self.records[0][x[j - 1]])

            # For the remaining machines, the starting time of the current job j is the
            # max of the following two times:
            # 1. The finish time of the previous job on the current machine
            # 2. The finish time of the current job on the previous machine
            for i in range(1, n_machines):
                machine_times[i].append(
                    max(
                        machine_times[i][j - 1] + self.records[i][x[j - 1]],  # 1
                        machine_times[i - 1][j] + self.records[i - 1][x[j]],  # 2
                    )
                )
        return machine_times


@default_random_state(seed=1)
def create_random_flowshop_problem(
    n_machines: int, n_jobs: int, random_state: Any = None, **kwargs: Any
) -> FlowshopScheduling:
    """Create a random flowshop scheduling problem.

    Args:
        n_machines: Number of machines.
        n_jobs: Number of jobs.
        random_state: Random state for reproducibility.
        **kwargs: Additional keyword arguments passed to FlowshopScheduling.

    Returns:
        A FlowshopScheduling problem instance.
    """
    T = random_state.random((n_machines, n_jobs)) * 50 + 50
    return FlowshopScheduling(T)


def visualize(
    problem: FlowshopScheduling,
    x: np.ndarray,
    path: str | None = None,
    label: bool = True,
) -> None:
    """Visualize the flowshop scheduling solution as a Gantt chart.

    Args:
        problem: The FlowshopScheduling problem instance.
        x: The job sequence solution.
        path: Optional path to save the figure.
        label: Whether to show job labels in the chart.
    """
    from pymoo.visualization.matplotlib import plt

    with plt.style.context("ggplot"):
        n_machines, n_jobs = problem.records.shape
        machine_times = problem.get_machine_times(x)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        Y = np.flip(np.arange(n_machines))

        for i in range(n_machines):
            for j in range(n_jobs):
                width = problem.records[i][x[j]]
                left = machine_times[i][j]
                ax.barh(
                    Y[i],
                    width,
                    left=left,
                    align="center",
                    color="gray",
                    edgecolor="black",
                    linewidth=0.8,
                )
                if label:
                    ax.text(
                        (left + width / 2),
                        Y[i],
                        str(x[j] + 1),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=15,
                    )
        ax.set_xlabel("Time")
        ax.set_yticks(np.arange(n_machines))
        ax.set_yticklabels(["M%d" % (i + 1) for i in Y])
        ax.set_title("Makespan: %s" % np.round(problem.makespan(x), 3))
        if path is not None:
            plt.savefig(path)
        plt.show()

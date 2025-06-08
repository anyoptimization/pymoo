import numpy as np


def find_non_dominated(F, epsilon=0.0):
    """
    Simple and efficient implementation to find only non-dominated points.
    Uses straightforward O(n²) algorithm with early termination.
    
    Parameters
    ----------
    F : np.ndarray
        Objective values matrix of shape (n_points, n_objectives)
    epsilon : float, optional
        Epsilon value for dominance comparison (default: 0.0)
        
    Returns
    -------
    np.ndarray
        Array of indices of non-dominated points
    """
    n_points = F.shape[0]
    non_dominated_indices = []
    
    if n_points == 0:
        return np.array([], dtype=int)
    
    # Check each point to see if it's non-dominated
    for i in range(n_points):
        is_dominated = False
        
        # Check if point i is dominated by any other point j
        for j in range(n_points):
            if i != j:
                # Check if j dominates i
                dominates = True
                at_least_one_better = False
                
                for k in range(F.shape[1]):  # for each objective
                    if F[j, k] + epsilon < F[i, k]:  # j is better than i in objective k
                        at_least_one_better = True
                    elif F[j, k] > F[i, k] + epsilon:  # j is worse than i in objective k
                        dominates = False
                        break  # Early termination in objective loop
                
                # j dominates i if j is at least as good in all objectives and better in at least one
                if dominates and at_least_one_better:
                    is_dominated = True
                    break  # Early termination - no need to check other points
        
        # If point i is not dominated by any other point, it's non-dominated
        if not is_dominated:
            non_dominated_indices.append(i)
    
    return np.array(non_dominated_indices, dtype=int)
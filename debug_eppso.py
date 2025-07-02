#!/usr/bin/env python3

"""
Debug script to identify non-deterministic behavior in EPPSO algorithm.
"""

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso_ep import EPPSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize


def test_eppso_determinism():
    """Test EPPSO for deterministic behavior with same seed."""
    
    # Simple test problem
    problem = get_problem("sphere")
    
    # Fixed parameters
    seed = 42
    n_gen = 10
    pop_size = 20
    
    results = []
    
    # Run algorithm multiple times with same seed
    for run in range(3):
        print(f"Run {run + 1}:")
        
        # Create algorithm with same seed
        algorithm = EPPSO(
            pop_size=pop_size,
            topology="star",  # Start with star topology
            seed=seed
        )
        
        # Run optimization
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', n_gen),
            seed=seed,
            verbose=False
        )
        
        # Store results
        results.append({
            'X': res.X.copy(),
            'F': res.F.copy(),
            'best_X': res.opt.get("X")[0].copy(),
            'best_F': res.opt.get("F")[0].copy()
        })
        
        print(f"  Best X: {results[-1]['best_X']}")
        print(f"  Best F: {results[-1]['best_F']}")
        print()
    
    # Compare results
    print("Comparing results:")
    for i in range(1, len(results)):
        diff_X = np.abs(results[0]['best_X'] - results[i]['best_X'])
        diff_F = np.abs(results[0]['best_F'] - results[i]['best_F'])
        
        print(f"Run 1 vs Run {i+1}:")
        print(f"  Max X difference: {np.max(diff_X):.2e}")
        print(f"  F difference: {diff_F[0]:.2e}")
        
        if np.max(diff_X) > 1e-10 or diff_F[0] > 1e-10:
            print(f"  NON-DETERMINISTIC BEHAVIOR DETECTED!")
        else:
            print(f"  Results are identical (within tolerance)")
        print()


def test_get_neighbors_determinism():
    """Test get_neighbors function for deterministic behavior."""
    from pymoo.algorithms.soo.nonconvex.pso_ep import get_neighbors
    
    print("Testing get_neighbors function:")
    
    # Test different topologies
    topologies = ["star", "ring", "random"]
    N = 10
    seed = 42
    
    for topology in topologies:
        print(f"\nTopology: {topology}")
        
        results = []
        for run in range(3):
            np.random.seed(seed)  # Reset numpy seed
            neighbors = get_neighbors(topology, N, random_state=np.random.RandomState(seed))
            results.append(neighbors)
            
        # Compare results
        if topology == "random":
            # For random topology, check if results are identical
            identical = True
            for i in range(1, len(results)):
                if not np.array_equal(results[0], results[i]):
                    identical = False
                    break
            
            if identical:
                print("  Results are identical (deterministic)")
            else:
                print("  Results differ (non-deterministic)")
                print(f"    Run 1: {results[0][:3]}")  # Show first 3 for brevity
                print(f"    Run 2: {results[1][:3]}")
        else:
            # Star and ring should always be identical
            print(f"  Results: {results[0][:3]}")  # Show first 3 for brevity


if __name__ == "__main__":
    print("EPPSO Non-determinism Debug Script")
    print("=" * 50)
    
    # Test get_neighbors function first
    test_get_neighbors_determinism()
    
    print("\n" + "=" * 50)
    
    # Test full algorithm
    test_eppso_determinism()
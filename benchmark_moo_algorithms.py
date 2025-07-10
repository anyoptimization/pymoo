#!/usr/bin/env python3
"""
Comprehensive benchmark of all Multi-Objective Optimization algorithms in pymoo
Testing against ZDT1 problem to compare performance with MOPSO
"""

import random
import numpy as np
import time
import traceback
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from pymoo.problems.multi import ZDT1
from pymoo.optimize import minimize
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV

# Import all MOO algorithms
from pymoo.algorithms.moo.mopso import MOPSO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def setup_algorithms():
    """Setup all algorithms with reasonable default parameters"""
    algorithms = {}
    
    # Import reference directions for algorithms that need them
    from pymoo.util.ref_dirs import get_reference_directions
    
    # For bi-objective problems, we can use uniform reference directions
    ref_dirs = get_reference_directions("uniform", 2, n_partitions=99)  # 100 reference directions
    
    # MOPSO with Crowding Distance
    try:
        algorithms['MOPSO'] = MOPSO(
            pop_size=100,
            w=0.4,
            c1=2.0,
            c2=2.0,
            archive_size=100,
        )
    except Exception as e:
        print(f"Failed to setup MOPSO: {e}")
    
    # NSGA-II
    try:
        algorithms['NSGA2'] = NSGA2(pop_size=100)
    except Exception as e:
        print(f"Failed to setup NSGA2: {e}")
    
    # NSGA-III
    try:
        algorithms['NSGA3'] = NSGA3(pop_size=100, ref_dirs=ref_dirs)
    except Exception as e:
        print(f"Failed to setup NSGA3: {e}")
    
    # SPEA2
    try:
        algorithms['SPEA2'] = SPEA2(pop_size=100)
    except Exception as e:
        print(f"Failed to setup SPEA2: {e}")
    
    # SMS-EMOA
    try:
        algorithms['SMS-EMOA'] = SMSEMOA(pop_size=100)
    except Exception as e:
        print(f"Failed to setup SMS-EMOA: {e}")
    
    # MOEA/D
    try:
        algorithms['MOEAD'] = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=20,
            prob_neighbor_mating=0.9
        )
    except Exception as e:
        print(f"Failed to setup MOEAD: {e}")
    
    # RVEA
    try:
        algorithms['RVEA'] = RVEA(pop_size=100, ref_dirs=ref_dirs)
    except Exception as e:
        print(f"Failed to setup RVEA: {e}")
    
    # AGE-MOEA
    try:
        algorithms['AGEMOEA'] = AGEMOEA(pop_size=100)
    except Exception as e:
        print(f"Failed to setup AGEMOEA: {e}")
    
    # AGE-MOEA-II
    try:
        algorithms['AGEMOEA2'] = AGEMOEA2(pop_size=100)
    except Exception as e:
        print(f"Failed to setup AGEMOEA2: {e}")
    
    # C-TAEA
    try:
        algorithms['CTAEA'] = CTAEA(pop_size=100, ref_dirs=ref_dirs)
    except Exception as e:
        print(f"Failed to setup CTAEA: {e}")
    
    # Try to add other variants if they exist
    try:
        from pymoo.algorithms.moo.rnsga2 import RNSGA2
        algorithms['RNSGA2'] = RNSGA2(pop_size=100)
    except Exception as e:
        pass  # Skip if not available or incompatible
    
    try:
        from pymoo.algorithms.moo.rnsga3 import RNSGA3
        algorithms['RNSGA3'] = RNSGA3(pop_size=100, ref_dirs=ref_dirs)
    except Exception as e:
        pass  # Skip if not available or incompatible
    
    try:
        from pymoo.algorithms.moo.dnsga2 import DNSGA2
        algorithms['DNSGA2'] = DNSGA2(pop_size=100)
    except Exception as e:
        pass  # Skip if not available or incompatible
    
    return algorithms


def run_benchmark():
    """Run benchmark comparing all algorithms"""
    print("=" * 80)
    print("MULTI-OBJECTIVE ALGORITHM BENCHMARK ON ZDT1")
    print("=" * 80)
    
    # Setup problem
    problem = ZDT1()
    pf = problem.pareto_front()
    print(f"Problem: {problem.__class__.__name__}")
    print(f"Variables: {problem.n_var}, Objectives: {problem.n_obj}")
    print(f"Pareto front points: {len(pf)}")
    
    # Setup performance indicators
    igd = IGD(pf)
    gd = GD(pf)
    hv = HV(ref_point=np.array([1.1, 1.1]))  # Reference point for hypervolume
    
    # Setup algorithms
    algorithms = setup_algorithms()
    print(f"\nTesting {len(algorithms)} algorithms:")
    for name in algorithms.keys():
        print(f"  - {name}")
    
    # Run benchmarks
    results = {}
    n_generations = 100
    
    print(f"\nRunning optimization for {n_generations} generations...")
    print("-" * 80)
    
    for name, algorithm in algorithms.items():
        print(f"\nTesting {name}...")
        
        try:
            # Time the optimization
            start_time = time.time()
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_generations),
                seed=42,  # Fixed seed for fair comparison
                verbose=False
            )
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Calculate performance metrics
            F = result.F
            n_solutions = len(F)
            
            # Calculate indicators
            igd_value = igd(F) if len(F) > 0 else float('inf')
            gd_value = gd(F) if len(F) > 0 else float('inf')
            
            try:
                hv_value = hv(F) if len(F) > 0 else 0.0
            except:
                hv_value = 0.0  # Some algorithms might have issues with HV calculation
            
            results[name] = {
                'n_solutions': n_solutions,
                'igd': igd_value,
                'gd': gd_value,
                'hv': hv_value,
                'runtime': runtime,
                'success': True,
                'F': F
            }
            
            print(f"  ✓ Success - {n_solutions} solutions found")
            print(f"    IGD: {igd_value:.6f}, GD: {gd_value:.6f}, HV: {hv_value:.6f}")
            print(f"    Runtime: {runtime:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results[name] = {
                'success': False,
                'error': str(e),
                'runtime': 0.0
            }
    
    return results, pf


def display_results(results, pf):
    """Display benchmark results in a nice format"""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    
    # Filter successful results
    successful = {name: data for name, data in results.items() if data.get('success', False)}
    failed = {name: data for name, data in results.items() if not data.get('success', False)}
    
    if successful:
        print(f"\n{'Algorithm':<12} {'Solutions':<10} {'IGD':<12} {'GD':<12} {'HV':<12} {'Runtime':<10}")
        print("-" * 80)
        
        # Sort by IGD (lower is better)
        sorted_results = sorted(successful.items(), key=lambda x: x[1]['igd'])
        
        for name, data in sorted_results:
            print(f"{name:<12} {data['n_solutions']:<10} "
                  f"{data['igd']:<12.6f} {data['gd']:<12.6f} "
                  f"{data['hv']:<12.6f} {data['runtime']:<10.2f}s")
        
        # Highlight best performers
        print("\n" + "=" * 50)
        print("BEST PERFORMERS:")
        print("=" * 50)
        
        best_igd = min(successful.values(), key=lambda x: x['igd'])
        best_gd = min(successful.values(), key=lambda x: x['gd'])
        best_hv = max(successful.values(), key=lambda x: x['hv'])
        best_time = min(successful.values(), key=lambda x: x['runtime'])
        
        for metric, best, direction in [
            ('IGD (lower better)', best_igd, 'igd'),
            ('GD (lower better)', best_gd, 'gd'), 
            ('HV (higher better)', best_hv, 'hv'),
            ('Runtime (faster)', best_time, 'runtime')
        ]:
            best_name = [name for name, data in successful.items() 
                        if data[direction] == best[direction]][0]
            print(f"Best {metric}: {best_name} ({best[direction]:.6f})")
        
        # MOPSO variants ranking
        mopso_variants = [name for name in successful.keys() if name.startswith('MOPSO')]
        if mopso_variants:
            print(f"\nMOPSO Variants Performance:")
            for variant in mopso_variants:
                if variant in successful:
                    variant_data = successful[variant]
                    igd_rank = sorted(successful.values(), key=lambda x: x['igd']).index(variant_data) + 1
                    gd_rank = sorted(successful.values(), key=lambda x: x['gd']).index(variant_data) + 1
                    hv_rank = sorted(successful.values(), key=lambda x: x['hv'], reverse=True).index(variant_data) + 1
                    
                    print(f"  {variant}:")
                    print(f"    IGD Rank: {igd_rank}/{len(successful)}")
                    print(f"    GD Rank: {gd_rank}/{len(successful)}")
                    print(f"    HV Rank: {hv_rank}/{len(successful)}")
    
    if failed:
        print(f"\n\nFAILED ALGORITHMS ({len(failed)}):")
        print("-" * 50)
        for name, data in failed.items():
            print(f"  {name}: {data.get('error', 'Unknown error')}")
    
    print(f"\nTotal algorithms tested: {len(results)}")
    print(f"Successful: {len(successful)}, Failed: {len(failed)}")


def generate_latex_table(results):
    """Generate LaTeX table from benchmark results"""
    # Filter successful results
    successful = {name: data for name, data in results.items() if data.get('success', False)}
    
    if not successful:
        return None
    
    # Sort by IGD (lower is better)
    sorted_results = sorted(successful.items(), key=lambda x: x[1]['igd'])
    
    latex_content = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}

\title{Multi-Objective Algorithm Benchmark Results on ZDT1}
\author{MOPSO Implementation Comparison}
\date{\today}

\begin{document}

\maketitle

\section{Introduction}
This document presents the benchmark results comparing various multi-objective optimization algorithms on the ZDT1 test problem. The benchmark evaluates performance using three key metrics: Inverted Generational Distance (IGD), Generational Distance (GD), and Hypervolume (HV).

\section{Performance Metrics}
\begin{itemize}
\item \textbf{IGD (Inverted Generational Distance)}: Lower values indicate better convergence and diversity. Measures the average distance from the true Pareto front to the obtained solutions.
\item \textbf{GD (Generational Distance)}: Lower values indicate better convergence. Measures the average distance from obtained solutions to the true Pareto front.
\item \textbf{HV (Hypervolume)}: Higher values indicate better performance. Measures the volume of objective space dominated by the solution set.
\item \textbf{Runtime}: Execution time in seconds for 100 generations.
\item \textbf{Solutions}: Number of non-dominated solutions found.
\end{itemize}

\section{Benchmark Results}

\begin{table}[h!]
\centering
\caption{Multi-Objective Algorithm Performance Comparison on ZDT1}
\label{tab:moo_benchmark}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Algorithm} & \textbf{Solutions} & \textbf{IGD} & \textbf{GD} & \textbf{HV} & \textbf{Runtime (s)} \\
\midrule
"""
    
    # Add table rows
    for i, (name, data) in enumerate(sorted_results):
        # Format the row
        latex_content += f"{name} & {data['n_solutions']} & {data['igd']:.6f} & {data['gd']:.6f} & {data['hv']:.6f} & {data['runtime']:.2f} \\\\\n"
    
    latex_content += r"""
\bottomrule
\end{tabular}
\end{table}

\section{Key Findings}

"""
    
    # Add key findings
    best_igd = min(successful.values(), key=lambda x: x['igd'])
    best_gd = min(successful.values(), key=lambda x: x['gd'])
    best_hv = max(successful.values(), key=lambda x: x['hv'])
    best_time = min(successful.values(), key=lambda x: x['runtime'])
    
    best_igd_name = [name for name, data in successful.items() if data['igd'] == best_igd['igd']][0]
    best_gd_name = [name for name, data in successful.items() if data['gd'] == best_gd['gd']][0]
    best_hv_name = [name for name, data in successful.items() if data['hv'] == best_hv['hv']][0]
    best_time_name = [name for name, data in successful.items() if data['runtime'] == best_time['runtime']][0]
    
    latex_content += f"""
\\begin{{itemize}}
\\item \\textbf{{Best IGD (Convergence + Diversity):}} {best_igd_name} ({best_igd['igd']:.6f})
\\item \\textbf{{Best GD (Convergence):}} {best_gd_name} ({best_gd['gd']:.6f})
\\item \\textbf{{Best HV (Overall Performance):}} {best_hv_name} ({best_hv['hv']:.6f})
\\item \\textbf{{Fastest Runtime:}} {best_time_name} ({best_time['runtime']:.2f}s)
\\end{{itemize}}

"""
    
    # Add MOPSO analysis
    mopso_variants = [name for name in successful.keys() if name.startswith('MOPSO')]
    if mopso_variants:
        latex_content += r"""
\section{MOPSO Variant Analysis}

The MOPSO implementation was tested with three different leader selection strategies:

\begin{itemize}
\item \textbf{MOPSO}: Crowding distance selection
\end{itemize}

"""
        
        # Performance comparison of MOPSO variants
        mopso_data = {name: successful[name] for name in mopso_variants}
        best_mopso_igd = min(mopso_data.values(), key=lambda x: x['igd'])
        best_mopso_gd = min(mopso_data.values(), key=lambda x: x['gd'])
        best_mopso_hv = max(mopso_data.values(), key=lambda x: x['hv'])
        
        best_mopso_igd_name = [name for name, data in mopso_data.items() if data['igd'] == best_mopso_igd['igd']][0]
        best_mopso_gd_name = [name for name, data in mopso_data.items() if data['gd'] == best_mopso_gd['gd']][0]
        best_mopso_hv_name = [name for name, data in mopso_data.items() if data['hv'] == best_mopso_hv['hv']][0]
        
        latex_content += f"""
Among the MOPSO variants:
\\begin{{itemize}}
\\item \\textbf{{Best IGD:}} {best_mopso_igd_name} ({best_mopso_igd['igd']:.6f})
\\item \\textbf{{Best GD:}} {best_mopso_gd_name} ({best_mopso_gd['gd']:.6f})
\\item \\textbf{{Best HV:}} {best_mopso_hv_name} ({best_mopso_hv['hv']:.6f})
\\end{{itemize}}

"""
    
    latex_content += r"""
\section{Conclusion}

This benchmark demonstrates the comparative performance of various multi-objective optimization algorithms on the ZDT1 problem. The results provide insights into the strengths and weaknesses of different algorithmic approaches, particularly highlighting the performance characteristics of the MOPSO implementation with different leader selection strategies.

\end{document}
"""
    
    return latex_content


def main():
    """Main benchmark function"""
    try:
        results, pf = run_benchmark()
        display_results(results, pf)
        
        # Generate LaTeX table
        print("\nGenerating LaTeX document...")
        latex_content = generate_latex_table(results)
        
        if latex_content:
            with open('moo_benchmark_results.tex', 'w', encoding='utf-8') as f:
                f.write(latex_content)
            print("✓ LaTeX document saved as 'moo_benchmark_results.tex'")
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 
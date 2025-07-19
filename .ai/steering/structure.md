# Project Structure

## Root Directory Layout
```
pymoo/                    # Main package directory
├── algorithms/           # Optimization algorithms (MOO, SOO)
├── core/                # Core framework classes and interfaces
├── problems/            # Test problems and problem definitions
├── operators/           # Genetic operators (crossover, mutation, selection)
├── indicators/          # Performance metrics and indicators
├── visualization/       # Plotting and visualization tools
├── gradient/            # Gradient computation utilities
├── constraints/         # Constraint handling mechanisms
├── termination/         # Termination criteria
├── util/                # Utility functions and helpers
└── functions/           # Compiled performance-critical functions

examples/                 # Usage examples and tutorials
tests/                   # Test suite
docs/                    # Documentation source
benchmark/               # Performance benchmarking scripts
```

## Core Architecture Patterns

### Problem Definition
- Base class: `pymoo.core.problem.Problem`
- Elementwise problems: `ElementwiseProblem` for single-point evaluation
- Meta problems: `MetaProblem` for problem composition
- All problems define `n_var`, `n_obj`, `n_constr`, `xl`, `xu` attributes

### Algorithm Structure
- Base class: `pymoo.core.algorithm.Algorithm`
- Algorithms follow ask-and-tell pattern
- Modular design with interchangeable operators
- Support for both population-based and single-point methods

### Operator Pattern
- All operators inherit from `pymoo.core.operator.Operator`
- Crossover, mutation, selection, sampling operators
- Configurable and composable design

### Data Structures
- `Population`: Collection of individuals
- `Individual`: Single solution with variables, objectives, constraints
- `Result`: Optimization result container

## File Organization Conventions

### Algorithm Files
- Multi-objective: `pymoo/algorithms/moo/`
- Single-objective: `pymoo/algorithms/soo/`
- Each algorithm in separate module with descriptive name

### Problem Files
- Multi-objective: `pymoo/problems/multi/`
- Single-objective: `pymoo/problems/single/`
- Many-objective: `pymoo/problems/many/`
- Dynamic problems: `pymoo/problems/dynamic/`

### Example Structure
- Organized by category in `examples/` subdirectories
- Each example is self-contained and executable
- Integration tests run examples as validation

### Test Organization
- Mirror package structure in `tests/`
- Separate test categories using pytest markers
- Performance tests in `benchmark/` directory

## Import Conventions
- Main API accessible via `from pymoo.optimize import minimize`
- Algorithm imports: `from pymoo.algorithms.moo.nsga2 import NSGA2`
- Problem imports: `from pymoo.problems import get_problem`
- Utility imports follow full module path
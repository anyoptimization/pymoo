# Technology Stack

## Core Technologies
- **Python**: 3.9+ (supports 3.9, 3.10, 3.11, 3.12, 3.13)
- **Build System**: setuptools with Cython compilation
- **Package Management**: pip/PyPI distribution

## Key Dependencies
- **numpy** (>=1.19.3): Core numerical operations
- **scipy** (>=1.1): Scientific computing algorithms
- **matplotlib** (>=3): Visualization and plotting
- **autograd** (>=1.4): Automatic differentiation
- **cma** (>=3.2.2): Covariance Matrix Adaptation
- **moocore** (>=0.1.7): Multi-objective optimization core utilities
- **Cython**: Performance-critical compiled extensions

## Optional Dependencies
- **Parallelization**: joblib, dask, ray
- **Development**: pytest, jupyter, pandas, numba
- **Optimization**: optuna

## Build Commands

### Installation
```bash
# Standard installation
pip install -U pymoo

# Development installation
git clone https://github.com/anyoptimization/pymoo
cd pymoo
pip install .

# With optional dependencies
pip install pymoo[full]  # All features
pip install pymoo[parallelization]  # Parallel computing
pip install pymoo[visualization]  # Enhanced plotting
```

### Development Commands
```bash
# Clean build artifacts
make clean

# Clean compiled extensions
make clean-ext

# Compile Cython extensions
make compile
python setup.py build_ext --inplace

# Create distribution
make dist

# Install from source
make install
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not long"  # Skip long-running tests
pytest -m examples    # Run example integration tests
pytest -m gradient    # Run gradient computation tests
```

### Verification
```bash
# Check if compiled extensions are working
python -c "from pymoo.functions import is_compiled;print('Compiled Extensions: ', is_compiled())"
```
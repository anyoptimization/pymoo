# pymoo Development Guide

This file contains essential knowledge for working with pymoo, including CLI tools, development workflows, and best practices.

## CLI Tools Overview

pymoo provides a unified CLI toolset in the `tools/` directory for managing development workflows. All tools are accessed through the `./tools/run` wrapper which handles conda environment setup and Python path configuration.

### Core Tools

#### **Python Execution - `./tools/run`**
**CRITICAL**: This is the ONLY way to execute Python code in pymoo. Never use `python` directly.

```bash
# Execute Python scripts with proper environment setup
./tools/run script.py
./tools/run path/to/script.py
```

Features:
- Automatically activates the correct conda environment (`default` by default)
- Sets up pymoo in the Python path
- Ensures consistent environment across all operations
- Configurable via `tools/.env` file

#### **Documentation Management - `./tools/run docs`**
Comprehensive documentation compilation and building system.

```bash
# Clean all generated notebooks
./tools/run docs clean

# Compile missing .md files to .ipynb and execute them
./tools/run docs compile

# Force compile all files (regenerate existing notebooks)
./tools/run docs compile --force

# Compile specific files or patterns
./tools/run docs compile file1.md file2.md
./tools/run docs compile "algorithms/*.md"

# Convert without executing (syntax check)
./tools/run docs compile --no-execute

# Build HTML documentation
./tools/run docs build
```

Features:
- Converts Markdown to Jupyter notebooks using jupytext
- Executes notebooks to generate outputs
- Supports selective compilation (only missing files)
- Detailed error reporting and logging
- Uses `default` environment for compilation, `pymoo-doc` for building

#### **Examples Testing - `./tools/examples`**
Pytest-based runner for all example files in the `examples/` directory.

```bash
# Run all examples (excludes long-running tests by default)
./tools/examples

# Include long-running examples
./tools/examples --all

# Run only previously failed examples (using pytest cache)
./tools/examples --lf

# Run failed examples first, then successful ones
./tools/examples --ff

# Stop on first failure
./tools/examples -x

# Run specific examples by keyword
./tools/examples -k "nsga2"
./tools/examples -k "algorithms/moo"

# Verbose output
./tools/examples -v
```

Features:
- Treats each .py file in examples/ as a test case
- Intelligent caching with pytest
- Excludes long-running tests by default
- Parallel execution support
- Comprehensive failure reporting

#### **Unit Testing - `./tools/tests`**
Pytest-based runner for the test suite in the `tests/` directory.

```bash
# Run all tests (excludes long-running tests by default)
./tools/tests

# Include long-running tests
./tools/tests --all

# Run only previously failed tests
./tools/tests --lf

# Run specific test files or directories
./tools/tests tests/algorithms/
./tools/tests tests/algorithms/test_nsga2.py

# Run tests matching a pattern
./tools/tests -k "nsga2"

# Stop on first failure
./tools/tests -x

# Verbose output
./tools/tests -v
```

Features:
- Standard pytest runner with pymoo-specific configurations
- Excludes long-running tests by default (marked with `@pytest.mark.long`)
- Intelligent test discovery and caching
- Supports all pytest options and markers


## Development Workflows

### Systematic Debugging Guides
pymoo includes comprehensive debugging guides in `.claude/commands/`:

- **`fix-examples.md`**: Systematic workflow for fixing failing examples
- **`fix-tests.md`**: Systematic workflow for fixing failing unit tests  
- **`fix-docs.md`**: Systematic workflow for fixing documentation compilation

These guides provide step-by-step processes for:
1. Identifying failures
2. Iteratively fixing issues
3. Using intelligent caching to avoid re-running successful tests
4. Final verification

### Key File Structure

```
tools/
├── .env                     # Environment configuration
├── run                      # Main execution wrapper (handles all Python execution)
├── docs                     # Documentation management tool
├── examples                 # Examples testing tool
└── tests                    # Unit testing tool

.claude/commands/
├── fix-examples.md        # Examples debugging guide
├── fix-tests.md           # Tests debugging guide
└── fix-docs.md            # Documentation debugging guide

```

## Documentation System

pymoo uses a hybrid documentation system:
- **Markdown (.md) files**: Source files for documentation content
- **Jupyter Notebooks (.ipynb)**: Generated from .md files for interactive execution. NEVER modify them directly.
- **Sphinx**: Documentation generator that processes both formats
- **Jupytext**: Tool that syncs between .md and .ipynb formats

## Development Best Practices

### Python Execution
- **ALWAYS** use `./tools/run` for any Python execution
- **NEVER** use `python` directly - it won't have proper environment setup
- Use `./tools/run script.py` for running scripts
- Use `./tools/run -m module` for running modules

### Testing Strategy
- Use `./tools/examples` for validating example code
- Use `./tools/tests` for unit testing
- Leverage `--lf` flag for efficient iteration on failed tests
- Exclude long-running tests during development (`--all` for complete testing)

### Documentation Workflow
- Start with `./tools/run docs clean` for fresh compilation
- Use `./tools/run docs compile` iteratively until all files process
- Check `docs/compile.log` for detailed error information
- Use `./tools/run docs build` for final HTML generation

### Code Conventions
- Use numpy docstring format throughout the codebase
- Follow existing patterns and conventions in the codebase
- Maintain consistency with pymoo's API design principles


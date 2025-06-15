# Fix Tests Command

A systematic workflow for debugging and fixing failing PyMoo unit tests.

## Purpose

This command provides a structured approach to fix failing tests by:
1. Running all unit tests to identify failures
2. Iteratively fixing only the failed tests
3. Using pytest's cache to save time by re-running only failed tests
4. Final verification that all tests pass

## Workflow

### Step 1: Initial Assessment
Run all tests (excluding long-running ones) to see the current state and identify failing tests:

```bash
./tools/tests
```

This will show you:
- Total number of tests run
- Which specific tests are failing
- Error messages for each failure
- Test duration and performance summary

### Step 2: Note the Failures
Pay attention to the output and note:
- Which test files/functions are failing
- The specific error messages and stack traces
- Any patterns in the failures (e.g., import errors, API changes, etc.)
- Whether failures are in algorithms, operators, problems, or utilities

### Step 3: Fix the Issues
For each failing test:
1. Open the failing test file (located in `tests/` directory)
2. Analyze the error message and stack trace
3. **Prioritize test fixes by category**:
   - **Unit test bugs**: Fix test logic that's incorrect
   - **API changes**: Update tests to match current PyMoo API
   - **Import errors**: Fix import paths or missing dependencies
   - **Tolerance issues**: Adjust numerical tolerances for floating-point comparisons
   - **Core functionality bugs**: Fix genuine bugs in PyMoo core code
4. Common test fixes include:
   - Import errors (update import paths)
   - Deprecated function calls (use newer API)
   - Parameter name changes (update to current API)
   - Tolerance adjustments for numerical tests
   - Test data file path issues
   - Missing test dependencies

### Step 4: Test Only Failed Tests
After making fixes, run only the previously failed tests to save time:

```bash
./tools/tests --lf
```

This uses pytest's cache to run only tests that failed in the last run.

### Step 5: Iterate
Repeat steps 3-4 until all previously failed tests pass:
- Fix remaining issues
- Run `./tools/tests --lf` again
- Continue until no failures remain

### Step 6: Final Verification
Once all previously failed tests pass, run the complete test suite to ensure:
- All tests still pass
- No regressions were introduced
- The fixes didn't break other tests

```bash
./tools/tests
```

## Useful Commands

- `./tools/tests` - Run all tests (excluding long-running ones)
- `./tools/tests --all` - Run all tests including long-running ones
- `./tools/tests --lf` - Run only previously failed tests
- `./tools/tests --ff` - Run failed tests first, then successful ones
- `./tools/tests -x` - Stop on first failure (useful for debugging one issue at a time)
- `./tools/tests -k "keyword"` - Run only tests matching a keyword
- `./tools/tests -k "specific_test_name"` - Run only a specific test by name
- `./tools/tests -v` - Extra verbose output for debugging
- `./tools/tests -m "marker"` - Run tests with specific markers

### Running Specific Tests

To run only specific tests (useful when fixing individual test failures):

```bash
# Run only tests with "nsga2" in the name
./tools/tests -k "nsga2"

# Run only algorithm tests
./tools/tests tests/algorithms/

# Run only a specific test file
./tools/tests tests/algorithms/test_nsga2.py

# Run only tests from the operators directory
./tools/tests tests/operators/

# Run only gradient tests
./tools/tests tests/gradients/

# Run multiple test categories
./tools/tests -k "nsga2 or de or ga"
```

The `-k` flag uses pytest's keyword matching, so you can use:
- Exact test names: `-k "test_nsga2_binary"`
- Partial names: `-k "nsga"`
- Directory names: `-k "algorithms"` or `-k "operators"`
- Class names: `-k "TestNSGA2"`
- Multiple keywords: `-k "nsga2 or algorithm"`

### Test Categories

PyMoo tests are organized into these main categories:

- **algorithms/**: Algorithm implementation tests
  - `test_algorithms.py` - General algorithm tests
  - `test_nsga2.py`, `test_rvea.py`, etc. - Specific algorithm tests
  - `test_single_objective.py` - Single-objective algorithm tests
- **operators/**: Genetic operator tests
  - `test_crossover.py` - Crossover operator tests
  - `test_mutation.py` - Mutation operator tests
- **problems/**: Test problem tests
  - `test_correctness.py` - Problem correctness validation
  - `test_problems_*.py` - Specific problem suite tests
- **indicators/**: Performance indicator tests
- **gradients/**: Gradient computation tests
- **misc/**: Utility and miscellaneous tests

## Tips

1. **Fix tests, then code**: Try to fix the test first. Only modify PyMoo core code if there's a genuine bug
2. **Start with import errors**: These are often the easiest to fix and may resolve multiple test failures
3. **Use -x flag**: When debugging, use `./tools/tests --lf -x` to stop on the first failure and focus on one issue at a time
4. **Test specific categories**: Use directory paths to run tests by category (algorithms, operators, etc.)
5. **Check for API changes**: Many test failures are due to API changes in PyMoo
6. **Look for patterns**: If many tests fail similarly, there might be a common underlying issue
7. **Test incrementally**: Use `--lf` to avoid re-running long tests that already pass
8. **Use verbose output**: Add `-v` for detailed test output when debugging
9. **Check numerical tolerances**: Floating-point comparison tests may need tolerance adjustments
10. **Review test markers**: Some tests are marked as "long" - exclude them for faster iteration

## Common Issues and Solutions

- **Import errors**: Check if modules have been moved or renamed in PyMoo
- **API changes**: Update test calls to match current PyMoo API
- **Numerical precision**: Adjust tolerances in `assert_allclose` or similar assertions
- **Missing test data**: Ensure test data files exist and paths are correct
- **Deprecated warnings**: Update tests to use newer API calls
- **Parameter changes**: Check if function signatures have changed
- **Environment issues**: Verify all required test packages are installed

## Test-Specific Considerations

### Algorithm Tests
- May fail due to random seed changes or numerical precision
- Check for parameter name changes in algorithm constructors
- Verify termination criteria and convergence tolerances

### Operator Tests
- Often fail due to API changes in operator interfaces
- Check input/output shapes and types
- Verify parameter passing to operators

### Problem Tests
- May fail due to changes in problem definitions
- Check objective function implementations
- Verify constraint handling

### Gradient Tests
- Sensitive to numerical precision
- May require adjustment of finite difference tolerances
- Check automatic differentiation backends

## Success Criteria

The fix-tests process is complete when:
1. `./tools/tests` shows 0 failures
2. All unit tests run without errors
3. No critical warnings (deprecation warnings are acceptable if noted)
4. The output shows "All tests passed!"
5. Core PyMoo functionality is verified through test coverage

## Notes

- Long-running tests are excluded by default. Use `--all` to include them if needed
- Some tests may be environment-specific (e.g., requiring specific dependencies)
- Focus on non-long-running tests for faster iteration during development
- Document any tests that need to be skipped due to environment constraints
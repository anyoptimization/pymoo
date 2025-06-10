# Fix Examples Command

A systematic workflow for debugging and fixing failing PyMoo examples.

## Purpose

This command provides a structured approach to fix failing examples by:
1. Running all examples to identify failures
2. Iteratively fixing only the failed examples
3. Using pytest's cache to save time by re-running only failed tests
4. Final verification that all examples pass

## Workflow

### Step 1: Initial Assessment
Run all examples to see the current state and identify failing tests:

```bash
./tools/examples
```

This will show you:
- Total number of examples
- Which specific examples are failing
- Error messages for each failure

### Step 2: Note the Failures
Pay attention to the output and note:
- Which example files are failing
- The specific error messages
- Any patterns in the failures (e.g., missing dependencies, import errors, etc.)

### Step 3: Fix the Issues
For each failing example:
1. Open the failing example file (located in `examples/` directory)
2. Analyze the error message
3. **Always try to fix the example first** - Examples should be updated to work with the current codebase
4. Only modify the main codebase if:
   - The example reveals a genuine bug in PyMoo
   - The API change breaks backward compatibility unintentionally
   - The issue affects core functionality
5. Common example fixes include:
   - Import errors (update import paths)
   - Deprecated function calls (use newer API)
   - Missing dependencies (add required imports)
   - Incorrect parameter usage (update to current API)
   - File path issues (fix relative paths)

### Step 4: Test Only Failed Examples
After making fixes, run only the previously failed tests to save time:

```bash
./tools/examples --lf
```

This uses pytest's cache to run only tests that failed in the last run.

### Step 5: Iterate
Repeat steps 3-4 until all previously failed tests pass:
- Fix remaining issues
- Run `./tools/examples --lf` again
- Continue until no failures remain

### Step 6: Final Verification
Once all previously failed tests pass, run the complete test suite to ensure:
- All examples still pass
- No regressions were introduced
- The fixes didn't break other examples

```bash
./tools/examples
```

## Useful Commands

- `./tools/examples` - Run all examples
- `./tools/examples --lf` - Run only previously failed tests
- `./tools/examples --ff` - Run failed tests first, then successful ones
- `./tools/examples -x` - Stop on first failure (useful for debugging one issue at a time)
- `./tools/examples -k "keyword"` - Run only examples matching a keyword
- `./tools/examples -k "specific_example_name"` - Run only a specific example by name
- `./tools/examples -v` - Extra verbose output for debugging

### Running a Specific Example

To run only a specific example (useful when fixing individual examples):

```bash
# Run only examples with "nsga2" in the name
./tools/examples -k "nsga2"

# Run only the basic GA example
./tools/examples -k "ga.py"

# Run only examples from the algorithms/moo directory
./tools/examples -k "moo"
```

The `-k` flag uses pytest's keyword matching, so you can use:
- Exact filenames: `-k "nsga2.py"`
- Partial names: `-k "nsga"`
- Directory names: `-k "soo"` or `-k "moo"`
- Multiple keywords: `-k "nsga2 or de"`

## Tips

1. **Fix examples, not code**: Always try to fix the example first. Only modify the main codebase if there's a genuine bug or API issue
2. **Start with the easiest fixes**: Often multiple examples fail for the same reason (e.g., an import change)
3. **Use -x flag**: When debugging, use `./tools/examples --lf -x` to stop on the first failure and focus on one issue at a time
4. **Test specific examples**: Use `-k "example_name"` to run just the example you're working on
5. **Check dependencies**: Many example failures are due to missing or changed dependencies
6. **Look for patterns**: If many examples fail similarly, there might be a common underlying issue
7. **Test incrementally**: Use `--lf` to avoid re-running long-running tests that already pass
8. **Keep notes**: Track which examples you've fixed and what the issues were

## Common Issues and Solutions

- **Import errors**: Check if modules have been moved or renamed
- **Deprecated warnings**: Update to use newer API calls
- **Missing files**: Ensure example data files exist and paths are correct
- **Parameter changes**: Check if function signatures have changed
- **Environment issues**: Verify all required packages are installed

## Success Criteria

The fix-examples process is complete when:
1. `./tools/examples` shows 0 failures
2. All examples run without errors
3. No deprecation warnings (ideally)
4. The output shows "All examples tests passed!"
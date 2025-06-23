# Fix Tests Command

Find and fix failing tests in the **#$ARGUMENTS** test suite. Use systematic stop-early strategy to efficiently identify and resolve all failures.

**Strategy: Use efficient iteration with stop-early to fix failures, then run complete suite for final verification to ensure no regressions.**

## Arguments

Fix the test suite called **#$ARGUMENTS**. The tool for this test suite is located at:
```bash
./tools/testing #$ARGUMENTS
```

**⏰ Timeout Note for Claude**: When running any test commands with the Bash tool, always use `timeout: 600000` (10 minutes) to prevent long-running operations from blocking progress.

## Workflow

### Step 1: Find First Failure

Start by finding the first failing test without running everything:

```bash
./tools/testing #$ARGUMENTS -x
```

**⚠️ Note: This stops at the first failure, saving significant time!** No need to wait for all tests to run initially.

**⏰ Important for Claude**: When using the Bash tool to run any command, set `timeout: 600000` (10 minutes).

**🔍 Check Results**: If all tests pass (exit code 0), **STOP HERE** - no fixes needed! Only proceed with the remaining steps if at least one test is failing.

### Step 2: Analyze and Fix the First Issue

For the failing test:

1. **Note the exact failing item** from the output (test name, file path, or specific test case)

2. **Open the failing file** - examine the test output to identify the specific file and location of the failure

3. **THINK HARD before implementing any fix**:
   - Understand the root cause of the failure
   - Consider the broader implications of your fix
   - Evaluate whether this affects other tests in the suite
   - Determine if this is a symptom of a larger issue
   - **Check the historical context using Git/GitHub**:
     - `git log --oneline path/to/failing_file` - See recent changes
     - `git blame path/to/failing_file` - See who changed what and when
     - `gh search issues "failing_item_name"` - Search for related GitHub issues
     - `gh search prs "failing_item_name"` - Search for related pull requests
     - Check if recent commits to core files might have broken this
     - Look for patterns: multiple items failing due to the same underlying change

4. **Analyze the error message thoroughly** - understand exactly what's failing and why

5. **Apply systematic fixing strategies**:
   - Fix test logic that's incorrect or outdated
   - Update code to match current API expectations
   - Fix import paths or missing dependencies
   - Adjust numerical tolerances for floating-point comparisons
   - Fix file path issues and missing resources
   - Update deprecated function calls and parameter usage
   - Fix genuine bugs in the underlying code if the test reveals a real issue

### Step 3: Verify Your Fix

Test only the specific item you just fixed:

```bash
./tools/testing #$ARGUMENTS -k "specific_test_name"
```

Use the exact test name, file name, or pattern from the failure output to target just the item you fixed.

**⏰ Important for Claude**: Use `timeout: 600000` (10 minutes) in the Bash tool call.

### Step 4: Continue from Where You Left Off

Once your fix is verified, continue finding the next failure:

```bash
./tools/testing #$ARGUMENTS --lf -x
```

**⏰ Important for Claude**: Use `timeout: 600000` (10 minutes) in the Bash tool call.

This will:
- Skip the item you just fixed (since it now passes)
- Continue from where the previous run left off  
- Stop at the next failure

### Step 5: Repeat the Fix-and-Continue Cycle

Repeat steps 2-4 for each failure:
1. Fix the failing item
2. Test the specific fix (Step 3 commands with 10-minute timeout)
3. Continue to next failure (Step 4 commands with 10-minute timeout)
4. Repeat until no more failures are found

### Step 6: Final Comprehensive Verification

Once no more failures are found with `-x`, run the complete test suite to ensure:
- All tests pass
- No regressions were introduced by your fixes
- Fixing one test didn't break another test

```bash
./tools/testing #$ARGUMENTS
```

**This final run is CRITICAL** - it verifies that your fixes didn't introduce new failures elsewhere in the codebase. Even though you've been fixing tests iteratively, changes to shared code or dependencies could have broken other tests.

**⏰ Important for Claude**: Use `timeout: 600000` (10 minutes) in the Bash tool call for final verification too.

## Test Suite Commands Reference

### Core Commands
```bash
./tools/testing #$ARGUMENTS                     # Run all tests in the suite
./tools/testing #$ARGUMENTS --lf                # Run only previously failed tests  
./tools/testing #$ARGUMENTS --ff                # Run failed tests first, then successful
./tools/testing #$ARGUMENTS -x                  # Stop on first failure
./tools/testing #$ARGUMENTS -k "keyword"        # Run tests matching keyword
./tools/testing #$ARGUMENTS -v                  # Verbose output
./tools/testing #$ARGUMENTS --all               # Include long-running tests (if applicable)
```

### Selective Execution
```bash
./tools/testing #$ARGUMENTS path/to/directory/   # Run tests in specific directory
./tools/testing #$ARGUMENTS path/to/file.py      # Run specific test file
./tools/testing #$ARGUMENTS -k "exact_name"      # Run tests with exact name match
./tools/testing #$ARGUMENTS -k "partial"         # Run tests with partial name match
```

## Advanced Usage Patterns

### Pattern Matching with `-k`

The `-k` flag uses powerful keyword matching to target specific tests:

- **Exact names**: `-k "test_specific_function"`
- **Partial names**: `-k "keyword"`  
- **Directory/module names**: `-k "algorithms"` or `-k "operators"`
- **Class names**: `-k "TestClassName"`
- **Multiple keywords**: `-k "keyword1 or keyword2"`
- **Complex patterns**: `-k "keyword1 and not keyword2"`

### Selective Testing by Category

```bash
./tools/testing #$ARGUMENTS path/to/category/        # Tests in specific category
./tools/testing #$ARGUMENTS -k "category"            # Tests matching category keyword
./tools/testing #$ARGUMENTS -k "slow"                # Tests marked as slow
./tools/testing #$ARGUMENTS -k "integration"         # Integration tests
./tools/testing #$ARGUMENTS -k "unit"                # Unit tests
```

## Common Issues and Solutions

### Import Errors
- **Issue**: `ImportError` or `ModuleNotFoundError`
- **Solution**: Update import statements to match current module structure
- **Example**: Fix import paths when modules have been moved or renamed

### API Changes
- **Issue**: `AttributeError` or `TypeError` when calling functions
- **Solution**: Check current API and update function calls
- **Example**: Update deprecated parameter names or function signatures

### Numerical Precision Issues
- **Issue**: Floating-point comparison failures in tests
- **Solution**: Adjust tolerances in `assert_allclose` or similar assertions
- **Example**: Increase `atol` or `rtol` parameters for numerical comparisons

### Missing Dependencies
- **Issue**: Code execution fails due to missing packages or modules
- **Solution**: Add required imports or ensure dependencies are available
- **Example**: Add missing import statements or check package availability

### File Path Issues
- **Issue**: `FileNotFoundError` when loading test data or resource files
- **Solution**: Fix relative paths or ensure resource files exist
- **Example**: Update paths to test data files or ensure test fixtures are available

### Environment Issues
- **Issue**: Tests fail due to environment-specific problems
- **Solution**: Check environment setup and dependencies
- **Example**: Ensure proper Python path, virtual environment, or system dependencies

## Git and GitHub Investigation Strategies

### Git History Analysis
```bash
# See recent changes to core codebase
git log --since="1 month ago" --oneline src/

# See recent changes to specific file
git log --oneline path/to/failing_file

# See detailed changes in a specific commit  
git show <commit-hash>

# Compare recent changes in a directory
git diff HEAD~5..HEAD src/main_module/

# See who changed what and when
git blame path/to/failing_file
```

### GitHub Search Techniques
```bash
# Search for related issues using specific error messages
gh search issues "ImportError specific_module"
gh search issues "test failure error_message"
gh search issues "failing_test_name"

# Search for related pull requests
gh search prs "fix failing_test"
gh search prs "update test_suite"
gh search prs "path/to/failing_file"

# Search by file paths or test patterns
gh search prs "tests/specific_area/"
gh search issues "test_pattern"

# Look for API migration guides
gh search issues "API breaking change"
gh search prs "deprecate" 
```

## Success Criteria

The fix-tests process is complete when:

1. `./tools/testing #$ARGUMENTS` shows 0 failures
2. All tests in the suite run without errors
3. Core functionality is verified through test coverage
4. No regressions were introduced by the fixes

## Tips for Efficient Debugging

1. **Fix systematically**: Fix test logic first, then underlying code if there's a genuine bug

2. **Use efficient iteration with comprehensive verification**: This workflow uses stop-early iteration to fix issues efficiently, then runs complete suite to catch any regressions

3. **Always verify individual fixes**: Test each fix with `-k "item_name"` before moving on

4. **Leverage test caching**: `--lf` and `--ff` flags are incredibly powerful for iteration

5. **Look for patterns**: If similar tests fail, apply the same fix to multiple files

6. **Start with import errors**: These are often easiest to fix and may resolve multiple failures

7. **Check for recent changes**: Use Git history to understand what might have broken

8. **Use exact names**: Copy exact test names from failure output for `-k` flag

9. **Document your progress**: Keep notes on what you've fixed to identify patterns

10. **Consider environment factors**: Some failures may be environment-specific

This approach provides consistent, efficient debugging for any test suite using a single, parameterized testing tool.
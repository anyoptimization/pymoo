# Fix Docs Command

A systematic workflow for debugging and fixing failing PyMoo documentation compilation.

## Purpose

This command provides a structured approach to fix failing documentation by:
1. Cleaning all generated notebook files to start fresh
2. Iteratively running compilation until all .md files are successfully converted to .ipynb
3. Fixing compilation errors as they arise
4. Final verification that all documentation compiles successfully

## Workflow

### Step 1: Clean All Generated Files
Start with a clean slate by removing all generated .ipynb files:

```bash
./tools/run docs clean
```

This will:
- Remove all .ipynb files from docs/source/
- Remove .ipynb_checkpoints directories  
- Remove build artifacts
- Remove Python cache files

### Step 2: Initial Compilation Attempt
Run the first compilation to identify failures:

```bash
./tools/run docs compile
```

This will show you:
- Total number of .md files found
- How many files need to be processed (should be all files after cleaning)
- Which specific files are failing to compile or execute
- Error messages for each failure

### Step 3: Note the Failures
Pay attention to the output and note:
- Which documentation files are failing
- The specific error messages and stack traces
- Any patterns in the failures (e.g., import errors, missing dependencies, API changes)
- Whether failures are in conversion (.md → .ipynb) or execution phases

### Step 4: Fix the Issues
For each failing documentation file:
1. Open the failing .md file (located in `docs/source/` directory)
2. Analyze the error message from the compilation log
3. **Prioritize fixes by category**:
   - **Import errors**: Update import statements to match current PyMoo API
   - **Code execution errors**: Fix example code that no longer works
   - **Missing dependencies**: Add required imports or check environment
   - **API changes**: Update function calls to match current PyMoo interface
   - **File path issues**: Fix relative paths to data files or examples
   - **Syntax errors**: Fix markdown formatting or code syntax issues
4. Common documentation fixes include:
   - Updating import paths (e.g., `from pymoo.algorithms.moo.nsga2 import NSGA2`)
   - Fixing deprecated function calls
   - Correcting parameter names that have changed
   - Updating visualization code
   - Fixing file paths to example data

### Step 5: Iterative Compilation
After making fixes, run compilation again to process remaining files:

```bash
./tools/run docs compile
```

Key points:
- The compile command only processes files where the .ipynb is missing
- Since you're fixing files, their .ipynb files may have been deleted due to errors
- This creates a natural iteration cycle where only previously failed files get reprocessed

### Step 6: Iterate Until Complete
Repeat steps 4-5 until the compilation shows:
- **0 files need to be processed** (all .ipynb files exist)
- **No compilation errors**
- All .md files have been successfully converted and executed

### Step 7: Final Verification
Once all files compile successfully, run a complete build to ensure everything works:

```bash
./tools/run docs build
```

This verifies that:
- All notebooks were generated correctly
- Sphinx can build the HTML documentation
- No broken references or links exist
- The complete documentation builds successfully

## Useful Commands

### Core Commands
- `./tools/run docs clean` - Remove all generated .ipynb files
- `./tools/run docs compile` - Process missing notebook files only
- `./tools/run docs compile --force` - Force process all files (regenerate existing notebooks)
- `./tools/run docs build` - Build complete HTML documentation

### Specific File Processing
- `./tools/run docs compile file.md` - Process only a specific file
- `./tools/run docs compile file1.md file2.md` - Process multiple specific files
- `./tools/run docs compile "algorithms/*.md"` - Process all files in algorithms directory
- `./tools/run docs compile "algorithms/moo/*.md"` - Process files in specific subdirectory

### Debug Options
- `./tools/run docs compile --no-execute` - Convert .md to .ipynb without executing (syntax check)
- `./tools/run docs compile --force file.md` - Force regenerate a specific file

### Legacy Commands (Alternative)
- `./docs/clean.sh` - Alternative clean command
- `./docs/compile.sh` - Alternative compile command with more verbose output
- `./docs/compile.sh --help` - Detailed usage for legacy compile script

## Tips

1. **Always start with clean**: Use `./tools/run docs clean` to ensure a fresh start
2. **Fix one error at a time**: The iterative approach naturally focuses on remaining issues
3. **Check the logs**: Look at `docs/compile.log` for detailed error information
4. **Test specific files**: Use file-specific compilation to test fixes quickly
5. **Use --no-execute for syntax**: Test .md → .ipynb conversion without running code
6. **Look for patterns**: Similar files often have similar issues
7. **Check imports first**: Many failures are due to import path changes
8. **Update examples gradually**: Fix simpler examples first, then complex ones
9. **Verify environments**: Ensure the conda environment has required packages
10. **Test complete build**: Always run the final build step to catch integration issues

## Common Issues and Solutions

### Import Errors
- **Issue**: `ImportError` or `ModuleNotFoundError` in code blocks
- **Solution**: Update import statements to match current PyMoo structure
- **Example**: Change `from pymoo.algorithms.nsga2 import NSGA2` to `from pymoo.algorithms.moo.nsga2 import NSGA2`

### API Changes
- **Issue**: `AttributeError` or `TypeError` when calling PyMoo functions
- **Solution**: Check current PyMoo API and update function calls
- **Example**: Update deprecated parameter names or function signatures

### Missing Dependencies
- **Issue**: Code execution fails due to missing packages
- **Solution**: Add required imports or install missing packages in documentation environment
- **Example**: Add `import matplotlib.pyplot as plt` when plotting examples fail

### File Path Issues
- **Issue**: `FileNotFoundError` when loading example data
- **Solution**: Fix relative paths or ensure example data files exist
- **Example**: Update paths to test problems or data files

### Visualization Errors
- **Issue**: Plotting code fails with matplotlib or other visualization libraries
- **Solution**: Update plotting code to current API or add missing imports
- **Example**: Fix deprecated matplotlib function calls

### Syntax Errors
- **Issue**: Markdown formatting breaks notebook conversion
- **Solution**: Fix markdown syntax, especially code block formatting
- **Example**: Ensure proper triple-backtick code block formatting

## Progress Tracking

The compilation process provides clear progress indicators:

1. **After cleaning**: Shows total .md files found (e.g., "120 files found")
2. **During compilation**: Shows files being processed (e.g., "Processing 45 files...")
3. **Success indicator**: When compilation shows "No files to process" - this means all .ipynb files exist
4. **Error tracking**: Failed files are removed, so they'll be reprocessed in the next run

## Success Criteria

The fix-docs process is complete when:
1. `./tools/run docs clean` followed by `./tools/run docs compile` shows "No files to process"
2. All .md files have corresponding .ipynb files in the docs/source/ directory
3. No compilation or execution errors occur
4. `./tools/run docs build` completes successfully
5. The documentation builds without warnings or errors

## Environment Notes

- **Compilation environment**: Uses `default` conda environment (configurable via `CONDA_ENV`)
- **Build environment**: Uses `pymoo-doc` conda environment (configurable via `DOCS_ENV`)  
- **Configuration**: Environment settings stored in `tools/.env`
- **Dependencies**: Ensure both environments have required packages for PyMoo and documentation

## Advanced Usage

### Parallel Development
If working on specific documentation sections:

```bash
# Work on algorithm documentation only
./tools/run docs compile "algorithms/*.md"

# Work on getting started guide only  
./tools/run docs compile "getting_started/*.md"

# Test single file changes
./tools/run docs compile --force specific-file.md
```

### Debugging Workflow
For troubleshooting specific issues:

```bash
# 1. Clean everything
./tools/run docs clean

# 2. Test conversion only (no execution)
./tools/run docs compile --no-execute problem-file.md

# 3. If conversion works, test execution
./tools/run docs compile problem-file.md

# 4. Check detailed logs
cat docs/compile.log
```

This systematic approach ensures all PyMoo documentation compiles successfully and maintains consistency across the entire documentation system.
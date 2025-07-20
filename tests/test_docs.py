"""
Documentation pytest module.

Tests that all markdown files in docs/source/ can be converted to notebooks
and executed successfully using jupytext and jupyter nbconvert.
"""

import time
from pathlib import Path
from typing import List, Tuple

import pytest
import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class DocsManager:
    """Manages documentation pytest: discovery, conversion, and execution."""
    
    def __init__(self):
        self.pymoo_root = Path(__file__).parent.parent
        self.docs_source = self.pymoo_root / "docs" / "source"
    
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files, excluding underscore directories."""
        markdown_files = []
        
        for md_file in self.docs_source.rglob('*.md'):
            # Skip files in directories starting with underscore or containing checkpoints
            relative_path = md_file.relative_to(self.docs_source)
            skip = any(part.startswith('_') or '.ipynb_checkpoints' in part 
                      for part in relative_path.parts)
            
            if not skip:
                markdown_files.append(md_file)
        
        return sorted(markdown_files)
    
    def convert_to_notebook(self, md_file: Path) -> Tuple[bool, str]:
        """Convert markdown file to notebook using jupytext."""
        try:
            nb = jupytext.read(md_file)
            nb_file = md_file.with_suffix('.ipynb')
            self._write_notebook_with_fixed_lexer(nb, nb_file)
            return True, "Conversion successful"
        except Exception as e:
            return False, f"Conversion failed: {str(e)}"
    
    def _write_notebook_with_fixed_lexer(self, nb, nb_file: Path):
        """Write notebook with corrected lexer settings."""
        # Fix jupytext metadata
        if 'jupytext' in nb.metadata and 'default_lexer' in nb.metadata['jupytext']:
            if nb.metadata['jupytext']['default_lexer'] in ['ipython3', 'ipython']:
                nb.metadata['jupytext']['default_lexer'] = 'python3'
        
        # Fix language_info metadata
        if 'language_info' not in nb.metadata:
            nb.metadata['language_info'] = {"name": "python", "pygments_lexer": "python3"}
        elif 'pygments_lexer' in nb.metadata['language_info']:
            if nb.metadata['language_info']['pygments_lexer'] in ['ipython3', 'ipython']:
                nb.metadata['language_info']['pygments_lexer'] = 'python3'
        
        # Write the notebook
        with open(nb_file, 'w') as f:
            nbformat.write(nb, f)
    
    def execute_notebook(self, nb_file: Path) -> Tuple[bool, str, float]:
        """Execute notebook and return success, message, and execution time."""
        try:
            with open(nb_file) as f:
                nb = nbformat.read(f, as_version=4)
            
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            
            start_time = time.time()
            ep.preprocess(nb, {'metadata': {'path': str(nb_file.parent)}})
            execution_time = time.time() - start_time
            
            # Write back the executed notebook with fixed lexer
            self._write_notebook_with_fixed_lexer(nb, nb_file)
            
            return True, f"Executed successfully in {execution_time:.2f}s", execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            # Clean up failed notebook
            if nb_file.exists():
                nb_file.unlink()
            error = self._extract_error(str(e))
            return False, f"Execution failed: {error}", execution_time
    
    def _extract_error(self, error_str: str) -> str:
        """Extract meaningful error message from error string."""
        lines = [line.strip() for line in error_str.split('\n') if line.strip()]
        
        # Look for common error patterns
        for line in lines:
            if any(keyword in line for keyword in [
                'ImportError:', 'ModuleNotFoundError:', 'NameError:', 
                'ValueError:', 'TypeError:', 'AttributeError:',
                'CellExecutionError:', 'Error executing cell:'
            ]):
                return line[:150]  # Limit length
        
        # Fallback to last meaningful line
        if lines:
            return lines[-1][:150]
        
        return "Unknown execution error"


# Initialize manager and discover files
docs_manager = DocsManager()
MARKDOWN_FILES = docs_manager.find_markdown_files()


@pytest.mark.docs
@pytest.mark.long
@pytest.mark.parametrize('md_file', MARKDOWN_FILES, ids=lambda f: str(f.relative_to(docs_manager.docs_source)))
def test_documentation_file(md_file: Path):
    """Test that a documentation file converts to notebook and executes successfully."""
    nb_file = md_file.with_suffix('.ipynb')
    relative_path = md_file.relative_to(docs_manager.docs_source)

    # Always delete existing notebook first to ensure fresh conversion
    if nb_file.exists():
        nb_file.unlink()
    
    # Convert markdown to notebook
    success, message = docs_manager.convert_to_notebook(md_file)
    if not success:
        pytest.fail(f"Failed to convert {relative_path}: {message}")

    # Execute the notebook
    success, message, exec_time = docs_manager.execute_notebook(nb_file)
    
    if not success:
        pytest.fail(f"Failed to execute {relative_path}: {message}")
    
    print(f"✓ {relative_path} - {message}")


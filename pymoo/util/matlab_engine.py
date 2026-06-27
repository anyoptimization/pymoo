"""MATLAB engine integration utilities."""

from typing import Any


def install_matlab() -> None:
    """Print instructions for installing MATLAB Python interface."""
    print("Please install the Matlab python interface:")
    print(
        "Tutorial: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html"
    )
    print("---------------------------")
    print("Go to:")
    print("Windows:", 'cd "matlabroot\\extern\\engines\\python"')
    print("Linux/Mac:", 'cd "matlabroot/extern/engines/python"')
    print("python setup.py install")
    print("---------------------------")


class MatlabEngine:
    """Singleton wrapper for MATLAB engine instance.

    Launching the MATLAB engine is time-consuming, so this singleton pattern
    ensures only one instance is created and reused across multiple problems.
    """

    __instance: Any = None

    @staticmethod
    def get_instance() -> Any:
        """Get or create the MATLAB engine instance.

        Returns:
            The MATLAB engine instance.
        """
        if MatlabEngine.__instance is None:
            try:
                import matlab
                import matlab.engine
            except:  # noqa: E722
                install_matlab()

            MatlabEngine.__instance = matlab.engine.start_matlab(option="")

        return MatlabEngine.__instance

    @staticmethod
    def shutdown() -> None:
        """Shutdown the MATLAB engine instance."""
        MatlabEngine.__instance.quit()

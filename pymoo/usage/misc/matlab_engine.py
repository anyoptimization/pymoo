def install_matlab():
    print("Please install the Matlab python interface:")
    print("Tutorial: https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html")
    print("---------------------------")
    print("Go to:")
    print("Windows:", 'cd "matlabroot\extern\engines\python"')
    print("Linux/Mac:", 'cd "matlabroot/extern/engines/python"')
    print("python setup.py install")
    print("---------------------------")


class MatlabEngine:
    """
    Launching the Matlab engine can become time-consuming and thus shall only be done once and the instance
    could be reused by different kinds of problems at the same time.

    This is an implementation based on the singleton pattern where only one instance of the engine is used.

    """

    __instance = None

    @staticmethod
    def get_instance():
        if MatlabEngine.__instance is None:

            try:
                import matlab
                import matlab.engine
            except:
                print(install_matlab())

            MatlabEngine.__instance = matlab.engine.start_matlab(option="")

        return MatlabEngine.__instance

    @staticmethod
    def shutdown():
        MatlabEngine.__instance.quit()

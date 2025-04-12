import json
import os
import urllib.request
from os.path import join, dirname, abspath

import numpy as np

from pymoo.config import Config


class Remote:
    # -------------------------------------------------
    # Singleton Pattern
    # -------------------------------------------------
    __instance = None

    @staticmethod
    def get_instance():
        if Remote.__instance is None:
            server = Config.data()
            folder = join(dirname(dirname(abspath(__file__))), 'data')
            Remote.__instance = Remote(server, folder)
        return Remote.__instance

    # -------------------------------------------------

    def __init__(self, server, folder=None) -> None:
        super().__init__()
        self.server = server
        self.folder = folder

    def load(self, *args, to="numpy"):

        # the local file we can try loading
        f = join(str(self.folder), *args)

        # check if that path already exists
        if not os.path.exists(f):

            # if not make sure to create it that the file can be written
            folder = dirname(f)
            if not os.path.exists(folder):
                os.makedirs(folder)

            # create the url to load it from and download the file remotely
            url = self.server + "/".join(args)
            urllib.request.urlretrieve(url, f)

        if to == "numpy":
            return np.loadtxt(f)
        elif to == "json":
            with open(f) as file:
                return json.load(file)

        return f

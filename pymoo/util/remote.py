"""Remote data loading utilities using singleton pattern."""

import json
import os
import tempfile
import urllib.request
from os.path import abspath, dirname, exists, join

import numpy as np
from numpy.typing import NDArray

from pymoo.config import Config

# seconds to wait for the data server before giving up (override via env)
DOWNLOAD_TIMEOUT = float(os.environ.get("PYMOO_DATA_TIMEOUT", "30"))


def _user_cache_dir() -> str:
    """Writable per-user cache directory for downloaded data files.

    Honors ``PYMOO_DATA_DIR``, then ``XDG_CACHE_HOME``, falling back to
    ``~/.cache/pymoo``. Deliberately NOT the installed package directory, which is
    often read-only (system Python, Docker non-root, Nix, multi-user hosts).
    """
    override = os.environ.get("PYMOO_DATA_DIR")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or join(os.path.expanduser("~"), ".cache")
    return join(base, "pymoo")


class Remote:
    """Singleton class for loading remote data files."""

    __instance: "Remote | None" = None

    @staticmethod
    def get_instance() -> "Remote":
        """Get or create the singleton instance.

        Returns:
            The singleton Remote instance.
        """
        if Remote.__instance is None:
            server = Config.data()
            # data bundled (read-only) inside the package, if any — used as a
            # fallback so previously-cached files keep resolving offline.
            package_data = join(dirname(dirname(abspath(__file__))), "data")
            Remote.__instance = Remote(server, _user_cache_dir(), package_data)
        return Remote.__instance

    def __init__(
        self, server: str, folder: str | None = None, package_data: str | None = None
    ) -> None:
        """Initialize the Remote instance.

        Args:
            server: Server URL for remote data.
            folder: Writable local folder to cache downloaded data.
            package_data: Optional read-only folder of data bundled in the package.
        """
        super().__init__()
        self.server = server
        self.folder = folder
        self.package_data = package_data

    def _download(self, args: tuple[str, ...], dest: str) -> str:
        """Download ``server/<args>`` to *dest* atomically, with a timeout.

        Writes to a temp file in the destination directory and renames it into
        place, so an interrupted or failed download never leaves a truncated file
        at *dest* (which would otherwise be cached as if valid). On any failure the
        partial file is removed and a clear, actionable error is raised.
        """
        os.makedirs(dirname(dest), exist_ok=True)
        url = self.server + "/".join(args)
        tmp_fd, tmp = tempfile.mkstemp(dir=dirname(dest), suffix=".part")
        os.close(tmp_fd)
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp:
                data = resp.read()
            with open(tmp, "wb") as out:
                out.write(data)
            os.replace(tmp, dest)  # atomic on the same filesystem
        except Exception as e:
            if exists(tmp):
                os.remove(tmp)
            raise OSError(
                f"pymoo could not download required data from {url}: {e}. "
                f"Check your network connection, or pre-fetch the file and point "
                f"PYMOO_DATA_DIR at the directory containing it."
            ) from e
        return dest

    def load(self, *args: str, to: str = "numpy") -> NDArray | dict | str:
        """Load remote data file locally.

        Resolution order: the writable user cache, then a read-only copy bundled
        in the package (if present), otherwise download into the user cache.

        Args:
            *args: Path components to the remote file.
            to: Format to load ("numpy", "json", or "path").

        Returns:
            Loaded data in requested format (numpy array, dict, or file path).
        """
        # the writable cache location
        f = join(str(self.folder), *args)

        if not exists(f):
            # fall back to a read-only copy bundled in the package, if present
            bundled = join(str(self.package_data), *args) if self.package_data else None
            if bundled and exists(bundled):
                f = bundled
            else:
                f = self._download(args, f)

        if to == "numpy":
            return np.loadtxt(f)
        elif to == "json":
            with open(f) as file:
                return json.load(file)

        return f

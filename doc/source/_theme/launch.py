from pathlib import Path
from typing import Any, Dict, Optional

from docutils.nodes import document
from sphinx.application import Sphinx
from sphinx.util import logging
from shutil import copy2


SPHINX_LOGGER = logging.getLogger(__name__)


def add_hub_urls(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: Dict[str, Any],
    doctree: Optional[document],
):
    """Builds a binder link and inserts it in HTML context for use in templating.

    This is a ``html-page-context`` sphinx event (see :ref:`sphinx:events`).

    :param pagename: The sphinx docname related to the page
    :param context: A dictionary of values that are given to the template engine,
        to render the page and can be modified to include custom values.
    :param doctree: A doctree when the page is created from a reST documents;
        it will be None when the page is created from an HTML template alone.

    """

    # First decide if we'll insert any links
    path = app.env.doc2path(pagename)
    extension = Path(path).suffix

    # If so, insert the URLs depending on the configuration
    config_theme = app.config["html_theme_options"]
    launch_buttons = config_theme.get("launch_buttons", {})
    if not launch_buttons or not _is_notebook(app, pagename):
        return

    # Check if we have a markdown notebook, and if so then add a link to the context
    if _is_notebook(app, pagename) and (
        context["sourcename"].endswith(".md")
        or context["sourcename"].endswith(".md.txt")
    ):
        # Figure out the folders we want
        build_dir = Path(app.outdir).parent
        ntbk_dir = build_dir.joinpath("jupyter_execute")
        sources_dir = build_dir.joinpath("html", "_sources")
        # Paths to old and new notebooks
        path_ntbk = ntbk_dir.joinpath(pagename).with_suffix(".ipynb")
        path_new_notebook = sources_dir.joinpath(pagename).with_suffix(".ipynb")
        # Copy the notebook to `_sources` dir so it can be downloaded
        path_new_notebook.parent.mkdir(exist_ok=True, parents=True)
        copy2(path_ntbk, path_new_notebook)
        context["ipynb_source"] = pagename + ".ipynb"

    repo_url = _get_repo_url(config_theme)

    # Parse the repo parts from the URL
    org, repo = _split_repo_url(repo_url)
    if org is None and repo is None:
        # Skip the rest because the repo_url isn't right
        return

    branch = _get_branch(config_theme)

    # Construct the extra URL parts (app and relative path)
    notebook_interface_prefixes = {"classic": "tree", "jupyterlab": "lab/tree"}
    notebook_interface = launch_buttons.get("notebook_interface", "classic")
    if notebook_interface not in notebook_interface_prefixes:
        raise ValueError(
            (
                "Notebook UI for Binder/JupyterHub links must be one"
                f"of {tuple(notebook_interface_prefixes.keys())},"
                f"not {notebook_interface}"
            )
        )
    ui_pre = notebook_interface_prefixes[notebook_interface]

    # Check if we have a non-ipynb file, but an ipynb of same name exists
    # If so, we'll use the ipynb extension instead of the text extension
    if extension != ".ipynb" and Path(path).with_suffix(".ipynb").exists():
        extension = ".ipynb"

    # Construct a path to the file relative to the repository root
    book_relpath = config_theme.get("path_to_docs", "").strip("/")
    if book_relpath != "":
        book_relpath += "/"
    path_rel_repo = f"{book_relpath}{pagename}{extension}"

    # Now build infrastructure-specific links
    jupyterhub_url = launch_buttons.get("jupyterhub_url")
    binderhub_url = launch_buttons.get("binderhub_url")
    colab_url = launch_buttons.get("colab_url")
    if binderhub_url:
        url = (
            f"{binderhub_url}/v2/gh/{org}/{repo}/{branch}?"
            f"urlpath={ui_pre}/{path_rel_repo}"
        )
        context["binder_url"] = url

    if jupyterhub_url:
        url = (
            f"{jupyterhub_url}/hub/user-redirect/git-pull?"
            f"repo={repo_url}&urlpath={ui_pre}/{repo}/{path_rel_repo}&branch={branch}"
        )
        context["jupyterhub_url"] = url

    if colab_url:
        url = f"{colab_url}/github/{org}/{repo}/blob/{branch}/{path_rel_repo}"
        context["colab_url"] = url

    # Add thebe flag in context
    if launch_buttons.get("thebe", False):
        context["use_thebe"] = True


def _split_repo_url(url):
    """Split a repository URL into an org / repo combination."""
    if "github.com/" in url:
        end = url.split("github.com/")[-1]
        org, repo = end.split("/")[:2]
    else:
        SPHINX_LOGGER.warning(
            f"Currently Binder/JupyterHub repositories must be on GitHub, got {url}"
        )
        org = repo = None
    return org, repo


def _get_repo_url(config):
    repo_url = config.get("repository_url")
    if not repo_url:
        raise ValueError(
            "You must provide the key: `repository_url` to use launch buttons."
        )
    return repo_url


def _is_notebook(app, pagename):
    return app.env.metadata[pagename].get("kernelspec")


def _get_branch(config_theme):
    branch = config_theme.get("repository_branch")
    if not branch:
        branch = "master"
    return branch

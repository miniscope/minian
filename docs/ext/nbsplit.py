import json
import os
import re
import shutil
import warnings

import numpy as np


def is_spliter(cell):
    return cell["cell_type"] == "markdown" and bool(
        re.search(r"^#\s", cell["source"][0])
    )


def is_hv_init(cell):
    for src in cell["source"]:
        if bool(re.search(r"notebook_extension", src)):
            return True
    return False


def split_and_parse(app):
    for src_nb, out_dir in app.config.nbsplit_dict.items():
        if not os.path.exists(src_nb):
            warnings.warn(f"notebook not found: {src_nb}", RuntimeWarning)
            continue
        srcpath = os.path.split(src_nb)[0]
        outpath = os.path.join(os.path.abspath(app.srcdir), out_dir)
        os.makedirs(outpath, exist_ok=True)
        # copy images
        img_path = os.path.join(srcpath, "img")
        if os.path.exists(img_path):
            shutil.copytree(img_path, os.path.join(outpath, "img"))
        # read source notebook
        with open(src_nb) as notebook:
            jnote = json.load(notebook)
        cells = jnote["cells"]
        # split and process cells
        split = np.where([is_spliter(c) for c in cells])[0]
        assert split[0] == 0, "Notebook must start with a first-level heading!"
        hv_init = np.where([is_hv_init(c) for c in cells])[0]
        assert len(hv_init) > 0, "Can't find holoviews init cell!"
        hv_init = cells[hv_init.item()].copy()
        hv_init["source"] = ["# holoviews initialization"]
        hv_init["execution_count"] = None
        # write notebooks
        for isplit, start in enumerate(split):
            fname = f"notebook_{isplit}.ipynb"
            meta = jnote["metadata"].copy()
            meta["name"] = fname
            try:
                stop = split[isplit + 1]
            except IndexError:
                stop = None
            cur_cells = [hv_init] + cells[start:stop]
            cur_cells[0]["outputs"] = hv_init["outputs"]
            cur_notebook = {
                "cells": cur_cells,
                "metadata": meta,
                "nbformat": jnote["nbformat"],
                "nbformat_minor": jnote["nbformat_minor"],
            }
            with open(os.path.join(outpath, fname), "w") as nbfile:
                json.dump(cur_notebook, nbfile, indent=4)


def setup(app):
    app.add_config_value("nbsplit_dict", dict(), "html")
    app.connect("builder-inited", split_and_parse)

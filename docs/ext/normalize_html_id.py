import os
from bs4 import BeautifulSoup


def normalize_id(app, exception):
    if exception is not None:
        return
    build_dir = os.path.abspath(app.outdir)
    IDs = set()

    for root, dirs, files in os.walk(build_dir):
        if os.path.basename(root) in ["pipeline", "cross_reg"]:
            continue
        for html in list(filter(lambda fn: fn.endswith(".html"), files)):
            with open(os.path.join(root, html)) as html_doc:
                soup = BeautifulSoup(html_doc, "html.parser")
            for tag in soup.find_all(name="dt", id=True):
                tid = IDs.add(tag.get("id"))

    IDs = list(filter(lambda i: "." in i, IDs))
    ID_dict = {i: i.replace(".", "-") for i in IDs}

    for root, dirs, files in os.walk(build_dir):
        if os.path.basename(root) in ["pipeline", "cross_reg"]:
            continue
        for html in list(filter(lambda fn: fn.endswith(".html"), files)):
            with open(os.path.join(root, html)) as html_doc:
                soup = BeautifulSoup(html_doc, "html.parser")
            for tag in soup.find_all(name=True, id=True):
                tid = tag.get("id")
                try:
                    tag["id"] = ID_dict[tid]
                except KeyError:
                    pass
            for tag in soup.find_all(name=True, href=True):
                href = tag.get("href")
                for id_old, id_new in ID_dict.items():
                    tag["href"] = tag["href"].replace(id_old, id_new)
            with open(os.path.join(root, html), "w", encoding="utf-8") as outf:
                outf.write(str(soup))


def setup(app):
    app.connect("build-finished", normalize_id)

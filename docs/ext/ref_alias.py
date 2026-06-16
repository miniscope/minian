"""
A small Sphinx extension to map arbitrary alias for intersphinx resolving

ref: https://stackoverflow.com/questions/62293058/how-to-add-objects-to-sphinxs-global-index-or-cross-reference-by-alias
"""

from sphinx.addnodes import pending_xref
from sphinx.ext.intersphinx import missing_reference
from docutils.nodes import Text


def resolve_intersphinx_aliases(app, env, node, contnode):
    reftarget_aliases = app.config.ref_aliases
    alias = node.get("reftarget", None)
    if alias is not None and alias in reftarget_aliases:
        real_ref, text_to_render = reftarget_aliases[alias]
        # this will resolve the ref
        node["reftarget"] = real_ref

        # this will rewrite the rendered text:
        # find the text node child
        text_node = next(iter(contnode.findall(lambda n: n.tagname == "#text")))
        # remove the old text node, add new text node with custom text
        text_node.parent.replace(text_node, Text(text_to_render))

        # delegate all the rest of dull work to intersphinx
        return missing_reference(app, env, node, contnode)


def setup(app):
    app.add_config_value("ref_aliases", {}, "html")
    app.connect("missing-reference", resolve_intersphinx_aliases)

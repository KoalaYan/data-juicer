"""Mapper exports.

The default remains backwards compatible and imports every mapper. Portrait
pipelines set ``DATA_JUICER_LAZY_OP_IMPORT=1`` and let ``ops.load`` import
only modules named in their process configuration.
"""

import os


if os.environ.get("DATA_JUICER_LAZY_OP_IMPORT", "").lower() not in {
    "1",
    "true",
    "yes",
    "on",
}:
    from . import _eager

    __all__ = _eager.__all__
    globals().update({name: getattr(_eager, name) for name in __all__})
else:
    __all__ = []

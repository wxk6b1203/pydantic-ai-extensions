"""Community extensions for Pydantic AI."""

from .context_compression import ContextCompression, SummaryRecord, SummaryStore
from .version import (
    __branch__,
    __commit__,
    __commit_full__,
    __describe__,
    __dirty__,
    __version__,
    __version_info__,
    get_version,
)

__all__ = [
    "ContextCompression",
    "SummaryRecord",
    "SummaryStore",
    "__branch__",
    "__commit__",
    "__commit_full__",
    "__describe__",
    "__dirty__",
    "__version__",
    "__version_info__",
    "get_version",
]

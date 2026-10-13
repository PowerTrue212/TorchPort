"""兼容入口：TaskSpec 与内容身份实现位于 :mod:`torchport.contracts`。"""

from .contracts import (
    CAPTURE_VERSION,
    CODE_SUFFIXES,
    DEFAULT_PROTOCOL,
    JUDGE_KEYS,
    SCHEMA_VERSION,
    SIDES,
    SKIP_DIRS,
    digest,
    file_hash,
    load_task,
    normalize,
    oracle_key,
    side_identity,
    tree_hash,
)

__all__ = [
    "CAPTURE_VERSION",
    "CODE_SUFFIXES",
    "DEFAULT_PROTOCOL",
    "JUDGE_KEYS",
    "SCHEMA_VERSION",
    "SIDES",
    "SKIP_DIRS",
    "digest",
    "file_hash",
    "load_task",
    "normalize",
    "oracle_key",
    "side_identity",
    "tree_hash",
]

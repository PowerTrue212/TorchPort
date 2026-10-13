"""兼容入口；聚合实现位于 reporting.aggregate。"""

from .reporting.aggregate import aggregate, main

__all__ = ["aggregate", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

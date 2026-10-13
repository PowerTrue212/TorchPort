"""兼容入口；性能测量实现位于 reporting.performance。"""

from .reporting.performance import main, measure_performance

__all__ = ["measure_performance", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

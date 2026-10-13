"""兼容入口；provider 实现位于 agents.deepseek。"""

from .agents.deepseek import (
    DeepSeekAgent,
    DeepSeekConfig,
    _usage,
    load_env,
    main,
    post_json,
)

__all__ = ["DeepSeekAgent", "DeepSeekConfig", "_usage", "load_env", "post_json", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

"""把 API key / 端点准备好，并给出**能照着排查的**报错。

============================ 为什么要自己读环境变量文件 ============================
`~/.bashrc` 顶上有这么一段（Ubuntu 默认）：

    case $- in
        *i*) ;;
          *) return;;
    esac

也就是说**非交互 shell 根本不会执行后面的 export**。从编辑器、cron、systemd、
Jupyter、CI、或 `bash -c 'python ...'` 里启动的进程都拿不到 `SFZY_ELEMENT_EXTRACT`，
而手工在终端里跑同样的命令却是好的——这种"我明明配了 key"的坑排查起来很费时间。

所以这里显式支持"从环境变量文件里补读"：进程环境 → `api.env_file`（配置指定）
→ `<repo>/.env` → `~/.bashrc` → `~/.profile`。找到就 `setdefault` 进环境，
日志里只打掩码后的前缀。

============================ 为什么按文件路径加载 api_runtime ============================
`sfzy/judge/__init__.py` 会 import 到 torch 那条链（judge.runtime）。抽取这一路
只需要一个纯标准库的 HTTP 客户端，所以按文件路径加载 `api_runtime.py`
（`tests/test_api_runtime.py` 用的是同一个办法），让本模块保持 torch-free。
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sfzy.elements.config import ApiConfig

logger = logging.getLogger("sfzy.elements")

ROOT = Path(__file__).resolve().parents[3]

#: 环境变量文件的候选位置（按优先级）。`~/.bashrc` 在列是因为很多人习惯写在那里。
DEFAULT_ENV_FILES = (".env", "~/.bashrc", "~/.profile")

_ENV_LINE = re.compile(
    r"""^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$"""
)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # 去掉行尾注释（`KEY=abc  # 说明`）——但别把值里的 # 当注释
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def read_env_file(path: Path) -> Dict[str, str]:
    """读一个 `KEY=VALUE` / `export KEY=VALUE` 文件（不执行任何 shell 语法）。"""

    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), _strip_quotes(match.group(2))
        if value:
            out[name] = value
    return out


def load_env_files(paths: Sequence[Path]) -> List[Path]:
    """把文件里的变量补进环境（已存在的**不覆盖**），返回真正补了值的文件。"""

    used: List[Path] = []
    for path in paths:
        values = read_env_file(path)
        # `KEY: ${KEY}` 这类 shell 写法读出来是垃圾，直接判为无效
        values = {k: v for k, v in values.items() if "${" not in v and "$" not in v}
        added = {k: v for k, v in values.items() if k not in os.environ}
        if added:
            os.environ.update(added)
            used.append(path)
    return used


def env_file_candidates(api: ApiConfig) -> List[Path]:
    """按优先级列出"可以去哪里找 key"：配置指定 → 仓库 .env → ~/.bashrc → ~/.profile。"""

    paths: List[Path] = []
    if api.env_file:
        paths.append(Path(api.env_file).expanduser())
    for spec in DEFAULT_ENV_FILES:
        paths.append(ROOT / spec if not spec.startswith("~") else Path(spec).expanduser())
    seen: set = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def ensure_api_key(api: ApiConfig) -> Optional[Path]:
    """保证 `api.api_key_env` 在环境里可用；返回**补读成功的文件**（没有则 None）。"""

    if api.api_key or os.environ.get(api.api_key_env):
        return None
    used = load_env_files(env_file_candidates(api))
    for path in used:
        if api.api_key_env in read_env_file(path):
            return path
    return None


def mask(secret: Optional[str]) -> str:
    if not secret:
        return "<空>"
    return f"{secret[:6]}…{secret[-4:]}（长度 {len(secret)}）" if len(secret) > 12 else "***"


def _load_api_runtime_module():
    """按文件路径加载 `sfzy/judge/api_runtime.py`（绕开 judge 包的 torch 依赖）。"""

    name = "sfzy_api_runtime"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "src" / "sfzy" / "judge" / "api_runtime.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:              # pragma: no cover
        raise ImportError(f"加载不了 api_runtime：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_runtime(api: ApiConfig) -> Any:
    """构造 `APIRuntime`（失败时把"key 到底有没有读到"讲清楚）。"""

    filled = ensure_api_key(api)
    if filled is not None:
        logger.info("从 %s 补读了 $%s", filled, api.api_key_env)
    if not os.environ.get(api.api_key_env) and not api.api_key:
        raise RuntimeError(
            f"没有拿到 API 密钥：环境变量 $%s 为空，且这些文件里也没找到："
            f"{[str(p) for p in env_file_candidates(api)]}。\n"
            "  排查：① 把 `export SFZY_ELEMENT_EXTRACT=...` 写进 ~/.bash_profile 或 "
            "用 `api.env_file` 指定一个 KEY=VALUE 文件；\n"
            "  ② 注意 ~/.bashrc 在**非交互 shell 里会被跳过**（顶上那段 case $- 判断）。"
            % api.api_key_env
        )
    module = _load_api_runtime_module()
    # 走 judge 那边的 build_api_runtime：配置键的映射（timeout_s → timeout 等）
    # 只有那一处知道，自己再抄一份迟早两边漂移。
    runtime = module.build_api_runtime(api.as_runtime_config())
    logger.info(
        "抽取服务：%s model=%s key=%s（$%s）",
        runtime.base_url,
        runtime.model,
        mask(runtime.api_key),
        api.api_key_env,
    )
    return runtime


def explain_api_error(exc: BaseException) -> str:
    """把 API 报错翻译成"下一步该干什么"。"""

    text = str(exc)
    if "402" in text or "insufficient_funds" in text or "余额" in text:
        return (
            "调用被拒：**账户余额不足**（HTTP 402，账户级，换模型也一样）。\n"
            "  这是计费侧的问题，不是 key 或 base_url 的问题；充值后重试即可。\n"
            f"  原始报错：{text[:300]}"
        )
    if "401" in text or "Authentication" in text or "invalid_api_key" in text:
        return (
            "调用被拒：**鉴权失败**（HTTP 401）。key 可能过期/被截断，"
            "或带着换行空白发出去。\n"
            f"  原始报错：{text[:300]}"
        )
    if "404" in text or "model" in text and "not" in text:
        return f"调用被拒：模型名或 base_url 不对。\n  原始报错：{text[:300]}"
    if re.search(r"timed?\s*out|TimeoutError|ReadTimeout|ConnectTimeout", text, re.I):
        return f"调用超时：可以调大 api.timeout_s，或降低 api.concurrency。\n  原始报错：{text[:300]}"
    return text


# ---------------------------------------------------------------- 聊天客户端


@dataclass
class Completion:
    """一次回复。**推理模型**会把思维链放在 `reasoning` 里，`content` 才可能是空的。"""

    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: 这一条调用本身失败了（超时/限流/额度…）。**单条失败不能拖垮整批**：
    #: 40 条里有一条慢，另外 39 条的结果必须留下。
    error: str = ""

    @property
    def truncated_by_length(self) -> bool:
        return self.finish_reason == "length"


class ChatClient:
    """批量聊天：直接用 runtime 的 HTTP 层，好拿到 `reasoning_content` 和 `finish_reason`。

    为什么不走 `runtime.generate_messages`：它只回 `message.content`。推理模型
    （例如 teleai/xing4.0）常常把预算全花在思维链上，此时 `content` 是空串而
    `reasoning_content` 有几千字——只看 `content` 会变成"模型什么都没输出"，
    排查时完全看不出是 `max_tokens` 不够。

    并发与重试沿用 runtime 自己的（`_post` 内部带指数退避重试）。
    """

    def __init__(self, runtime: Any, *, max_tokens: int = 1024, json_mode: bool = True) -> None:
        self.runtime = runtime
        self.max_tokens = int(max_tokens)
        self.json_mode = bool(json_mode)
        self.concurrency = max(1, int(getattr(runtime, "concurrency", 8)))
        self._can_use_post = callable(getattr(runtime, "_post", None))
        self._done = 0
        self._lock = threading.Lock()
        self._total = 0

    def chat(self, conversations: Sequence[List[Dict[str, str]]]) -> List[Completion]:
        if not conversations:
            return []
        self._total = len(conversations)
        self._done = 0
        if not self._can_use_post:
            # 没有 HTTP 层的 runtime（测试替身）：退回标准接口，拿不到思维链
            texts = self.runtime.generate_messages(list(conversations), self.max_tokens)
            return [Completion(content=text) for text in texts]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, conversations))

    def _one(self, messages: List[Dict[str, str]]) -> Completion:
        try:
            body = self.runtime._post(
                messages, max_tokens=self.max_tokens, json_mode=self.json_mode
            )
        except Exception as exc:                                  # noqa: BLE001
            return Completion(error=explain_api_error(exc))
        with self._lock:
            self._done += 1
            if self._total >= 8 and (self._done % max(1, self._total // 10) == 0):
                logger.info("抽取进度 %d/%d", self._done, self._total)
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        return Completion(
            content=(message.get("content") or "").strip(),
            reasoning=(message.get("reasoning_content") or message.get("reasoning") or "").strip(),
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def diagnose_empty_completion(completion: Completion) -> str:
    """回复里没有可解析内容时，给出**具体**原因，而不是"解析失败"。"""

    if completion.reasoning and not completion.content:
        if completion.truncated_by_length:
            return (
                f"模型只输出了思维链（{len(completion.reasoning)} 字）就被 max_tokens 截断，"
                "没有输出最终 JSON。推理模型要留足预算：把 api.max_tokens 调到 4096 以上。"
            )
        return (
            f"模型只输出了思维链（{len(completion.reasoning)} 字），没有输出最终 JSON。"
            "可以调大 api.max_tokens，或在 prompt 里明确要求「思考完成后必须输出 JSON」。"
        )
    if completion.truncated_by_length:
        return "回复被 max_tokens 截断（finish_reason=length），JSON 不完整。请调大 api.max_tokens。"
    if not completion.content:
        return "回复为空（content 为空且没有思维链）。检查模型名 / 供应商是否正常。"
    return "回复里没有可解析的 JSON。"


__all__ = [
    "build_runtime",
    "ensure_api_key",
    "explain_api_error",
    "ChatClient",
    "Completion",
    "diagnose_empty_completion",
    "load_env_files",
    "read_env_file",
    "env_file_candidates",
    "mask",
    "DEFAULT_ENV_FILES",
    "ROOT",
]

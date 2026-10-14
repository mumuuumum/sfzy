"""抽取 + 逐要素 ROUGE 的配置解析与校验。

============================ 权重只能来自配置文件 ============================
沿用 `sfzy/rl/reward_spec.py` 的规矩：**代码里不存权重数值**。

    element_rouge:
      api:                            # OpenAI 兼容端点
        base_url: https://api.tokenplus.cn/v1
        model: teleai/xing4.0-29b-a4b
        api_key_env: SFZY_ELEMENT_EXTRACT
      prompt:                         # 抽取 prompt（{summary} 是正文占位符）
        system: ...
        user: ...
      rouge:
        mode: jieba                   # char | jieba
        weights: {rouge1: 0.2, rouge2: 0.4, rougel: 0.4}
        element_weights:              # 三要素在"总分"里的权重（可选）
          plaintiff_claims: 0.30
          defendant_defenses: 0.20
          court_facts: 0.50

校验口径和 reward 那边一致：权重必须齐全、非负、和 > 0；缺了就在**启动时**报错
并告诉你该往哪一段写什么，而不是悄悄用一个藏在源码里的默认值。

`rouge.weights` 必须显式写（这是本次可配置的核心诉求）；`element_weights` 允许
省略，省略时按 `DEFAULT_ELEMENT_WEIGHTS`（与 judge 的六要素权重同比例归一化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from sfzy.elements.prompts import PromptTemplate
from sfzy.elements.schema import ELEMENTS

#: ROUGE 三项的键名（kebab 风格与 `sfzy.eval.rouge.score_pair` 的输出对齐）。
ROUGE_KEYS: Tuple[str, ...] = ("rouge1", "rouge2", "rougel")

#: `element_weights` 省略时的默认（judge 六要素里这三个的权重，归一化后）。
DEFAULT_ELEMENT_WEIGHTS: Dict[str, float] = {
    "plaintiff_claims": 0.30,
    "defendant_defenses": 0.20,
    "court_facts": 0.50,
}

#: 官方口径（`sfzy.eval.rouge.OFFICIAL_WEIGHTS`），写进配置文件的模板里。
OFFICIAL_ROUGE_WEIGHTS: Dict[str, float] = {"rouge1": 0.2, "rouge2": 0.4, "rougel": 0.4}


class ElementConfigError(ValueError):
    """配置有问题，启动时就该停下。"""


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ElementConfigError(f"{where} 必须是数字，收到 {value!r}") from None


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    model: str
    api_key_env: str = "SFZY_ELEMENT_EXTRACT"
    api_key: Optional[str] = None
    #: 环境变量文件（KEY=VALUE / export KEY=VALUE）。留空则按顺序尝试
    #: `<repo>/.env` → `~/.bashrc` → `~/.profile`：见 `client.ensure_api_key`
    #: ——非交互 shell 不会 source .bashrc，只有从这里补读才拿得到 key。
    env_file: Optional[str] = None
    timeout_s: float = 90.0
    max_retries: int = 3
    concurrency: int = 16
    temperature: float = 0.0
    max_tokens: int = 1024
    response_format_json: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    #: 解析失败时额外重试几次（格式问题重发一次通常就好）
    parse_retries: int = 1

    def as_runtime_config(self) -> Dict[str, Any]:
        """转成 `sfzy.judge.api_runtime.APIRuntime` 认识的那一段配置。"""

        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key,
            "api_key_env": self.api_key_env,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "concurrency": self.concurrency,
            "temperature": self.temperature,
            "response_format_json": self.response_format_json,
            "extra_body": dict(self.extra_body),
            "extra_headers": dict(self.extra_headers),
        }

    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "ApiConfig":
        if not isinstance(raw, Mapping):
            raise ElementConfigError(
                "element_rouge.api 必须是 {base_url, model, ...} 形式的映射"
            )
        base_url = str(raw.get("base_url", "")).strip()
        model = str(raw.get("model", "")).strip()
        if not base_url or not model:
            raise ElementConfigError(
                "element_rouge.api 里必须写 base_url 和 model（抽取要打到哪台服务、哪个模型）"
            )
        env_file = raw.get("env_file")
        return cls(
            base_url=base_url,
            model=model,
            api_key_env=str(raw.get("api_key_env", "SFZY_ELEMENT_EXTRACT")),
            api_key=raw.get("api_key"),
            env_file=None if env_file in (None, "") else str(env_file),
            timeout_s=_as_float(raw.get("timeout_s", 90), "api.timeout_s"),
            max_retries=int(raw.get("max_retries", 3)),
            concurrency=int(raw.get("concurrency", 16)),
            temperature=_as_float(raw.get("temperature", 0.0), "api.temperature"),
            max_tokens=int(raw.get("max_tokens", 1024)),
            response_format_json=bool(raw.get("response_format_json", True)),
            extra_body=dict(raw.get("extra_body") or {}),
            extra_headers=dict(raw.get("extra_headers") or {}),
            parse_retries=int(raw.get("parse_retries", 1)),
        )


@dataclass(frozen=True)
class RougeConfig:
    """逐要素 ROUGE 的口径。"""

    mode: str
    weights: Dict[str, float]
    element_weights: Dict[str, float]
    #: 只有一侧为空时的得分（默认 0：没抽出来就该拿 0，而不是被排除掉）
    one_side_empty_score: float = 0.0
    #: 两侧都为空（人工摘要里本来就没有这一要素）时：exclude = 不计入平均
    both_empty: str = "exclude"

    def normalized_element_weights(self) -> Dict[str, float]:
        total = sum(self.element_weights.values())
        return {k: v / total for k, v in self.element_weights.items()}

    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "RougeConfig":
        if not isinstance(raw, Mapping):
            raise ElementConfigError(
                "element_rouge.rouge 必须是 {mode, weights, element_weights} 形式的映射"
            )
        mode = str(raw.get("mode", "jieba"))
        if mode not in ("char", "jieba"):
            raise ElementConfigError(f"rouge.mode 只能是 char 或 jieba，收到 {mode!r}")

        # ---- 三项权重：必须显式写全（这是本次要可配置的东西）----
        weights_raw = raw.get("weights")
        if not isinstance(weights_raw, Mapping):
            raise ElementConfigError(
                "rouge.weights 必须写全三项，例如 "
                f"{{rouge1: 0.2, rouge2: 0.4, rougel: 0.4}}（当前：{weights_raw!r}）"
            )
        weights: Dict[str, float] = {}
        for key in ROUGE_KEYS:
            if key not in weights_raw:
                raise ElementConfigError(
                    f"rouge.weights 缺了 {key}；三项都要写：{list(ROUGE_KEYS)}"
                )
            value = _as_float(weights_raw[key], f"rouge.weights.{key}")
            if value < 0:
                raise ElementConfigError(f"rouge.weights.{key} 不能为负：{value}")
            weights[key] = value
        total = sum(weights.values())
        if total <= 0:
            raise ElementConfigError(f"rouge.weights 三项之和必须大于 0，当前 {total}")

        # ---- 要素权重：可省略（省略时用默认，见模块 docstring）----
        element_weights_raw = raw.get("element_weights")
        if element_weights_raw is None:
            element_weights = dict(DEFAULT_ELEMENT_WEIGHTS)
        else:
            if not isinstance(element_weights_raw, Mapping):
                raise ElementConfigError(
                    "rouge.element_weights 必须是 {要素: 权重} 形式的映射"
                )
            unknown = sorted(set(element_weights_raw) - set(ELEMENTS))
            if unknown:
                raise ElementConfigError(
                    f"rouge.element_weights 里有不认识的要素 {unknown}；可用 {list(ELEMENTS)}"
                )
            element_weights = {}
            for name in ELEMENTS:
                if name not in element_weights_raw:
                    raise ElementConfigError(
                        f"rouge.element_weights 缺了 {name}；三个要素都要写：{list(ELEMENTS)}"
                    )
                value = _as_float(element_weights_raw[name], f"rouge.element_weights.{name}")
                if value < 0:
                    raise ElementConfigError(
                        f"rouge.element_weights.{name} 不能为负：{value}"
                    )
                element_weights[name] = value
            if sum(element_weights.values()) <= 0:
                raise ElementConfigError("rouge.element_weights 之和必须大于 0")

        both_empty = str(raw.get("both_empty", "exclude"))
        if both_empty not in ("exclude", "one"):
            raise ElementConfigError(
                f"rouge.both_empty 只能是 exclude 或 one，收到 {both_empty!r}"
            )
        return cls(
            mode=mode,
            weights=weights,
            element_weights=element_weights,
            one_side_empty_score=_as_float(
                raw.get("one_side_empty_score", 0.0), "rouge.one_side_empty_score"
            ),
            both_empty=both_empty,
        )


@dataclass(frozen=True)
class ElementRougeConfig:
    """整段 `element_rouge` 配置。"""

    api: ApiConfig
    prompt: PromptTemplate
    rouge: RougeConfig
    #: 抽取结果缓存（人工摘要离线抽一次就够；候选摘要也会命中重复文本）
    cache_path: Optional[str] = None
    output_dir: str = "outputs/element_rouge"

    @classmethod
    def from_config(cls, raw: Optional[Mapping[str, Any]]) -> "ElementRougeConfig":
        if not isinstance(raw, Mapping):
            raise ElementConfigError(
                "配置里没有 element_rouge 这一段；抽取端点、prompt、ROUGE 权重都要写在里面"
            )
        cache_path = raw.get("cache_path")
        return cls(
            api=ApiConfig.from_config(raw.get("api")),
            prompt=PromptTemplate.from_config(raw.get("prompt")),
            rouge=RougeConfig.from_config(raw.get("rouge")),
            cache_path=None if cache_path in (None, "") else str(cache_path),
            output_dir=str(raw.get("output_dir", "outputs/element_rouge")),
        )


def load_element_config(cfg: Mapping[str, Any]) -> ElementRougeConfig:
    """从整份 YAML（已 load 成 dict）里取 `element_rouge` 一段。"""

    return ElementRougeConfig.from_config(cfg.get("element_rouge"))


__all__ = [
    "ElementConfigError",
    "ApiConfig",
    "RougeConfig",
    "ElementRougeConfig",
    "load_element_config",
    "ROUGE_KEYS",
    "OFFICIAL_ROUGE_WEIGHTS",
    "DEFAULT_ELEMENT_WEIGHTS",
]

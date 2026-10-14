"""用大模型 API 抽三要素，并逐要素算 ROUGE（人工摘要 vs GRPO 候选摘要）。

    from sfzy.elements import (
        ElementRougeConfig, ElementExtractor, ElementRougePipeline, build_runtime,
    )

    cfg = ElementRougeConfig.from_config(yaml.safe_load(open("configs/element_extract.yaml"))["element_rouge"])
    runtime = build_runtime(cfg.api)                  # 密钥从 $SFZY_ELEMENT_EXTRACT 读
    extractor = ElementExtractor(runtime, cfg.prompt, cache=..., model=cfg.api.model,
                                 max_tokens=cfg.api.max_tokens, parse_retries=cfg.api.parse_retries)
    pipeline = ElementRougePipeline(extractor, cfg)
    ref, per_candidate = pipeline.evaluate(人工摘要, {"grpo": 候选摘要})

命令行入口：`tools/element_rouge.py`（配置见 `configs/element_extract.yaml`）。
本包保持 **torch-free**：HTTP 客户端按文件路径加载，不经过 `sfzy.judge.__init__`。
"""

from sfzy.elements.client import build_runtime, ensure_api_key, explain_api_error
from sfzy.elements.config import (
    ApiConfig,
    ElementConfigError,
    ElementRougeConfig,
    RougeConfig,
    load_element_config,
)
from sfzy.elements.extract import ElementCache, ElementExtractor, Extraction, build_extractor
from sfzy.elements.pipeline import ElementRougePipeline, SampleResult
from sfzy.elements.prompts import PromptTemplate, default_prompt
from sfzy.elements.schema import ELEMENTS, ELEMENT_ZH, ElementSet, parse_elements
from sfzy.elements.score import ElementScore, aggregate, score_elements, weighted_score

__all__ = [
    "ELEMENTS",
    "ELEMENT_ZH",
    "ElementSet",
    "parse_elements",
    "PromptTemplate",
    "default_prompt",
    "ApiConfig",
    "RougeConfig",
    "ElementRougeConfig",
    "ElementConfigError",
    "load_element_config",
    "build_runtime",
    "ensure_api_key",
    "explain_api_error",
    "ElementExtractor",
    "ElementCache",
    "Extraction",
    "build_extractor",
    "ElementRougePipeline",
    "SampleResult",
    "ElementScore",
    "aggregate",
    "score_elements",
    "weighted_score",
]

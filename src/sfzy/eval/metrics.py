"""生成式语义指标：把"摘要好不好"交给一个 LLM 裁判来打分。

============================ 为什么不用余弦类指标 ============================
实测（1340 条 SFT 输出，`tools/bench_metrics.py`）：

    把金额改错一位          余弦 -0.0004   ROUGE-L -0.0022
    调换两个相邻句子的顺序   余弦 -0.0011   ROUGE-L -0.0333

余弦的问题不是"没区分度"，是**动态范围塌了**：取值全挤在 0.95~0.99，
语义破坏带来的降幅只有语序扰动的 36%。GRPO 的 advantage 要除以组内标准差，
这点差异放大的全是噪声。**不能当 reward 用的原因是信噪比，不是方向。**

生成式裁判反过来：它逐项核对，读得懂"48000 被写成 48001"是错的。
代价是慢（一条 0.3~1.5s），所以只用在 RL 的 reward 和最终评测上。

============================ 三种后端 ============================
  LocalJudgeScorer   本地 transformers 跑一个 7B 裁判（占一张卡）
  APIJudgeScorer     OpenAI 兼容接口（DeepSeek / GLM / GPT），不占显存
  CachedJudgeScorer  读预先算好的分数，只用于离线评测与对比

三者接口一致：`score_batch(items) -> List[Optional[float]]`，量程 0-100。
`None` 表示这条打分失败（截断/超时），**不能当 0 用** —— 失败和"很差"
是两件事，混起来会让统计系统性偏低。

============================ 两个 judge 的原则 ============================
reward 用的 judge 和**对外汇报**用的 judge 必须是两个（不同模型或不同 rubric
版本）。否则"语义指标提升"只是"我们优化的那个函数涨了"，是循环论证。
所以 `rubric_file` 是可配置的，配置里同时准备好 v1（进奖励）和 v2（汇报）。
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]

_SCORE_RE = re.compile(r"(?:总分|总分是|最终得分|最终分数|得分|分数)\D{0,6}(\d{1,3})")


# ---------------------------------------------------------------------------
# 评分标准与解析（与后端无关的纯函数，单独测）
# ---------------------------------------------------------------------------
def load_rubric(path: Optional[str] = None) -> Dict[str, Any]:
    """读取 rubric。缺省用仓库里的 v1。缺 pyyaml 时退化成"整份当模板"。"""
    p = Path(path) if path else ROOT / "configs" / "judge_rubric_v1.yaml"
    if not p.is_absolute():
        p = ROOT / p
    text = p.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return {"template": text, "scale": 100}


def build_judge_prompt(
    rubric: Dict[str, Any],
    reference: str,
    candidate: str,
    source: Optional[str] = None,
) -> str:
    """填模板。

    `{source}` 缺省时用短模板（只用参考摘要当金标准，快一倍）；
    给了原文就用长模板 —— **查幻觉必须给原文**，否则裁判没法判断
    某个金额是不是编的。
    """
    key = "template_with_source" if source else "template"
    tpl = rubric.get(key) or rubric["template"]
    return (
        tpl.replace("{reference}", (reference or "").strip())
        .replace("{candidate}", (candidate or "").strip())
        .replace("{source}", (source or "").strip())
    )


def parse_judge_score(text: str, scale: int = 100) -> Optional[float]:
    """从裁判输出里抠分数，失败返回 None。

    取**最后一个**带"总分/最终分数"字样的数字：rubric 要求逐维度写理由，
    每个维度行里也可能出现"此维度得分 20"，取第一个会抄成维度分。
    找不到关键字就退化成最后一个数字；超量程一律判失败。
    """
    last = None
    for m in _SCORE_RE.finditer(text):
        last = m
    if last is not None:
        value = float(last.group(1))
    else:
        nums = re.findall(r"\d{1,3}(?:\.\d+)?", text)
        if not nums:
            return None
        value = float(nums[-1])
    if value < 0 or value > scale:
        return None
    return value


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """统一字段名：三元组用 `output`，压力测试台导出用 `text`。"""
    return {
        "candidate": item.get("candidate") or item.get("output") or item.get("text") or "",
        "reference": item.get("reference", ""),
        "source": item.get("source"),
        "id": item.get("id"),
    }


# ---------------------------------------------------------------------------
# 后端
# ---------------------------------------------------------------------------
class SemanticScorer:
    """所有裁判后端的统一接口。"""

    name = "semantic"

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        raise NotImplementedError


class LocalJudgeScorer(SemanticScorer):
    """本地 transformers 裁判。

    逐条生成而不是批处理：批处理要左 padding，而带自定义 cache 的模型
    （ChatGLM3 那一类）在 padding 下会崩，见 `models/compat.py`。
    裁判模型通常是标准架构，批处理是安全的，但收益不值得这份复杂度 ——
    裁判只在 RL 的 rollout 上跑，G=8、每步 32 条，逐条完全够。
    """

    name = "judge_local"

    def __init__(
        self,
        model_path: str,
        rubric: Optional[Dict[str, Any]] = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        samples: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.rubric = rubric or load_rubric()
        self.scale = int(self.rubric.get("scale", 100))
        self.samples = samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs.pop("torch_dtype")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if not load_in_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        self._torch = torch
        self.n_truncated = 0     # 被 max_new_tokens 截断而丢弃的次数，CLI 会报告
        self.last_raw: List[List[str]] = []   # 每条样本的裁判原文，供人工核查

    def _render(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _one(self, prompt: str) -> List[float]:
        torch = self._torch
        text = self._render(prompt)
        ids = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out: List[float] = []
        raws: List[str] = []
        eos = self.tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(self.samples):
                gen = self.model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.samples > 1,
                    temperature=self.temperature if self.samples > 1 else None,
                    top_p=0.95 if self.samples > 1 else None,
                    pad_token_id=self.tokenizer.pad_token_id or eos,
                )
                new_ids = gen[0][ids["input_ids"].shape[1]:]
                decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                raws.append(decoded)
                # 截断即丢弃：被截断时最后一行往往是维度分，正则会把维度分
                # 当总分抄走 —— 错误分数比没有分数危险得多（实现时踩过）。
                if not len(new_ids) or int(new_ids[-1]) != eos:
                    self.n_truncated += 1
                    continue
                score = parse_judge_score(decoded, self.scale)
                if score is not None:
                    out.append(score)
        self.last_raw.append(raws)
        return out

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        self.last_raw = []
        results: List[Optional[float]] = []
        for item in items:
            it = normalize_item(item)
            prompt = build_judge_prompt(
                self.rubric, it["reference"], it["candidate"], it["source"]
            )
            got = self._one(prompt)
            results.append(st.mean(got) if got else None)
        return results


class APIJudgeScorer(SemanticScorer):
    """OpenAI 兼容接口的裁判（DeepSeek / 智谱 / OpenAI 都兼容）。

    优点是不占显存 —— 双卡机器上策略已经吃满一张卡，剩下的显存留给自己
    比塞下一个 7B 裁判划算。代价是网络延迟和按量计费。

    并发用线程池：裁判是纯 IO 等待，8~16 个并发就能把延迟藏掉。
    """

    name = "judge_api"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "JUDGE_API_KEY",
        rubric: Optional[Dict[str, Any]] = None,
        samples: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_workers: int = 8,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"没有 API key：环境变量 {api_key_env} 是空的。\n"
                f"  export {api_key_env}=sk-xxxx  （别写进 yaml，会跟着 git 走）"
            )
        self.rubric = rubric or load_rubric()
        self.scale = int(self.rubric.get("scale", 100))
        self.samples = samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.timeout = timeout
        self.retries = retries

    def _call_once(self, prompt: str) -> Optional[float]:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_judge_score(content, self.scale)

    def _judge_one(self, item: Dict[str, Any]) -> Optional[float]:
        it = normalize_item(item)
        prompt = build_judge_prompt(
            self.rubric, it["reference"], it["candidate"], it["source"]
        )
        got: List[float] = []
        for _ in range(self.samples):
            for attempt in range(self.retries + 1):
                try:
                    score = self._call_once(prompt)
                except Exception:
                    score = None
                if score is not None:
                    got.append(score)
                    break
        return st.mean(got) if got else None

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._judge_one, items))


class CachedJudgeScorer(SemanticScorer):
    """读预先算好的裁判分（`scripts/judge_score.py` 的产物）。

    只用于**离线评测**：同一个 checkpoint 换 rubric 重打分、或对比
    SFT 与 RL 的输出。RL 的 rollout 是新生成的文本，没有缓存可用。
    """

    name = "judge_cache"

    def __init__(self, files: Sequence[str]) -> None:
        self.scores: Dict[str, float] = {}
        for f in files:
            p = Path(f)
            if not p.is_absolute():
                p = ROOT / p
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("judge_score") is not None:
                    self.scores[str(r["id"])] = float(r["judge_score"])

    def score_batch(self, items: Sequence[Dict[str, Any]]) -> List[Optional[float]]:
        return [self.scores.get(str(normalize_item(it)["id"])) for it in items]


def build_scorer(cfg: Optional[Dict[str, Any]], override: Optional[str] = None) -> Optional[SemanticScorer]:
    """按配置构造裁判后端。`override` 可以强制指定后端（"api"/"local"/"none"）。"""
    spec = dict(cfg or {})
    backend = override or spec.get("backend", "none")
    if backend in ("none", "", None):
        return None

    rubric = load_rubric(spec.get("rubric_file"))
    common = {
        "rubric": rubric,
        "samples": int(spec.get("samples", 1)),
        "temperature": float(spec.get("temperature", 0.7)),
    }
    if backend == "local":
        model = spec.get("model")
        if not model:
            raise ValueError("semantic.backend=local 需要 semantic.model 指向裁判模型目录")
        return LocalJudgeScorer(
            model_path=model,
            device=spec.get("device", "cuda:0"),
            dtype=spec.get("dtype", "bfloat16"),
            load_in_4bit=bool(spec.get("load_in_4bit", False)),
            max_new_tokens=int(spec.get("max_new_tokens", 512)),
            **common,
        )
    if backend == "api":
        return APIJudgeScorer(
            base_url=spec.get("base_url", ""),
            model=spec.get("api_model") or spec.get("model", ""),
            api_key_env=spec.get("api_key_env", "JUDGE_API_KEY"),
            max_tokens=int(spec.get("max_new_tokens", 512)),
            max_workers=int(spec.get("max_workers", 8)),
            timeout=float(spec.get("timeout", 60.0)),
            retries=int(spec.get("retries", 2)),
            **common,
        )
    if backend == "cache":
        files = spec.get("cache_files") or []
        if not files:
            raise ValueError("semantic.backend=cache 需要 semantic.cache_files")
        return CachedJudgeScorer(files)
    raise ValueError(f"未知的 semantic.backend：{backend}（可选 local / api / cache / none）")

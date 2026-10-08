"""Prompt 模板：把文书原文（+ 检索结果）组装成对话消息。

========================================================
与 schema.py 的分工：
  * schema.py 决定"数据长什么样"（id / source / summary）
  * prompts.py 决定"送给模型什么"（system + user 消息）
  * collator.py 决定"哪些 token 算 loss"（M2 阶段实现）

统一约定：所有构造函数的返回值都是**对话消息列表**，形如
    [{"role": "system", "content": "..."},
     {"role": "user",   "content": "..."}]
这样上层可以直接塞给 models/chat_template.py，与具体模型的模板解耦。
=====================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

# =====================================================================
# 一、系统提示词
# =====================================================================
# 你要写的第一样东西。裁判文书摘要的任务约束建议包含：
#   * 角色设定（例如"你是协助法官整理裁判文书摘要的助手"）
#   * 输出要求（直接给摘要，不要客套话、不要"以下是摘要"这类前缀）
#   * 长度约束（参考摘要平均多少字？先用 tools/make_sample_data.py
#     抽样统计一下真实分布，再定这个数字 —— 别凭感觉写）
#   * 必须覆盖的内容要素（见下面的 RUBRIC）
SYSTEM_PROMPT = (
    "你是一名裁判文书摘要编辑。请直接输出摘要，不要任何前缀、客套话或解释。"
    "摘要应简洁、连贯，只依据文书原文内容，不要编造。"
)

# =====================================================================
# 二、摘要要素清单（民事一审判决书）
# =====================================================================
# 民事一审判决书的典型结构：
#   首部（当事人、案由、审理经过）
#   事实（原告诉称 / 被告辩称 / 经审理查明）
#   理由（本院认为 + 引用的法条）
#   主文（判决结果）
#   尾部（审判员、书记员、日期）
#
# 摘要应当保留哪些要素？这是你要做的核心判断，建议至少覆盖：
#   1. 当事人与案由      —— "谁和谁因为什么打官司"
#   2. 诉讼请求          —— 原告要什么
#   3. 认定的事实        —— 法院查明了什么
#   4. 法律依据          —— 引用了哪部法律的哪些条文
#   5. 裁判结果          —— 判了什么（这是最重要的，不能漏）
#
# 反例（不要写进摘要）：审判员/书记员姓名、开庭日期、送达方式等程序性内容，
# 以及"原告诉称……被告辩称……"这种把双方主张原样搬运的写法。

SUMMARY_RUBRIC = (
    "摘要至少覆盖以下六个要素：\n"
    "1. 案件类型\n"
    "2. 原告的诉讼请求\n"
    "3. 被告的辩称\n"
    "4. 法院审理查明、认定的事实、裁判理由和法院说理\n"
    "5. 法律依据\n"
    "6. 法院的裁判结果\n"
    "其它内容需要你自己判断"
)


# =====================================================================
# 三、三种 prompt 构造方式
# =====================================================================

def build_zeroshot(source: str) -> List[Dict[str, str]]:
    """零样本：只给 SYSTEM_PROMPT + 文书原文。

    这是你的基线。后面所有改进都要和它对比，所以它必须足够"干净"——
    不要偷偷加结构提示，否则 ablation 就没意义了。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]


def build_structured(source: str) -> List[Dict[str, str]]:
    """结构化提示：零样本 + 文书结构说明 + 摘要要素清单。

    对应 configs/data.yaml 里的 prompt_style: structured。
    这是你 2025-03 用过的方法，现在要把它显式、可复现地写进代码。

    提示：文档很长时，光靠"请包含以下要素"未必有效。
    可以考虑在原文里用标记词（如"本院认为""判决如下"）显式标出各段落，
    但这属于要实验验证的假设，不要直接当成既定结论写死。
    """
    
    user_content = (
        f"{SUMMARY_RUBRIC}\n\n"
        f"【文书原文】\n{source}\n\n"
        "请生成摘要："
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_rag(
    source: str, contexts: List[str]
) -> List[Dict[str, str]]:
    """检索增强：零样本 + 检索到的方法条 / 类案摘要。

    对应 configs/data.yaml 里的 prompt_style: rag。

    参数 contexts 是 rag/context.py 传来的检索结果文本列表（M3 阶段接入）。

    这里有两个必须想清楚的问题：
      1. **怎么告知模型这些条文的来源**？
         如果不标注，模型可能把检索到的法条当成案件事实照抄进摘要，
         这是 RAG 幻觉的典型来源。建议给每条上下文编号或注明出处。
      2. **max_context_tokens 怎么用**？
         检索内容会挤占原文的 token 预算，文书本就很长（平均数千字），
         所以必须设上限。超预算时按什么规则截断？按相似度从低到高丢？
         想清楚并在代码里注释说明。

    注意：本阶段（M1）只要让函数能跑；真正的检索质量评测在 M3 做。
    """
    # ---------- 1. 构建带编号的检索上下文 ----------
    # 给每条上下文编号并标注为“检索到的法条/类案”，与案件事实明确区分，
    # 降低模型把法条原文照抄进摘要的幻觉风险。
    def format_contexts(ctx_list: List[str]) -> str:
        if not ctx_list:
            return "（无检索结果）"
        return "\n".join(f"[{i}] {ctx.strip()}" for i, ctx in enumerate(ctx_list, start=1))

    context_text = format_contexts(contexts)

    # ---------- 2. 根据 max_context_tokens 截断检索内容 ----------
    # 说明：这里用字符数粗略近似 token 数（中文场景下 1 字符 ≈ 1 token，
    # 英文单词 ≈ 1.3 token，此处按字符数保守估计）。
    # 截断策略：假设 contexts 按相似度降序排列，前面的更相关，
    # 因此从后往前丢弃（即丢掉相似度最低的），直到满足预算。
    # if max_context_tokens is not None and contexts:
    #     current = list(contexts)
    #     while current:
    #         candidate_text = format_contexts(current)
    #         if len(candidate_text) <= max_context_tokens:
    #             break
    #         current.pop()  # 丢弃最后一个（相似度最低的）
    #     context_text = format_contexts(current)

    # ---------- 3. 构造 user 消息 ----------
    user_content = (
        "请参考以下检索到的法条/类案，结合文书原文生成摘要。\n"
        "注意：检索内容仅作参考，不要照抄。\n\n"
        "【检索到的法条/类案】\n"
        f"{context_text}\n\n"
        "【文书原文】\n"
        f"{source}\n\n"
        "请生成摘要："
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_messages(
    source: str,
    style: str = "structured",
    contexts: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """按 style 分发到上面三个构造函数。

    style 取值与 configs/data.yaml 的 prompt_style 一致：
        "zeroshot" / "structured" / "rag"
    style="rag" 但 contexts 为空时怎么办？想清楚：
    报错、还是退化成 structured？两条路都合理，但要在代码里写明理由，
    因为面试官很可能会问"检索为空时你的系统会怎样"。

    未知 style -> raise ValueError。
    """
    builders = {
        "zeroshot": build_zeroshot,
        "structured": build_structured,
        "rag": None # 需要if分支
    } # 表驱动
    
    
    if style == "rag":
        if not contexts:
            # 这里就是提示里让你想清楚的地方：检索为空时怎么办？
            # 选项 A：报错。选项 B：退化成 structured。
            # 这里我们选择退化成 structured，并在注释里写明理由。
            # 理由：检索系统可能因网络或索引问题返回空结果，此时系统不应崩溃，
            # 退化为纯结构化提示仍能生成摘要，保证系统可用性。
            return build_structured(source)
        return build_rag(source, contexts)
    
    builder = builders.get(style)
    if builder is None:
        raise ValueError(f"未实现的 style: {style}。")
    return builder(source)

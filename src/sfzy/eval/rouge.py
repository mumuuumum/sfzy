"""ROUGE-1/2/L 的自实现版本，严格对齐 CAIL2020 官方口径。

============================ 你要实现的文件 ============================
验收方式：
    python -m pytest tests/test_rouge.py -q

-------------------------------- 官方口径 --------------------------------
三项指标都取 **F-score**，总分：

    总分 = 0.2 * F1(ROUGE-1) + 0.4 * F1(ROUGE-2) + 0.4 * F1(ROUGE-L)

ROUGE-2 与 ROUGE-L 各占 0.4，权重明显高于 ROUGE-1 —— 这意味着
**词序和短语搭配比单个词的覆盖更重要**，是调 prompt 时的重要指引。

-------------------------------- 为什么不用现成包 --------------------------------
  * ROUGE-L 的 LCS 是面试高频考点，实现一遍比调包有价值。
=====================================================================
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

OFFICIAL_WEIGHTS: Dict[str, float] = {
    "rouge-1": 0.2,
    "rouge-2": 0.4,
    "rouge-l": 0.4,
}


def tokenize(text: str, mode: str = "char") -> List[str]:
    """分词。

    mode="char"  ：按字符切，**丢弃空白字符**。中文 ROUGE 的常用做法。
                   "a b c" 与 "abc" 必须得到相同结果（有测试）。
    mode="jieba" ：用 jieba 分词，需 `import jieba` 后 `jieba.lcut`。
                   注意 jieba 首次调用会构建前缀词典，比较慢，
                   可以缓存分词器实例来避免重复初始化。

    其它 mode 取值 -> raise ValueError（有测试）。

    提示：jieba 在函数内部 import，这样 char 模式不依赖 jieba 也能用。
    """
    if mode=="char":
        return [c for c in text if not c.isspace()]
    elif mode == "jieba":
        import jieba
        return jieba.lcut(text)
    else:
        raise ValueError(f"不支持mode：{mode}")


def rouge_n(
    hypothesis_tokens: Sequence[str], reference_tokens: Sequence[str], n: int
) -> Tuple[float, float, float]:
    """计算 ROUGE-N 的 (precision, recall, f1)。

    公式：
        overlap  = 假设与参考**共有**的 n-gram 个数
        precision = overlap / 假设的 n-gram 总数
        recall    = overlap / 参考的 n-gram 总数
        f1        = 2 * P * R / (P + R)

    两个关键细节：
      1. 重复 n-gram 要按**较小出现次数**计数。
         例：假设有 3 个 "a"，参考只有 1 个，则只算 1 个重叠。
         提示：collections.Counter 的 `&` 运算就是这个语义，不用自己写循环。
      2. 分母为 0 时（空串、或 n 大于序列长度）返回 0.0，
         **不要抛 ZeroDivisionError**，也不要返回 nan（有测试）。

    返回值一律是 float，即使全是 0。
    """
    if n <= 0:
        return 0.0, 0.0, 0.0

    def get_ngrams(tokens: Sequence[str]) -> list[tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    hyp_ngrams = get_ngrams(hypothesis_tokens)
    ref_ngrams = get_ngrams(reference_tokens)

    from collections import Counter
    hyp_counter = Counter(hyp_ngrams)  # a dict maps each ngram to num
    ref_counter = Counter(ref_ngrams)

    # Counter 的 & 运算：每个元素取两个计数中的较小值，正好是重叠语义
    overlap = sum((hyp_counter & ref_counter).values())
    # hyp_counter & ref_counter：得到一个全新的 Counter，里面存着每个重叠 n-gram 及其最小重叠次数。
    # .values()：提取出这些重叠次数（比如 [2, 1]）。
    # sum(...)：把它们加起来，得到总的重叠 n-gram 数量（比如 3）。

    hyp_total = len(hyp_ngrams)
    ref_total = len(ref_ngrams)

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def rouge_l(
    hypothesis_tokens: Sequence[str], reference_tokens: Sequence[str]
) -> Tuple[float, float, float]:
    """计算 ROUGE-L 的 (precision, recall, f1)，基于最长公共子序列（LCS）。

        recall    = LCS长度 / 参考长度
        precision = LCS长度 / 假设长度
        f1        = 2 * P * R / (P + R)

    注意 LCS 与"最长公共子串"的区别：**不要求连续**。
    "abc" 与 "acb" 的 LCS 是 "ab"，长度为 2。

    实现提示：
      * 标准 DP 是 O(n*m) 时间、O(n*m) 空间。
      * 空间可以优化到 O(min(n,m))：只保留 DP 表的前一行和当前行。
        这是很值得在面试里讲的优化点 —— 摘要动辄几百字，
        空间优化能实打实减少内存占用。
      * 同样要注意空输入的分母为 0 情况（有测试）。
    """
    def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
        # 让 b 成为较短的序列，从而将空间复杂度降到 O(min(len(a), len(b)))
        
        # dp[i][j] = dp[i−1][j−1]+1     如果 a[i−1]==b[j−1]
        # dp[i][j] = max(dp[i−1][j],dp[i][j−1])    如果 a[i−1]!=b[j−1]
        # 只用到了上一行(i-1)和当前行(i)

        if len(a) < len(b):
            a, b = b, a
        m, n = len(a), len(b)

        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            ai = a[i - 1]
            for j in range(1, n + 1):
                if ai == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    hyp_len = len(hypothesis_tokens)
    ref_len = len(reference_tokens)

    lcs = lcs_length(hypothesis_tokens, reference_tokens)

    precision = lcs / hyp_len if hyp_len > 0 else 0.0
    recall = lcs / ref_len if ref_len > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def _f1(precision: float, recall: float) -> float:
    """F1 = 2PR/(P+R)，P+R 为 0 时返回 0.0。

    这个辅助函数已经给你了，rouge_n / rouge_l / score_pair 都可以用。
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_pair(
    hypothesis: str, reference: str, mode: str = "char"
) -> Dict[str, float]:
    """单条样本的完整打分。

    返回的字典必须包含这 10 个键（测试逐个检查）：
        rouge-1-p, rouge-1-r, rouge-1-f
        rouge-2-p, rouge-2-r, rouge-2-f
        rouge-l-p, rouge-l-r, rouge-l-f
        overall

    其中 overall = 0.2*rouge-1-f + 0.4*rouge-2-f + 0.4*rouge-l-f
    （用 OFFICIAL_WEIGHTS，不要硬编码数字）

    提示：键名用 f"rouge-{n}-{k}" 这样的格式统一生成，避免手写出错。
    """
    hyp_token = tokenize(hypothesis,mode)
    ref_token = tokenize(reference,mode)
    
    res = {}
    res["overall"] = 0
    for i in range(1,3):
        res[f"rouge-{i}-p"],res[f"rouge-{i}-r"],res[f"rouge-{i}-f"] = rouge_n(hyp_token,ref_token,i)
        res["overall"] += OFFICIAL_WEIGHTS[f"rouge-{i}"] * res[f"rouge-{i}-f"]
    res["rouge-l-p"],res["rouge-l-r"],res["rouge-l-f"] = rouge_l(hyp_token,ref_token)
    res["overall"] += OFFICIAL_WEIGHTS["rouge-l"] * res["rouge-l-f"]
    return res    


def score_corpus(
    predictions: Dict[str, str], references: Dict[str, str], mode: str = "char"
) -> Dict[str, float]:
    """整个测试集的平均分。

    * 只评测 id 同时存在于 predictions 与 references 里的样本（有测试）。
    没有任何共同 id 时：num_samples 为 0，各指标与 overall 都是 0.0，
    **不能抛异常**。
    * 各指标取**宏平均**（每条样本先算分，再对所有样本取平均），
      这也是官方评测的做法。
    * 返回：score_pair 的 10 个键 + num_samples（int）。

    提示：宏平均是对每条样本的分数取平均，不是把所有人的预测拼起来算一次。
    两者数值会不一样，写报告时要说明用的是哪种。
    """
    # 只取两个字典中都存在的 id
    common_ids = set(predictions.keys()) & set(references.keys())
    num_samples = len(common_ids)

    # 没有共同 id 时，通过空输入获取指标键名，然后全部置 0
    if num_samples == 0:
        empty_scores = score_pair("", "", mode=mode)
        result = {k: 0.0 for k in empty_scores}
        result["num_samples"] = 0
        return result

    # 累加每条样本的分数
    totals: Dict[str, float] = {}
    for doc_id in common_ids:
        scores = score_pair(predictions[doc_id], references[doc_id], mode=mode)
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value

    # 宏平均：总和除以样本数
    result = {key: total / num_samples for key, total in totals.items()}
    result["num_samples"] = num_samples
    return result

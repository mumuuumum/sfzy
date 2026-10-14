"""裁判文书摘要的**纯规则**三要素切分器（无模型、无依赖）。

切分对象是摘要正文，产出三个字符区间：

    plaintiff_claims   原告诉讼请求
    defendant_defenses 被告辩称（可缺失）
    court_facts        法院认定事实及说理

摘要通常按"案件背景 → 原告诉请 → 被告辩称（可选）→ 法院事实与说理 →
法律依据及裁判结果"组织。本模块只做前面四个环节的切分：法律依据与裁判结果
是**尾部标志**，用来定 `court_facts` 的右边界，本身不产出区间。

============================ 三条设计原则 ============================

1. **强特征优先**。每条边界先找强标志（"被告辩称""经审理查明""依照《……》判决"），
   找不到才逐级降级到弱标志和句式推断。弱标志永远不会覆盖强标志定下的边界。

2. **顺序校验**。标志命中了不代表它就在那个位置：说理里也会写"被告辩称"，
   说理里也会出现"驳回"。所以任何标志都要先过一遍文本顺序（答辩必须早于法院
   叙述、诉请必须早于答辩），过不了就记一条 note 并按"该要素不存在"处理。
   这也是需求②③里点名的两个坑。

3. **字符级边界**。多个要素可以出现在同一句里（"原告请求判令被告支付租金，
   被告辩称已付清，本院认为……"），所以边界一律是字符偏移量，不按句子切。
   句子只用来做"弱特征补充"和推断。

============================ 缺失 ≠ 不确定 ============================
每个要素的状态只有三种：

  * `identified` —— 起止都锚上了（强/中置信度）；
  * `absent`     —— **确认缺失**：全文没有任何答辩标志（或标志被顺序校验否掉），
                    因此确实不存在被告辩称段落；
  * `uncertain`  —— **规则无法确定边界**：有迹象但定不下来（例如答辩标志找到了，
                    可整段既没有法院叙述标志、也没有可用的句边界推断）。

`uncertain` 绝不悄悄降级成 `absent`——需求明确要求区分这两种情况，因为下游
（无论人工检查还是将来的 reward）对"确实没有"和"我读不准"的处理方式不同。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sfzy.segment.patterns import (
    BACKGROUND_RULES,
    CLAIM_VERBS,
    COURT_TONE_STRONG,
    COURT_TONE_WEAK,
    D_ABSENT_ONLY,
    D_ACT_VERB_SUBJECT_BLOCK,
    D_BARE_BACKSTOP_ACT,
    D_DEFENSE_ACT_VERBS,
    D_DEFENSE_VERBS,
    D_ENTITY_SUFFIXES,
    D_FORBIDDEN_SUBJECTS,
    D_MARKER_RULES,
    D_NEGATION,
    D_QUANTIFIER_PREFIX,
    D_ROLE_WORDS,
    D_SUBJECT_CHARS,
    D_SUBJECT_TAILS,
    F_MARKER_RULES,
    HARD_SENT_DELIMS,
    P_MARKER_RULES,
    REASONING_TONE,
    RESULT_MARKER,
    SOFT_SENT_DELIMS,
    TAIL_AFTERMATH,
    TAIL_RULES,
    TRAILING_CLOSERS,
    MarkerRule,
    Strength,
    gap_is_citation_like,
)

#: 三要素的固定顺序。区间必须按这个顺序排列且互不重叠。
ELEMENTS: Tuple[str, ...] = ("plaintiff_claims", "defendant_defenses", "court_facts")

#: 中文名，报告里用它，避免满屏英文键名。
ELEMENT_NAMES: Dict[str, str] = {
    "plaintiff_claims": "原告诉请",
    "defendant_defenses": "被告辩称",
    "court_facts": "法院事实与说理",
}


# ---------------------------------------------------------------- 状态与数据模型


class Status(str, Enum):
    """要素状态。"""

    IDENTIFIED = "identified"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Confidence(str, Enum):
    """边界置信度：high=强标志；medium=弱标志或可靠推断；low=只能给个近似位置。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_CONF_ORDER = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}

#: 标志强度和置信度的对应关系。弱标志只能给 low，这是"强特征优先"的量化版本。
_STRENGTH_CONF = {
    Strength.STRONG: Confidence.HIGH,
    Strength.MEDIUM: Confidence.MEDIUM,
    Strength.WEAK: Confidence.LOW,
}


def _weaker(a: Confidence, b: Confidence) -> Confidence:
    """两个边界取更弱的一个——一个要素的置信度不会高于它较弱的那个边界。"""

    return a if _CONF_ORDER[a] >= _CONF_ORDER[b] else b


@dataclass(frozen=True)
class Sentence:
    """一句话。`start`/`end` 是原文里的字符偏移（左闭右开）。"""

    start: int
    end: int
    hard: bool  # True=以句号/叹号/问号结尾；False=以分号结尾

    def text_of(self, src: str) -> str:
        return src[self.start : self.end]


@dataclass
class ElementSpan:
    """一个要素的切分结果。"""

    name: str
    start: Optional[int]
    end: Optional[int]
    status: Status
    #: 起点/终点分别由哪条规则定的（写进报告，便于复盘"这一刀是谁下的"）
    start_rule: str = ""
    end_rule: str = ""
    confidence: Confidence = Confidence.LOW
    notes: List[str] = field(default_factory=list)

    @property
    def uncertain(self) -> bool:
        return self.status is Status.UNCERTAIN

    def text_of(self, src: str) -> str:
        if self.start is None or self.end is None:
            return ""
        return src[max(0, self.start) : max(0, self.end)]

    def to_dict(self, src: Optional[str] = None) -> Dict[str, object]:
        out: Dict[str, object] = {
            "element": self.name,
            "status": self.status.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence.value,
            "start_rule": self.start_rule,
            "end_rule": self.end_rule,
            "notes": list(self.notes),
        }
        if src is not None:
            out["text"] = self.text_of(src)
        return out


@dataclass
class SegmentationResult:
    """一条摘要的切分结果。"""

    summary: str
    spans: Dict[str, ElementSpan]
    order_ok: bool = True
    notes: List[str] = field(default_factory=list)

    # ---------------- 便捷访问 ----------------
    def get(self, name: str) -> ElementSpan:
        return self.spans[name]

    def text_of(self, name: str) -> str:
        return self.spans[name].text_of(self.summary)

    @property
    def uncertain_elements(self) -> List[str]:
        return [k for k in ELEMENTS if self.spans[k].uncertain]

    @property
    def complete(self) -> bool:
        """完整切分：三要素（含"确认缺失"）都定了下来，且顺序合法。

        注意"完整"不要求被告辩称存在——它本来就允许缺失。要求的是**它的状态
        是确定的**（identified 或 absent），而不是"我猜不出来"。
        """

        if not self.order_ok:
            return False
        p, d, f = (self.spans[k] for k in ELEMENTS)
        if p.status is not Status.IDENTIFIED or f.status is not Status.IDENTIFIED:
            return False
        return d.status in (Status.IDENTIFIED, Status.ABSENT)

    def covered_chars(self) -> int:
        """三要素覆盖的字符数（用于检查有没有整段文字掉在三个区间之外）。"""

        total = 0
        for k in ELEMENTS:
            s = self.spans[k]
            if s.status is Status.IDENTIFIED and s.start is not None and s.end is not None:
                total += max(0, s.end - s.start)
        return total

    def marked_text(self) -> str:
        """把三要素区间标出来，供人工检查。

        `【P】…【/P】【D】…【/D】【F】…【/F】`；`absent` 标成 `【D:缺失】`，
        `uncertain` 标成 `【F:不确定】`，这样一眼能看出"没切出来"的到底是哪种。
        """

        marks: List[Tuple[int, str]] = []
        for key, tag in zip(ELEMENTS, ("P", "D", "F")):
            span = self.spans[key]
            if span.status is Status.ABSENT:
                marks.append((span.start if span.start is not None else 0, f"【{tag}:缺失】"))
                continue
            if span.start is None or span.end is None:
                marks.append((0, f"【{tag}:不确定】"))
                continue
            label = tag if span.status is Status.IDENTIFIED else f"{tag}:不确定"
            marks.append((span.start, f"【{label}】"))
            marks.append((span.end, f"【/{tag}】"))
        out = self.summary
        for pos, mark in sorted(marks, key=lambda x: -x[0]):
            out = out[:pos] + mark + out[pos:]
        return out

    def to_dict(self, with_text: bool = True) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "order_ok": self.order_ok,
            "complete": self.complete,
            "uncertain_elements": self.uncertain_elements,
            "spans": {
                k: self.spans[k].to_dict(self.summary if with_text else None) for k in ELEMENTS
            },
            "notes": list(self.notes),
        }


def empty_result(summary: str) -> SegmentationResult:
    """空/全空白输入：三个要素都标 uncertain，不猜。"""

    spans = {
        k: ElementSpan(
            name=k,
            start=0,
            end=0,
            status=Status.UNCERTAIN,
            start_rule="空文本",
            end_rule="空文本",
            confidence=Confidence.LOW,
            notes=["输入为空或全空白，无法切分"],
        )
        for k in ELEMENTS
    }
    return SegmentationResult(summary=summary, spans=spans, order_ok=True, notes=["空文本"])


# ---------------------------------------------------------------- 句子切分


def split_sentences(text: str) -> List[Sentence]:
    """按标点切句，返回带偏移量的句子列表。

    只做"切"，不做任何清洗：`"".join(s.text_of(text)) == text` 恒成立
    （保证偏移量可用）。句尾的引号/括号收到上一句里，避免切出孤立符号。
    """

    if not text:
        return []
    out: List[Sentence] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in HARD_SENT_DELIMS or ch in SOFT_SENT_DELIMS:
            j = i + 1
            while j < n and text[j] in TRAILING_CLOSERS:
                j += 1
            out.append(Sentence(start, j, hard=ch in HARD_SENT_DELIMS))
            start = j
            i = j
        else:
            i += 1
    if start < n:
        out.append(Sentence(start, n, hard=False))
    return out


# ---------------------------------------------------------------- 标志命中


@dataclass(frozen=True)
class Hit:
    """一次标志命中。"""

    start: int
    end: int
    rule: str
    strength: Strength
    text: str

    @property
    def confidence(self) -> Confidence:
        return _STRENGTH_CONF[self.strength]


def _hits(
    rules: Sequence[MarkerRule],
    src: str,
    *,
    after: int = 0,
    before: Optional[int] = None,
    strengths: Optional[Iterable[Strength]] = None,
) -> List[Hit]:
    """收集标志命中，按位置排序。

    `after`/`before` 限定搜索区间（左闭右开），用于"只找答辩起点之后的法院叙述
    标志"这类顺序约束；`strengths` 过滤强度档。同一位置命中多条规则时，保留
    词表里靠前（更精确）的那条。
    """

    want = set(strengths) if strengths is not None else None
    found: List[Hit] = []
    for rule in rules:
        if want is not None and rule.strength not in want:
            continue
        for m in rule.pattern.finditer(src):
            if m.start() < after:
                continue
            if before is not None and m.start() >= before:
                break
            if m.end() == m.start():
                continue
            found.append(
                Hit(m.start(), m.end(), rule.name, rule.strength, src[m.start() : m.end()])
            )
    # 位置升序；**同位置按强度升序**（strength 强的 _CONF_ORDER 小，排在前面）——
    # 弱的不能顶掉强的：同一处既命中"被告辩称"（强）又命中"具名当事人辩称"（弱）时，
    # 规则名和置信度都要记强的那条。
    found.sort(key=lambda h: (h.start, _CONF_ORDER[h.confidence]))
    deduped: List[Hit] = []
    seen: set[Tuple[int, int]] = set()
    for h in found:
        if (h.start, h.end) in seen:
            continue
        seen.add((h.start, h.end))
        deduped.append(h)
    return deduped


# ---------------------------------------------------------------- 切分器

#: 动词锚定的弱答辩标志扫描：从主语起最多回看多少字符。
_D_SUBJECT_LOOKBACK = 22
#: 回看"这是个被告吗"的最大距离。比主语回看宽：句子里常夹着
#: "被告华美达公司、李朗华无应诉，未答辩"、"被告费某2经合法传唤未到庭应诉"这类描述。
_D_ROLE_LOOKBACK = 40

#: 一次扫完的"候选词"：答辩类动词 + "未到庭/缺席"这类只表示没到庭的说法。
_D_SCAN_RE = re.compile(
    "|".join(
        map(
            re.escape,
            D_DEFENSE_VERBS + D_DEFENSE_ACT_VERBS + ("缺席", "未到庭", "未出庭", "未应诉"),
        )
    )
)
_D_DEFENSE_ONLY = frozenset(D_DEFENSE_VERBS)
_D_ACT_VERBS = frozenset(D_DEFENSE_ACT_VERBS)
#: 只表示"人没到庭"、自身不含"答辩"二字的说法。它们必须和角色主体同现才算缺失声明。
_D_ABSENT_ONLY_WORDS = frozenset(("缺席", "未到庭", "未出庭", "未应诉"))


def _weak_defense_hits(src: str, *, after: int = 0, before: Optional[int] = None) -> List[Hit]:
    """扫"弱"答辩标志：未答辩声明 / 具名当事人 / 并列主体 / 角色主体 / 光名主体。

    做法是**动词锚定**：先线性扫出所有"答辩/辩称/抗辩/答辩意见/辨称"，再对每个动词
    往前回看最多 22 个字符，判断主语长什么样，据此定规则名与强度。不这么做的话，
    "主语 + 动词"的大正则需要枚举主语长度 × 后缀词表，在几百字的摘要上会慢一个量级。

    主语回看的分支（按优先级）：

    * 动词前面是"未/没有/无/不" → **未答辩声明**（需求②点名的起点标志），
      主语可以是被告、具名当事人，也可以是"张玉山、孙淑清"这样的光名字列表；
    * 主语带企业/机构后缀 → **具名当事人辩称**；
    * 主语是"保证人/第三人/上诉人"等角色词 → **角色主体辩称**；
    * 主语里带顿号/与/及 → **并列主体共同辩称**（多个被告连续答辩的常见写法）；
    * 其余 2~4 字的光名字 → **光名主体辩称**；
    * 主语是"本院/法院/原告/证人"等不可能答辩的主体 → **弃掉**。

    所有分支都只是候选：最终采不采用由 `_resolve_d_start` 的位置与顺序校验决定。
    """

    out: List[Hit] = []
    stop = len(src) if before is None else min(before, len(src))
    for vm in _D_SCAN_RE.finditer(src, after, stop):
        v_start, v_end = vm.start(), vm.end()
        is_defense_word = vm.group() in _D_DEFENSE_ONLY
        # "被告的抗辩""答辩意见的效力"——动词后面紧跟"的"时是名词性说法，不是段落起点
        if is_defense_word and src.startswith("的", v_end):
            continue
        # 往前回看主语：先跳过空白（模型输出里可能出现"被告 \n 辩称"）
        i = v_start
        while i > after and src[i - 1].isspace():
            i -= 1
        neg = None
        head = i
        # 情况一：动词前是否定词（"未答辩""没有答辩""未出庭答辩"）
        j = i
        while j > after and D_SUBJECT_CHARS.match(src[j - 1]):
            j -= 1
        window = src[j:i]
        m = D_NEGATION.search(window) or D_ABSENT_ONLY.search(window)
        if m:
            neg = m
            head = j + m.start()
        elif vm.group() in _D_ABSENT_ONLY_WORDS:
            # "缺席/未到庭"这类词本身不是答辩动词，只有和"被告"一起才说明没答辩
            subject_text = src[max(after, v_start - _D_ROLE_LOOKBACK) : v_start]
            role = _last_role_word(subject_text)
            if role is None:
                continue
            # "二被告未出庭应诉"：数量词前缀（二/两/四…）属于主语，起点要含进去
            back = role[0]
            while back > 0 and D_QUANTIFIER_PREFIX.match(subject_text[back - 1 : back]):
                back -= 1
            start = max(after, v_start - _D_ROLE_LOOKBACK) + back
            out.append(
                Hit(start, v_end, "被告辩称:未答辩声明", Strength.MEDIUM, src[start:v_end])
            )
            continue
        # 主语本体：从 head 往前收"名字字符"（含并列连接符），并剥掉尾部修饰词
        k = head
        while k > after and src[k - 1].isspace():
            k -= 1
        subj_end = k
        subj_start = subj_end
        while subj_start > max(after, subj_end - _D_SUBJECT_LOOKBACK):
            ch = src[subj_start - 1]
            if D_SUBJECT_CHARS.match(ch):
                subj_start -= 1
            else:
                break
        subject = src[subj_start:subj_end]
        for tail in D_SUBJECT_TAILS:
            if subject.endswith(tail):
                subject = subject[: -len(tail)]
                subj_end -= len(tail)
                break

        if not subject:
            # 主语被标点切断（"…无应诉，未答辩"），退回"往前找最后一个角色主体"
            subject_text = src[max(after, v_start - _D_ROLE_LOOKBACK) : v_start]
            role = _last_role_word(subject_text)
            if role is None:
                continue
            subj_start = max(after, v_start - _D_ROLE_LOOKBACK) + role[0]
            subject = subject_text[role[0] :]
            subj_end = v_start
        if any(bad in subject[-6:] for bad in D_FORBIDDEN_SUBJECTS):
            continue

        if neg is not None:
            # "被告经合法传唤无正当理由未到庭"这类：主语回看里混进了叙述，起点回退到主体。
            # 但"两/二/三/…/各/众被告"这种**数量词前缀**是主体的一部分，回退会
            # 把切点切进词中间（"…违约责任。两|被告均未答辩"），所以不打算剥。
            role = _last_role_word(subject)
            if role is not None and not D_QUANTIFIER_PREFIX.match(subject[: role[0]]):
                subj_start += role[0]
                subject = subject[role[0] :]
            rule, strength = "被告辩称:未答辩声明", Strength.MEDIUM
        elif vm.group() in _D_ACT_VERBS:
            # "被告承认…""被告表示…""被告不同意…""被告无异议"——诉状/答辩里常见的陈述
            # 动作词。主语必须**像当事人**：否则"双方当事人真实意思表示"里的"表示"、
            # "各方无异议"里的"无异议"都会被当成答辩段落起点。
            party_like = (
                any(subject.endswith(r) for r in D_ROLE_WORDS)
                or any(subject.endswith(s) for s in D_ENTITY_SUFFIXES)
            )
            if not party_like:
                if any(bad in subject for bad in D_ACT_VERB_SUBJECT_BLOCK):
                    continue
                if not (2 <= len(subject) <= 4):
                    continue
            rule, strength = "被告辩称:陈述动作词", Strength.WEAK
        elif any(subject.endswith(s) for s in D_ENTITY_SUFFIXES) and len(subject) > 2:
            rule, strength = "被告辩称:具名当事人辩称", Strength.WEAK
        elif any(subject.endswith(r) for r in D_ROLE_WORDS):
            rule, strength = "被告辩称:角色主体辩称", Strength.WEAK
        elif any(sep in subject for sep in ("、", "与", "及")):
            rule, strength = "被告辩称:并列主体共同辩称", Strength.WEAK
        elif 2 <= len(subject) <= 4:
            rule, strength = "被告辩称:光名主体辩称", Strength.WEAK
        else:
            continue
        out.append(Hit(subj_start, v_end, rule, strength, src[subj_start:v_end]))
    return out


def _last_role_word(text: str) -> Optional[Tuple[int, str]]:
    """在 `text` 里找**最后**出现的角色主体（"被告/保证人/第三人"……），返回 (位置, 词)。"""

    best: Optional[Tuple[int, str]] = None
    for word in D_ROLE_WORDS:
        idx = text.rfind(word)
        if idx >= 0 and (best is None or idx > best[0]):
            best = (idx, word)
    return best


class Segmenter:
    """纯规则切分器。线程安全、无状态，可以随便复用/并发调用。"""

    #: 尾部候选只在这些位置之后才考虑——尾部一定是全文最后的段落，否则说理里
    #: 的"依据《……》驳回"会被当成尾部（需求③点名的第二个坑）。
    tail_lookback_sentences = 4
    tail_lookback_ratio = 0.45

    def __init__(
        self,
        *,
        save_notes: bool = True,
        prefer_bare_defendant: bool = False,
        bare_defendant_backstop: bool = True,
    ) -> None:
        self.save_notes = save_notes
        #: 兜底：所有答辩标志都落空时，用"字面 `。被告` + 紧跟答辩动作词"的位置当起点。
        #: 实测只在 82 条摘要上生效（占 0.8%），其中约六成是真答辩段落、四成是以被告
        #: 为主语的法院说理；命中时来源记成 `被告辩称:句首被告兜底`（可单独统计/关掉）。
        self.bare_defendant_backstop = bare_defendant_backstop
        #: 打开后走"用户规则"：**文本里第一个句首"被告"就是答辩起点**。
        #: 默认关闭——在 train 摘要上实测它是亏的（见 docs/segmenter.md 的对照表）：
        #:   * 61.9% 与现有规则一致；
        #:   * 5.8% 会把诉请列表的条目（"。被告承担诉讼费。"）当成答辩起点，切短诉请；
        #:   * 2.6% 会把起点推到"被告XX辩称"之后（具名被告先答辩的那种）；
        #:   * 剩下的增量里多数是"以被告为主语的法院说理"。
        #: 想按这个直觉跑一版对比就把它打开（`--d-bare-first`）。
        self.prefer_bare_defendant = prefer_bare_defendant

    # ------------------------------------------------ 对外入口

    def segment(self, text: str) -> SegmentationResult:
        src = text or ""
        if not src.strip():
            return empty_result(src)

        sents = split_sentences(src)
        notes: List[str] = []

        # ---- 标志扫描（只扫一遍，后面都在这些结果上做顺序校验）
        p_hits = _hits(P_MARKER_RULES, src)
        # 答辩标志 = 强/中档正则（"被告辩称"等）+ 动词锚定的弱标志扫描
        d_hits_all = sorted(
            list(_hits(D_MARKER_RULES, src)) + _weak_defense_hits(src),
            key=lambda h: (h.start, _CONF_ORDER[h.confidence]),
        )
        f_strong = _hits(F_MARKER_RULES, src, strengths=(Strength.STRONG,))
        f_medium = _hits(F_MARKER_RULES, src, strengths=(Strength.MEDIUM,))
        f_weak = _hits(F_MARKER_RULES, src, strengths=(Strength.WEAK,))
        tail = self._find_tail(src, sents)

        # ---- ① 原告诉请起点
        p_start, p_start_rule, p_start_conf = self._resolve_p_start(src, sents, p_hits)

        # ---- ② 被告答辩起点（带顺序校验）
        d_hit = self._resolve_d_start(src, d_hits_all, p_start, f_strong, f_medium, notes)
        if d_hit is None and self.bare_defendant_backstop:
            d_hit = self._bare_defendant_backstop(src, p_start, f_strong, f_medium, notes)
        if self.prefer_bare_defendant:
            d_hit = self._apply_bare_defendant_rule(
                src, sents, p_start, d_hit, f_strong, f_medium, notes
            )

        # ---- ③ 法院叙述锚点：答辩终点 / 无答辩时的诉请终点
        # 注意这里必须**先合并再按位置排序**：强/中两档分别有序，直接拼接会
        # 让"位置更晚的强标志"盖掉"位置更早的中标志"（例如说明段里的
        # "详见本院认为部分"会顶掉前面的"经查明"）。
        f_anchors = sorted(
            list(f_strong) + list(f_medium),
            key=lambda h: (h.start, _CONF_ORDER[h.confidence]),
        )
        f_anchor = f_anchors[0] if f_anchors else None
        if f_anchor is not None and d_hit is not None and f_anchor.start <= d_hit.start:
            notes.append(
                f"法院叙述标志({f_anchor.rule})出现在答辩标志({d_hit.rule})之前，"
                "答辩标志按说理回述处理"
            )
            d_hit = None

        tail_start = tail.start if tail is not None else None

        # ---- 组装区间
        f_status = Status.IDENTIFIED
        p_status = Status.IDENTIFIED
        p_extra: List[str] = []
        if d_hit is not None:
            p_end, p_end_rule, p_end_conf = (
                d_hit.start,
                f"被告答辩起点({d_hit.rule})",
                d_hit.confidence,
            )
            no_answer = "未答辩" in d_hit.rule
            weak_anchor = None
            # 弱标志（"本案争议焦点："）只能补位：必须晚于答辩起点（否则它只是
            # 案由段的"焦点"提法），还必须是**句首**——法院归纳争议焦点是独立
            # 一句，句中的"争议焦点"很可能仍在转述当事人的说法。
            after_d = [h for h in f_weak if h.start > d_hit.start]
            if after_d and _is_sentence_start(sents, after_d[0].start):
                if f_anchor is None or after_d[0].start < f_anchor.start:
                    weak_anchor = after_d[0]
            if weak_anchor is not None:
                # 弱标志更早且另起一句时优先："本案争议焦点为……"是法院在归纳焦点，
                # 属于法院事实；把它留在答辩里会把答辩段撑长一整句。
                d_end, d_end_rule, d_end_conf = (
                    weak_anchor.start,
                    f"弱法院叙述标志({weak_anchor.rule})",
                    weak_anchor.confidence,
                )
                d_extra = []
            elif f_anchor is not None:
                d_end, d_end_rule, d_end_conf = (
                    f_anchor.start,
                    f"法院叙述标志({f_anchor.rule})",
                    f_anchor.confidence,
                )
                d_extra: List[str] = []
            elif no_answer:
                # "被告未答辩。"这类**声明式**答辩段落：它本身就是一句完整的话
                # （"未答辩/未到庭/未提交答辩状"说完了就完了），所以按句边界收尾
                # 不算"强行切分"——这是结构化证据，不是猜测。
                d_end = _sentence_end_after(sents, d_hit.start)
                d_end_rule = "未答辩声明句边界"
                d_end_conf = Confidence.MEDIUM
                d_extra = ["仅声明未答辩，无实体答辩内容"]
            else:
                d_end, d_end_rule, d_end_conf, d_extra = self._infer_d_end(
                    src, sents, d_hit.start, tail_start
                )
            d_unresolved = d_end_rule.startswith("推断失败")
            if d_unresolved:
                # 答辩终点定不下来 → 答辩和法院事实两个要素都只能记"不确定"，
                # 不能把答辩终点硬塞到尾部起点上假装切完（需求③）。
                d_status = Status.UNCERTAIN
                f_status = Status.UNCERTAIN
            else:
                d_status = Status.IDENTIFIED
                f_status = Status.IDENTIFIED
            d_span = ElementSpan(
                name="defendant_defenses",
                start=d_hit.start,
                end=d_end,
                status=d_status,
                start_rule=f"{d_hit.rule}",
                end_rule=d_end_rule,
                confidence=_weaker(d_hit.confidence, d_end_conf),
                notes=list(d_extra),
            )
            f_start, f_start_rule, f_start_conf = d_end, f"继承答辩终点({d_end_rule})", d_end_conf
            f_extra: List[str] = []
        else:
            d_end = None
            d_span = ElementSpan(
                name="defendant_defenses",
                start=None,
                end=None,
                status=Status.ABSENT,
                start_rule="无答辩标志",
                end_rule="无答辩标志",
                confidence=Confidence.MEDIUM,
                notes=[],
            )
            if f_anchor is not None:
                f_start, f_start_rule, f_start_conf = (
                    f_anchor.start,
                    f"法院叙述标志({f_anchor.rule})",
                    f_anchor.confidence,
                )
                f_extra: List[str] = []
            else:
                f_start, f_start_rule, f_start_conf, f_extra = self._infer_f_start(
                    src, sents, p_start, tail_start
                )
                if f_start_rule.startswith("推断失败"):
                    f_status = Status.UNCERTAIN
                    # 没有答辩时诉请的终点**就是**法院事实的起点，起点定不下来，
                    # 诉请的右边界同样定不下来——不能只把法院事实记成不确定。
                    p_status = Status.UNCERTAIN
                    p_extra = ["法院事实起点无法确定，诉请终点（=法院事实起点）同样不确定"]
                elif f_start_rule.startswith("推断"):
                    # 诉请里的编号列表常常跨句（"……；4、原告无需支付赔偿金。"），
                    # 推断出的法院事实起点如果正落在编号项中间，说明这一项还是诉请。
                    advanced = _skip_list_items(src, sents, f_start)
                    if advanced != f_start:
                        f_extra.append(
                            f"推断出的法院事实起点落在编号列表中途，并入该编号项（{f_start}→{advanced}）"
                        )
                        f_start = advanced
            p_end, p_end_rule, p_end_conf = (
                f_start,
                "无答辩，诉请终点=法院事实起点",
                f_start_conf,
            )

        # ---- ④ 法院事实终点 = 尾部起点
        if tail is not None:
            f_end, f_end_rule, f_end_conf = tail.start, f"尾部标志({tail.rule})", tail.confidence
        else:
            f_end, f_end_rule, f_end_conf = (
                len(src),
                "未找到尾部标志，取文末",
                Confidence.LOW,
            )
            notes.append("未找到'法律依据+裁判结果'尾部标志，法院事实终点不可确定")

        # ---- 装配
        p_span = ElementSpan(
            name="plaintiff_claims",
            start=p_start,
            end=p_end,
            status=p_status,
            start_rule=p_start_rule,
            end_rule=p_end_rule,
            confidence=_weaker(p_start_conf, p_end_conf),
            notes=list(p_extra),
        )
        f_span = ElementSpan(
            name="court_facts",
            start=f_start,
            end=f_end,
            status=f_status,
            start_rule=f_start_rule,
            end_rule=f_end_rule,
            confidence=_weaker(f_start_conf, f_end_conf),
            notes=list(f_extra),
        )
        spans = {
            "plaintiff_claims": p_span,
            "defendant_defenses": d_span,
            "court_facts": f_span,
        }

        # ---- 顺序/合法性校验：不确定的降级，非法的记 note 并修
        self._enforce_order(src, spans, notes)
        self._trim_edge_punctuation(src, spans)
        order_ok = self._check_order(spans)
        if not self.save_notes:
            notes = []
        return SegmentationResult(summary=src, spans=spans, order_ok=order_ok, notes=notes)

    def segment_many(self, texts: Iterable[str]) -> List[SegmentationResult]:
        return [self.segment(t) for t in texts]

    # ------------------------------------------------ ① 诉请起点

    def _resolve_p_start(
        self,
        src: str,
        sents: Sequence[Sentence],
        p_hits: Sequence[Hit],
    ) -> Tuple[int, str, Confidence]:
        """需求①：默认第二句；第一句有标志就从标志处开始；第一句是案情背景就跳过。"""

        first = p_hits[0] if p_hits else None

        # 第一句里就有标志（"原告诉称：判令……"）→ 从标志处开始
        if first is not None and first.start < sents[0].end:
            return first.start, f"第一句标志({first.rule})", first.confidence

        # 跳过开头的案情背景句。最多跳两句：摘要结构是固定的，"案件背景"几乎
        # 总在开头一两句，跳多了会把真正的诉请吃掉。
        #
        # 判定条件是"像案情背景 **且** 这句话里没有诉请标志"。这里刻意不用
        # 裸的动词表兜底：案由本身就是名词短语（"原告某某与被告继承纠纷一案"里的
        # "继承"和"请求分割"的"分割"是同一个词）。做法是先把"……纠纷/一案/关系"
        # 这段案由剥掉，再看剩下的部分有没有诉请动词——
        #   "原被告系借款合同纠纷。"                    → 剥完是空 → 跳过（正确）
        #   "原被告继承纠纷一案，告诉请依法继承房屋份额。" → 剥完还剩"告诉请继承…"
        #                                                → 不跳（正确）
        idx = 0
        skipped = 0
        while idx < len(sents) - 1 and skipped < 2:
            seg = sents[idx].text_of(src)
            has_p_marker = any(h.start < sents[idx].end for h in p_hits)
            rest = CASE_TYPE_TAIL.sub("", seg)
            is_background = any(r.pattern.search(seg) for r in BACKGROUND_RULES)
            if is_background and not has_p_marker and not CLAIM_VERBS.search(rest):
                idx += 1
                skipped += 1
            else:
                break

        base = sents[idx].start if idx < len(sents) else 0
        base_rule = "默认第二句" if idx == 1 else (f"跳过{skipped}句案件背景" if skipped else "文首")

        if idx >= len(sents) - 0:  # 理论上进不来，兜底
            return base, base_rule, Confidence.MEDIUM

        # 标志落在"默认句"里 → 从标志处开始更精确（"……。被告向原告借款，原告诉请判令……"）
        in_base = [
            h
            for h in p_hits
            if base <= h.start < (sents[idx].end if idx < len(sents) else len(src))
        ]
        if in_base and in_base[0].start > base:
            # 无主语的标志（"……，请求判令被告偿还"）如果紧跟在句首，就把句首
            # 的主语一并留下（"原告崔某1请求判令……"）——切出来的片段要能读通。
            prefix = src[base : in_base[0].start]
            if len(prefix) <= 16 and not CLAIM_VERBS.search(prefix):
                return base, base_rule, in_base[0].confidence
            return in_base[0].start, f"默认句内标志({in_base[0].rule})", in_base[0].confidence
        if in_base:
            return in_base[0].start, base_rule, in_base[0].confidence

        # 默认句里连诉请动词都没有，而后面一句有明确标志 → 说明默认句是背景/事实，
        # 继续往后走一句（置信度降为中）。
        nxt_end = (
            sents[min(idx + 1, len(sents) - 1)].end if idx + 1 < len(sents) else len(src)
        )
        if not CLAIM_VERBS.search(src[base:nxt_end]):
            later = [h for h in p_hits if h.start >= nxt_end]
            if later:
                return (
                    later[0].start,
                    f"跳过无诉请内容的句子后取标志({later[0].rule})",
                    Confidence.MEDIUM,
                )
        return base, base_rule, Confidence.MEDIUM if skipped else Confidence.HIGH

    # ------------------------------------------------ ② 答辩起点

    def _resolve_d_start(
        self,
        src: str,
        d_hits: Sequence[Hit],
        p_start: int,
        f_strong: Sequence[Hit],
        f_medium: Sequence[Hit],
        notes: List[str],
    ) -> Optional[Hit]:
        """取第一个答辩标志，并做顺序校验（需求②的两个"不能"）。"""

        if not d_hits:
            return None
        hit = d_hits[0]

        # 不能仅凭"被告"认定答辩开始 —— 词表层面已经要求带动词，这里再挡一次：
        # 标志前面紧邻的如果是"原告/本院/法院"，说明是"原告诉称被告辩称……"之类，
        # 不是答辩段落的起点。
        if hit.start > 0:
            prefix = src[max(0, hit.start - 4) : hit.start]
            if re.search(r"(?:本院|法院)(?:认为|查明)?$", prefix):
                notes.append(f"答辩标志({hit.rule})紧跟在法院叙述之后，按说理回述处理")
                return None

        # 答辩必须晚于诉请起点。早于诉请起点的答辩标志几乎都是案情背景里的
        # 转述（"被告主张……"出现在案由段）。正好落在诉请起点上的情况（摘要里
        # 根本没有诉请段，只有"被告未答辩。"）不在这里挡——那会让诉请区间为空，
        # 由 _enforce_order 把诉请标成"不确定"，比反过来丢掉答辩段落更诚实。
        if hit.start < p_start:
            notes.append(f"答辩标志({hit.rule})位于诉请起点之前，不作为答辩段落起点")
            return None

        # 说理里再次提到"被告辩称"：如果第一个法院叙述标志已经在它前面，
        # 那它是法院复述，不是新的答辩段落（需求②点名的第二个坑）。
        anchors = sorted(list(f_strong) + list(f_medium), key=lambda h: h.start)
        if anchors and anchors[0].start < hit.start:
            notes.append(
                f"答辩标志({hit.rule})出现在法院叙述标志({anchors[0].rule})之后，"
                "按说理回述处理，不计入答辩段落"
            )
            return None
        return hit

    # ------------------------------------------------ ③ 答辩终点（推断）

    def _bare_defendant_backstop(
        self,
        src: str,
        p_start: int,
        f_strong: Sequence[Hit],
        f_medium: Sequence[Hit],
        notes: List[str],
    ) -> Optional[Hit]:
        """"字面 `。被告`"兜底（严格说：句号紧挨着"被告"）。

        只在**所有答辩标志都落空**时才用，且要求那句紧跟着答辩动作词、不是判决条目、
        不带法院语气——否则命中的多半是"以被告为主语的法院说理"。
        """

        m = re.search(r"。被告", src[p_start:])
        if m is None:
            return None
        pos = p_start + m.start() + 1
        # 该句的正文（到下一个句末标点为止）
        end = pos
        while end < len(src) and src[end] not in HARD_SENT_DELIMS + SOFT_SENT_DELIMS:
            end += 1
        sentence = src[pos : end + 1]
        if not D_BARE_BACKSTOP_ACT.match(sentence):
            return None
        if RESULT_MARKER.search(sentence) or COURT_TONE_STRONG.search(sentence):
            return None
        anchors = sorted(list(f_strong) + list(f_medium), key=lambda h: h.start)
        if anchors and anchors[0].start < pos:
            notes.append(f"句首'被告'({pos})落在法院叙述标志之后，兜底不启用")
            return None
        notes.append(f"无常规答辩标志，按'。被告'兜底取起点 {pos}")
        return Hit(pos, min(pos + 2, len(src)), "被告辩称:句首被告兜底", Strength.WEAK, "被告")

    def _apply_bare_defendant_rule(
        self,
        src: str,
        sents: Sequence[Sentence],
        p_start: int,
        d_hit: Optional[Hit],
        f_strong: Sequence[Hit],
        f_medium: Sequence[Hit],
        notes: List[str],
    ) -> Optional[Hit]:
        """"字面 `。被告` 第一次出现处就是答辩起点"——对照规则（默认关闭）。

        注意是**字面**匹配："句号 + 被告" 两个字符紧挨着。句号换成逗号/分号/顿号都
        不算——诉请列表里的"…；被告偿还原告借款本金…"、"…，被告胡建波未作答辩"都
        不匹配，这正是这条规则能躲开诉请条目的原因。

        仍然过一遍顺序校验（必须晚于诉请起点、早于第一个法院叙述标志），否则
        会切出 D 落在法院事实里的非法区间；被校验挡掉的条数会在报告里体现。
        """

        m = re.search(r"。被告", src[p_start:])
        pos = None if m is None else p_start + m.start() + 1
        if pos is None:
            return d_hit
        anchors = sorted(list(f_strong) + list(f_medium), key=lambda h: h.start)
        if anchors and anchors[0].start < pos:
            notes.append(f"句首'被告'({pos})在法院叙述标志之后，顺序校验挡下，沿用原起点")
            return d_hit
        if d_hit is not None and d_hit.start == pos:
            return d_hit
        notes.append(f"启用'句首被告'规则：答辩起点取 {pos}")
        return Hit(pos, min(pos + 2, len(src)), "被告辩称:句首被告(用户规则)", Strength.MEDIUM, src[pos : pos + 2])

    def _infer_d_end(
        self,
        src: str,
        sents: Sequence[Sentence],
        d_start: int,
        tail_start: Optional[int],
    ) -> Tuple[int, str, Confidence, List[str]]:
        """没有法院叙述标志时推断答辩终点。

        需求②要求"不得默认答辩只占一句"：这里从答辩起点所在句的**下一句**开始
        逐句打分，第一句明显是法院语气（本院/法院/经查/查明/证据/合法有效……）
        且不像答辩的句子，才算答辩结束。

        打分是"计数"而不是"命中与否"：一句话里同时出现"有效""违约""应当承担"
        这类说理词，比只出现一个更像法院的话。分数要到 2 分才认——单靠一个
        "合法"就下刀，会把"被告辩称合同合法有效"这种答辩句当成说理切掉。

        找不到就返回"推断失败"：调用方会把答辩和法院事实都记成不确定，并把
        答辩终点暂放在尾部起点，宁可报"定不下来"，也不砍一刀假装很确定。
        """

        notes: List[str] = []
        d_sent_idx = 0
        for i, s in enumerate(sents):
            if s.start <= d_start < s.end:
                d_sent_idx = i
                break
        for s in sents[d_sent_idx + 1 :]:
            if tail_start is not None and s.start >= tail_start:
                break
            seg = s.text_of(src)
            if not seg.strip():
                continue
            if re.search(r"辩称|答辩|抗辩", seg):
                # 多被告连续答辩：这一句还是答辩。
                # 这里用裸词检查而不是跑一遍 _hits：推断是逐句做的，跑整套词表
                # 会让切分从"每毫秒几十条"掉到"每秒几百条"。
                continue
            score = _court_like_score(seg)
            if score >= 2:
                conf = Confidence.MEDIUM if score >= 3 else Confidence.LOW
                notes.append(f"无法院叙述标志，按句边界+法院语气推断答辩终点（{s.start}）")
                return s.start, f"推断:法院语气句边界({seg[:12]}…)", conf, notes

        end = tail_start if tail_start is not None else len(src)
        notes.append("答辩段落没有可识别的终点，法院事实起点无法确定")
        return end, "推断失败:无可用句边界", Confidence.LOW, notes

    # ------------------------------------------------ ③ 法院事实起点（推断）

    def _infer_f_start(
        self,
        src: str,
        sents: Sequence[Sentence],
        p_start: int,
        tail_start: Optional[int],
    ) -> Tuple[int, str, Confidence, List[str]]:
        """既没有答辩、也没有法院叙述标志时的法院事实起点推断。"""

        notes: List[str] = []
        for s in sents:
            if s.start <= p_start:
                continue
            if tail_start is not None and s.start >= tail_start:
                break
            seg = s.text_of(src)
            if not seg.strip():
                continue
            if _court_like_score(seg) >= 2 and not re.search(r"辩称|答辩", seg):
                notes.append(f"无答辩、无法院叙述标志，按法院语气句推断法院事实起点（{s.start}）")
                return s.start, f"推断:法院语气句({seg[:12]}…)", Confidence.LOW, notes
        pos = tail_start if tail_start is not None else len(src)
        notes.append("无答辩且无法院叙述标志，法院事实起点无法确定")
        return pos, "推断失败:无可用句边界", Confidence.LOW, notes

    # ------------------------------------------------ 尾部（法律依据 + 裁判结果）

    def _find_tail(self, src: str, sents: Sequence[Sentence]) -> Optional[Hit]:
        """定位"法律依据及裁判结果"的起点，它就是 `court_facts` 的终点。

        流程：候选只在全文尾部区域里找（前 4 句/45% 的靠前那个为界）→ 按强度
        档取候选（强档有就绝不用弱档）→ 同档里取**最靠右**的候选（尾部一定是
        最后一段，取最靠右可避开说理里"依据《……》驳回"这种伪尾部）→ 再做
        "把前面的纯法律依据从句并进来"的扩展。
        """

        if not sents:
            return None
        n = len(src)
        region_start = min(
            sents[max(0, len(sents) - self.tail_lookback_sentences)].start,
            int(n * self.tail_lookback_ratio),
        )

        candidates: List[Tuple[Confidence, int, str, re.Match[str]]] = []
        for rule in TAIL_RULES:
            for m in rule.pattern.finditer(src):
                if m.start() < region_start:
                    continue
                # 迟延履行利息/诉讼费这类收尾句不算法律依据段：只有候选**起点本身**
                # 就落在收尾句里（"……本判决生效之日起十日内……"是裁判结果的一部分，
                # 不能因为出现"本判决"就把整个尾部否掉）才跳过。
                if TAIL_AFTERMATH.search(src[max(0, m.start() - 8) : m.start() + 4]):
                    continue
                if rule.citation_gap and not gap_is_citation_like(_gap_before_result(m)):
                    continue
                candidates.append((_STRENGTH_CONF[rule.strength], m.start(), rule.name, m))

        # 句首就是裁判结果标志的句子（"判决驳回原告的诉讼请求。"），补一条候选
        head_re = re.compile(
            r"\s*(?:据此|为此|故|综上(?:所述)?|依照|依据|根据|依|依法|判决|裁定|驳回)[^。；;！？!?]{0,6}?"
            r"(?:判决|裁定|驳回)"
        )
        for s in sents:
            if s.start < region_start:
                continue
            seg = s.text_of(src)
            if head_re.match(seg) and not TAIL_AFTERMATH.search(seg[:24]):
                candidates.append((Confidence.MEDIUM, s.start, "尾部:句首裁判结果标志", None))  # type: ignore[arg-type]

        if not candidates:
            return None

        best_rank = min(_CONF_ORDER[c[0]] for c in candidates)
        tier = [c for c in candidates if _CONF_ORDER[c[0]] == best_rank]
        conf, pos, rule_name, m = max(tier, key=lambda c: c[1])

        start = pos
        if m is not None:
            start = self._extend_tail_start(src, pos, m, rule_name)
        # 前一句是纯法律依据从句 → 一起并进尾部（"依照《X》第五条之规定。判决……"）
        while True:
            prev = None
            for s in sents:
                if s.end <= start:
                    prev = s
                else:
                    break
            if prev is None:
                break
            seg = prev.text_of(src)
            if not _is_legal_basis_clause(seg):
                break
            start = prev.start
        return Hit(start, len(src), rule_name, _conf_to_strength(conf), src[start:])

    def _extend_tail_start(
        self, src: str, pos: int, m: re.Match[str], rule_name: str
    ) -> int:
        """把尾部起点往前扩到"法律依据从句"的开头。

        两种情形：
          * "判决如下"前面半句就是"依照《……》之规定，"（同句内）；
          * "《合同法》、《民事诉讼法》的规定，判决……"（规则命中处就带书名号）。
        """

        sent_start = 0
        for s in split_sentences(src):
            if s.start <= pos < s.end:
                sent_start = s.start
                break
        seg = src[sent_start:pos]
        if not seg:
            return pos

        # 找**从句开头**的那一个法律依据引导词："依照《X》第二条、《Y》第三十一条
        # 之规定，判决如下"的起点是"依照"，不是后面那个书名号；反过来，如果从后
        # 往前取，就会把"依照"吞掉（只剩半句引用）。取第一个合格的引导词即可。
        #
        # 距离上限 80 字：语料里有整段没有句号的摘要（一句话几百字），不设上限时
        # "判决如下"会一路扩回段首，把法院事实整段吞进尾部。
        max_back = 80
        for mm in re.finditer(r"《|依照|依据|根据|按照|参照|依|据此|综上(?:所述)?|故|因此|为此", seg):
            if pos - (sent_start + mm.start()) > max_back:
                continue
            if mm.start() > 0:
                # 只在"从句开头"处扩展："无法律依据"里的"依据"不是从句开头，
                # 顺着它往前扩会把说理吃进尾部。
                prev_ch = seg[mm.start() - 1]
                if prev_ch not in "。；;！？!?，,：:、（）()“”\u3000 \t\n":
                    continue
            if REASONING_TONE.search(seg[mm.start() :]):
                continue
            return min(sent_start + mm.start(), pos)
        return pos

    # ------------------------------------------------ 顺序校验

    @staticmethod
    def _trim_edge_punctuation(src: str, spans: Dict[str, ElementSpan]) -> None:
        """把切片两端悬挂的标点收掉（"…优先受偿权，" → "…优先受偿权"）。

        边界是字符级的，下一个要素的起点可能落在一句话中间，于是前一个要素
        会以"、"结尾——那半个顿号属于**下一段**的引导，不该留在前一个要素里。
        只动标点，不动文字，所以偏移量仍然对得上原文。
        """

        for key in ELEMENTS:
            span = spans[key]
            if span.status is not Status.IDENTIFIED or span.start is None or span.end is None:
                continue
            # 只收**从句分隔符**（逗号/顿号/冒号）和空白：分号是软句界，
            # "……未答辩；"这种收尾本来就是合法的句子收尾，收掉反而变差。
            while span.end > span.start and src[span.end - 1] in "，,、：: \t\n":
                span.end -= 1
            while span.end > span.start and src[span.start] in "，,、：: \t\n":
                span.start += 1

    def _enforce_order(
        self, src: str, spans: Dict[str, ElementSpan], notes: List[str]
    ) -> None:
        """把违反 P→D→F 或不重叠约束的结果降级成 `uncertain`，并记 note。

        这一步是"顺序校验"的最后一道闸：单看某个标志都合理，但拼起来越界
        （例如答辩终点落在起点之前）时，宁可不切也不产出一个非法区间。
        """

        n = len(src)
        p, d, f = spans["plaintiff_claims"], spans["defendant_defenses"], spans["court_facts"]

        def clamp(span: ElementSpan, label: str) -> None:
            if span.start is None or span.end is None:
                return
            if span.start < 0:
                span.start = 0
                notes.append(f"{label}起点越界，已截到 0")
            if span.end > n:
                span.end = n
                notes.append(f"{label}终点越界，已截到文末")

        for key in ELEMENTS:
            clamp(spans[key], ELEMENT_NAMES[key])

        if p.status is Status.IDENTIFIED and (p.end is None or p.end <= p.start):
            notes.append("诉请区间为空或反向，降级为不确定")
            p.status = Status.UNCERTAIN
            p.notes.append("诉请区间为空或反向")

        if d.status is Status.IDENTIFIED:
            if d.end is None or d.end <= d.start:
                notes.append("答辩区间为空或反向，降级为不确定")
                d.status = Status.UNCERTAIN
                d.notes.append("答辩区间为空或反向")
            elif p.start is not None and d.start < p.end:
                notes.append("答辩起点早于诉请终点，降级为不确定")
                d.status = Status.UNCERTAIN
                d.notes.append("答辩起点早于诉请终点")

        if f.status is Status.IDENTIFIED and (f.end is None or f.end <= f.start):
            # 法院事实为空：可能是"摘要里确实没有法院事实段落"，但规则无法确认，
            # 按需求记不确定，不直接判缺失。位置留成空区间，方便人工定位。
            notes.append("法院事实区间为空（尾部紧接答辩/诉请终点），降级为不确定")
            f.status = Status.UNCERTAIN
            f.notes.append("法院事实区间为空，无法确认是缺失还是解析失败")
            if f.start is None:
                f.start = f.end

        # 答辩起点落在法院事实（甚至尾部）内部：这说明那是个误检的答辩标志
        # （例如"被告承认…"出现在法院说理里）。撤销答辩段落，并把诉请终点
        # 交还给法院事实起点——不能输出 D.start > F.end 这种非法区间。
        if (
            d.status is not Status.ABSENT
            and d.start is not None
            and f.start is not None
            and d.start >= f.start
        ):
            notes.append(
                f"答辩起点({d.start})落在法院事实起点({f.start})之后，判为误检并撤销答辩段落"
            )
            d.status = Status.ABSENT
            d.start = None
            d.end = None
            d.notes.append("答辩标志落在法院事实/尾部区间内，按误检撤销")
            if p.end is None or p.end > f.start:
                p.end = f.start
                p.end_rule = "答辩误检被撤销后，诉请终点=法院事实起点"
            if d.confidence is Confidence.HIGH:
                d.confidence = Confidence.MEDIUM

    @staticmethod
    def _check_order(spans: Dict[str, ElementSpan]) -> bool:
        """检查 P(start,end) → D(start,end)? → F(start,end) 有序且不重叠。"""

        seq = []
        for key in ELEMENTS:
            span = spans[key]
            if span.status is Status.ABSENT:
                continue
            if span.start is None or span.end is None:
                return False
            if span.end < span.start:
                return False
            seq.append((span.start, span.end))
        for (s1, e1), (s2, e2) in zip(seq, seq[1:]):
            if e1 > s2:
                return False
        return True


def _conf_to_strength(conf: Confidence) -> Strength:
    return {
        Confidence.HIGH: Strength.STRONG,
        Confidence.MEDIUM: Strength.MEDIUM,
        Confidence.LOW: Strength.WEAK,
    }[conf]


def _sentence_end_after(sents: Sequence[Sentence], pos: int) -> int:
    """返回包含 `pos` 的那句话的终点（找不到就用文末）。"""

    for s in sents:
        if s.start <= pos < s.end:
            return s.end
    return sents[-1].end if sents else pos


#: 编号项开头："1、""（2）""三."——诉请列表和裁判结果列表都用它。
LIST_ITEM_START = re.compile(r"^\s*(?:[（(]?[0-9０-９一二三四五六七八九十]{1,3}[)）]?\s*[、.．)）])")

#: 案由短语（"……借款合同纠纷一案"）。判断首句是不是案情背景时先把它剥掉，
#: 免得案由里的"继承""赔偿"被当成诉请动词。
CASE_TYPE_TAIL = re.compile(r"[^，,。；;！？!?]{0,20}?(?:纠纷|一案|关系)")

#: "像法律依据"的最小条件：要么带书名号，要么"第X条"，要么
#: "依照/依据/根据……法律/法规/规定/法条/解释/条例"。
#: 只写"根据……"是不够的——日常汉语里"根据某某制度/情况/约定"到处都是。
LEGAL_CLAUSE = re.compile(
    r"《|第[一二三四五六七八九十百千零0-9]{1,4}条|之规定|的规定|的有关规定|"
    r"(?:依照|依据|根据|按照|参照|依)[^，,。；;！？!?]{0,24}?(?:法律|法规|规定|法条|解释|条例)"
)


def _is_sentence_start(sents: Sequence[Sentence], pos: int) -> bool:
    """`pos` 是不是某句话的起点（用来判断标志是不是"另起一句"）。"""

    return any(s.start == pos for s in sents)


def _skip_list_items(src: str, sents: Sequence[Sentence], pos: int) -> int:
    """如果 `pos` 处是一个编号项的开头，就一路跳过连续的编号项，返回新位置。"""

    cur = pos
    while True:
        nxt = None
        for s in sents:
            if s.start == cur:
                nxt = s
                break
        if nxt is None or not LIST_ITEM_START.match(src[nxt.start : nxt.end]):
            return cur
        cur = nxt.end


def _gap_before_result(m: "re.Match[str]") -> str:
    """取"引用串 → 裁判结果动词"之间的那一段文字（`citation_gap` 校验用）。"""

    text = m.group()
    vm = re.search(r"(?:判决|裁定|驳回|如下)", text)
    return text[: vm.start()] if vm else text


def _court_like_score(seg: str) -> int:
    """一句话有多像"法院在叙述"。

    * 出现一个强语素（本院/法院/经查/查明/合法有效/依据……）+2，多一个再加 1；
    * 弱语素（合法/违约/责任/赔偿……）攒够 3 个才 +2；
    * 出现"诉请/诉求/请求判令"这类当事人主张用语 -2（这一句更像在转述主张）。
    """

    score = 0
    strong = {m.group() for m in COURT_TONE_STRONG.finditer(seg)}
    weak = {m.group() for m in COURT_TONE_WEAK.finditer(seg)}
    if strong:
        score += 2 + len(strong)
    if len(weak) >= 3:
        score += 2
    if re.search(r"诉请|诉求|诉讼请求|请求判令|请求判决", seg):
        score -= 2
    return score


def _is_legal_basis_clause(text: str) -> bool:
    """判断一小段文字是不是"纯法律依据从句"（可以用来把尾部起点往前扩）。"""

    seg = text.strip()
    if not seg or len(seg) > 120:
        return False
    # 必须是**真的引用了法律**："根据年度绩效考核制度发放绩效工资"里有"根据"，
    # 但那是"按照（制度）"的日常用法，不是法律依据，不能据此把尾部提前。
    if not LEGAL_CLAUSE.search(seg):
        return False
    if REASONING_TONE.search(seg):
        return False
    if CLAIM_VERBS.search(seg) and not re.search(r"判决|裁定|驳回", seg):
        return False
    return True


#: 默认实例。切分器无状态，直接用模块级函数最省事。
_DEFAULT = Segmenter()


def segment_summary(text: str) -> SegmentationResult:
    """模块级快捷入口（等价于 `Segmenter().segment(text)`）。"""

    return _DEFAULT.segment(text)


__all__ = [
    "ELEMENTS",
    "ELEMENT_NAMES",
    "Status",
    "Confidence",
    "Sentence",
    "Hit",
    "ElementSpan",
    "SegmentationResult",
    "Segmenter",
    "segment_summary",
    "split_sentences",
    "empty_result",
]

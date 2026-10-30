"""记忆提取器：把含噪音的原始文本整理成可入库的「记忆片段」。

它只负责一件事：把话记下来。去噪、切块、指代消解都在这一步做完，一条事实都不许丢。
抽实体/边/属性是下一步（LLM_Adapter.preheat_memories）的事，两者彻底解耦：
这一步出问题只会让图上少点东西，不会让记忆少存；反过来抽图失败，片段也已经入库了。
"""

from MomoGraph.Core.LLM import LLM_Adapter


class MemoryExtractor:
    def __init__(self, llm: LLM_Adapter | None = None):
        self.llm = llm or LLM_Adapter()

    def extract(self, text: str) -> list[str]:
        """原文 → 记忆片段。模型什么都没给时退回原文，保证记忆不会丢。"""
        text = text.strip()
        if not text:
            return []
        # 保底：抽取失败（空列表）就把原样整段存进去，宁可脏也不能漏
        fragments = self.llm.extract_memories(text) or [text]
        return list(dict.fromkeys(fragments))

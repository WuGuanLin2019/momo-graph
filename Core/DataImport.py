"""外部记忆导入：目前对接 mem0 导出的 JSON。

单独一个类的原因：导入要做的是「格式适配 + 批量 + 幂等」这类和数据落盘无关的事；
DataHandler 只负责把一批记忆变成图节点，这里负责怎么分批、怎么省钱、怎么续跑。
"""

from datetime import datetime
from time import time

from MomoGraph.Data.DataHandle import DataHandler
from MomoGraph.Data.DataStructure import Graph

DEFAULT_BATCH_SIZE = 15  # 一批走一次 LLM；165 条 ≈ 11 次调用，比一条一次省一个数量级


def _parse_time(value) -> int:
    """mem0 的 created_at 是带时区的 ISO 字符串，转成项目统一用的秒级时间戳。"""
    if not value:
        return int(time())
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return int(time())


class DataConnection():
    def __init__(self, data_handler: DataHandler | None = None):
        self.data_handler = data_handler or DataHandler()

    def import_data(
        self,
        graph: Graph,
        data,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_batch=None,
        per_item: bool = False,
    ) -> dict:
        """把 mem0 导出（或同结构的列表）导入图中，返回统计。

        data: {"memories": [...]} 或直接是记忆列表。
        on_batch: 每批成功后的回调，一般用来落盘——长导入中途断了，重跑会跳过已导入的，
                  不用重头再花一遍 LLM 钱。
        per_item: True 时逐条走 remember 的链路（LLM 切块+指代消解），质量最好但一条一次调用；
                  False 时批量抽取，一批一次调用，约省一个数量级。
        """
        memories = self._memories_of(data)
        report = {"total": len(memories), "imported": 0, "skipped": 0, "batches": 0, "llm_calls": 0}

        # 幂等：原文已经在 chunk 里（上次导了一半或者重复导出）就不再花 LLM
        existing = {chunk.content.strip() for chunk in graph.memory_chunks.values()}
        pending = []
        for item in memories:
            text = str(item.get("memory") or "").strip()
            if not text or text in existing:
                report["skipped"] += 1
                continue
            existing.add(text)
            pending.append(
                {
                    "memory": text,
                    "provider": item.get("attributed_to") or "user",
                    "create_time": _parse_time(item.get("created_at") or item.get("updated_at")),
                }
            )

        # 按时间顺序导入，图里的时间线和真实发生顺序一致
        pending.sort(key=lambda item: item["create_time"])

        step = 1 if per_item else batch_size
        for start in range(0, len(pending), step):
            batch = pending[start:start + step]
            if per_item:
                # 逐条走原链路：LLM 切块 + 指代消解，脏数据（错别字、指代）交给 LLM 兜
                for item in batch:
                    self.data_handler.input_sentence(
                        item["memory"], graph, item["provider"], create_time=item["create_time"]
                    )
            else:
                self.data_handler.import_memories(batch, graph)
            report["imported"] += len(batch)
            report["batches"] += 1
            report["llm_calls"] += len(batch) if per_item else 1
            if on_batch:
                on_batch(report)
        return report

    @staticmethod
    def _memories_of(data) -> list[dict]:
        if isinstance(data, dict):
            return list(data.get("memories") or [])
        return list(data or [])

    def export_data(self):
        pass

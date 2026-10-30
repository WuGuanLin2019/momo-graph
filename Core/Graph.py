import time
from MomoGraph.Core.Community import refresh_communities
from MomoGraph.Core.DataImport import DataConnection
from MomoGraph.Core.DataSave import DataSaver
from MomoGraph.Core.Embedding import update_embeddings
from MomoGraph.Core.LLM import LLM_Adapter
from MomoGraph.Core.Retriever import Retriever
from MomoGraph.Data.DataHandle import DataHandler
from MomoGraph.Data.DataStructure import Graph
from MomoGraph.Data.Dedup import Deduplicator
from MomoGraph.Data.MemoryText import build_prompt


class GraphHandler():
    def __init__(self):
        self.data_handler = DataHandler()
        self.data_saver = DataSaver()
        self.llm = LLM_Adapter()
        self.graph = Graph()

        self.init_data()

    def init_data(self):
        self.graph = self.data_saver.load_from_file()

    def remember(self, memory_str:str,provider:str="user",create_time:int|None=None)->str:
        #输入：自然语言。返回：已有记忆/没有值得记下的/成功记一下xxx
        #流程如下：
        # LLM切块+指代消解
        # 建图
        self.data_handler.input_sentence(memory_str,self.graph,provider,create_time)
        # 默认写到 MomoGraph/Temp/graph.json，dashboard 实时模式读的就是它
        self.data_saver.save_to_file(self.graph)

    def import_memories(self, payload, batch_size: int = 15, per_item: bool = False) -> dict:
        # 导入 mem0 这类导出的外部记忆；每批落盘一次，中途断了重跑会跳过已导入的
        def save(report):
            self.data_saver.save_to_file(self.graph)

        report = DataConnection(self.data_handler).import_data(
            self.graph, payload, batch_size=batch_size, on_batch=save, per_item=per_item
        )
        self.data_saver.save_to_file(self.graph)
        return report

    def remind(self, query:str) -> str:
        #输入：自然语言。返回：依据记忆拼 prompt，再让 LLM 回答
        result = Retriever(self.graph).retrieve(query)
        return build_prompt(query, result)

    def reindex(self) -> dict:
        # 补/刷新向量：chunk 缺失才补，实体画像与边每次重算（数据量小）
        report = update_embeddings(self.graph)
        self.data_saver.save_to_file(self.graph)
        return report

    def refresh_community(self, force: bool = False) -> dict:
        # 手动入口：只刷新社区结构+摘要，不跑去重
        report = refresh_communities(self.graph, self.llm, force=force)
        self.data_saver.save_to_file(self.graph)
        return report

    def meditate(self, refresh_community: bool = False) -> dict:
        # 时机:主动调用/或许后续嵌入到哪个操作里
        # 实体、边、属性去重（骨架期返回全 0，逻辑在 Deduplicator 里逐步填充）
        # 社区发现->写社区摘要
        start_time = time.time()
        report = Deduplicator(self.graph).run()
        if refresh_community:
            report["community"] = refresh_communities(self.graph, self.llm)
        self.data_saver.save_to_file(self.graph)
        end_time = time.time()
        print(f"meditate time: {end_time - start_time}")
        return report

if __name__ == "__main__":
    mm = GraphHandler()
    sample_memories = [
        """路人：我长话短说,这事情，说来话长了
路人：我爱上了我的同事阿志
路人：可是我没有告诉他
路人：我告诉了我另一个同事阿豪
路人：阿豪答应我不告诉别人
路人：可是我的同事阿德告诉我
路人：阿豪偷偷地告诉了另一个同事阿林
路人：阿林又和阿志以前的女朋友阿兰很熟
路人：我怕阿林会告诉阿兰
路人：然后阿兰回去告诉阿志
路人：这样我就会很尴尬""",
        """路人：幸好现在阿兰正在和阿德谈恋爱
路人：所以我就去找阿德帮我解决这个问题
路人：阿德跟我说阿兰和他已经分手了
路人：他现在跟阿林好上了让我去找阿林
路人：可是阿林跟我说阿豪其实根本没有跟他说什么
路人：现在我晕了我到底应该相信谁呀
曾小闲：阿欧""",
        """路人：我说的是阿志不是阿欧，阿欧是我另一个部门的同事
路人：阿欧是阿林的弟弟
路人：可是我不能和他说我喜欢的人是你哥哥的女朋友的前男友
路人：这样关系就更乱了
曾小闲：阿嚏
路人：阿T？
路人：你怎么知道我们公司里还有个同事叫阿T？
路人：他和阿林有仇，可能因为他暗恋阿兰的关系
路人：不过阿T和阿豪关系不错""",
    ]
    for memory in sample_memories:
        # mm.remember(memory)
        pass

  

    # query_result = mm.remind("路人的处境")
    # print(query_result) 
    # print(mm.llm.answer("路人很郁闷，她应该怎么办？你给点意见，下面是你知道的关于她的情况。\n" + query_result))
    # import json
    # payload = json.load(open("E:/Study/GitProject/test-data/mem0_export_momo_20261022_173352.json", encoding="utf-8"))
    # print(mm.import_memories(payload, batch_size=15))
    # mm.remember("记忆里的“用户”其实也是指Wu")
    # mm.reindex()
    # mm.meditate(True) 
    # mm.refresh_community()

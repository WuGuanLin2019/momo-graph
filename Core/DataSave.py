import json
import os

from MomoGraph.Data.DataStructure import Graph
from MomoGraph.Data.GraphIO import decode_graph, encode_graph, is_graph_payload

# 数据目录：MomoGraph/Temp。dashboard 也从这里读（经 /graph_view 转成视图）
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Temp")
DEFAULT_GRAPH_PATH = os.path.join(TEMP_DIR, "graph.json")


class DataSaver():
    def save_to_file(self, data: Graph, file_path: str = DEFAULT_GRAPH_PATH) -> str:
        # 存完整图结构（可逆），不是 dashboard 视图；看板需要的格式由 Viewer 现算
        payload = encode_graph(data)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        # 先写临时文件再替换，避免 dashboard 轮询时读到写了一半的文件
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)
        return file_path

    def load_from_file(self, file_path: str = DEFAULT_GRAPH_PATH) -> Graph:
        """把文件还原成 Graph；文件不存在或格式不对时返回空图（首次运行/旧格式升级）。"""
        if not os.path.exists(file_path):
            return Graph()

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not is_graph_payload(payload):
            # 旧版存的是 dashboard 视图（实体名/边/统计），信息不可逆，只能从空图开始
            print(f"[DataSaver] {file_path} 不是完整图格式，已忽略（本次从空图开始）")
            return Graph()

        return decode_graph(payload)

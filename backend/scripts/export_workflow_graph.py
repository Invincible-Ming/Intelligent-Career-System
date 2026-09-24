# import sys
# from pathlib import Path
#
# # 将 backend 目录动态添加到 Python 模块搜索路径中
# backend_dir = str(Path(__file__).resolve().parent.parent)
# if backend_dir not in sys.path:
#     sys.path.insert(0, backend_dir)
#
# from app.agents.workflow import match_workflow
#
#
# def generate_mermaid():
#     # 导出 LangGraph 的 Mermaid 流程图文本
#     mermaid_code = match_workflow.get_graph().draw_mermaid()
#
#     print("=== 生成的 Mermaid 语法如下 ===\n")
#     print(mermaid_code)
#     print("\n===============================")
#
#     # 保存到 backend 目录下的 workflow_graph.mmd
#     output_path = Path(backend_dir) / "workflow_graph.mmd"
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write(mermaid_code)
#     print(f"✅ 图表已保存至: {output_path}")
#
#
# if __name__ == "__main__":
#     generate_mermaid()

import langgraph
import langgraph.checkpoint.postgres

print("LangGraph 版本:", langgraph.__version__)
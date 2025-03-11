import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from codeminer.reference_build_cpp import build_graph

project_path = "/my_project/"
graph = build_graph(project_path)

print("\nFunction Calls:")
for u, v, data in graph.edges(data=True):
    print(f"{u} -> {v} (type: {data['edge_type']})")
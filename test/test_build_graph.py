from codeminer import build_graph
import os

def test_build_graph():
    repo_path = "./my_project"
    # expand to full path
    repo_path = os.path.abspath(repo_path)
    graph = build_graph(repo_path)
    # print(graph)
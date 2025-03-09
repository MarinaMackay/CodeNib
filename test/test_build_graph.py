import os

from codeminer import build_graph


def test_build_graph():
    repo_path = "./my_project"
    # expand to full path
    repo_path = os.path.abspath(repo_path)
    build_graph(repo_path)
    # print(graph)


if __name__ == "__main__":
    test_build_graph()

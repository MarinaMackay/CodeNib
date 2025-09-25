import re


def is_test_file(nid):
    """Check if a node ID belongs to a test file"""
    if ":" in nid:
        file_path = nid.split(":")[0]
    else:
        file_path = nid
    word_list = re.split(r" |_|\/", file_path.lower())  # split by ' ', '_', and '/'
    return any([word.startswith("test") for word in word_list])


def wrap_code_snippet(code_snippet, start_line, end_line):
    """Wrap code snippet with line numbers"""
    lines = code_snippet.split("\n")

    # Remove trailing empty lines caused by trailing newlines
    while lines and lines[-1] == "":
        lines.pop()

    # Handle None values for files (use 0-based line numbering)
    if start_line is None:
        start_line = 0
    if end_line is None:
        end_line = len(lines) - 1

    max_line_number = start_line + len(lines) - 1
    number_width = len(str(max_line_number))
    return "\n".join(
        f"{str(i + start_line).rjust(number_width)} | {line}"
        for i, line in enumerate(lines)
    )

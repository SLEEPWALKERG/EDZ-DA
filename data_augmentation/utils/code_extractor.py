import re


def extract_code_from_markdown(markdown_string):
    # 使用正则表达式匹配代码块
    code_pattern = re.compile(r'```(?:[a-zA-Z]+)?\n([^`]+)```', re.DOTALL)

    # 查找所有匹配的代码块
    matches = code_pattern.findall(markdown_string)

    # 返回匹配的代码块列表
    return matches
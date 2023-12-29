""" Take the outputs from all the spack repos and turn them into strings
    for CLM training.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
import json
from typing import List, Optional

# tpl imports
from alive_progress import alive_it
import numpy as np
from transformers import AutoTokenizer


def get_file_tree_string(root: List[dict], prefix: str = "", ascii: bool = False, depth: int = 0, max_depth: Optional[int] = None):
    """ Get the string representation of a file tree.
        It will be formatted as:

        root/
        ├── subdir/
        |   ├── file1
        |   └── file2
        └── file3
    """
    space =  '    ' if ascii else '    '
    branch = '|   ' if ascii else '│   '
    tee =    '|-- ' if ascii else '├── '
    last =   '|__ ' if ascii else '└── '

    output = ""
    pointers = [tee] * (len(root) - 1) + [last]
    for pointer, child in zip(pointers, root):
        yield prefix + pointer + child['name'] + ("/" if child['type'] == 'directory' else "")
        if child['type'] == 'directory' and 'contents' in child and len(child['contents']) > 0 and (max_depth is None or depth < max_depth):
            extension = branch if pointer == tee else space
            yield from get_file_tree_string(child['contents'], prefix + extension, ascii=ascii, depth=depth+1, max_depth=max_depth)


def remove_top_comment(python_src: str) -> str:
    """ Remove lines that start with # from the top of the file.
    """
    lines = python_src.split("\n")
    while len(lines) > 0 and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines)

def get_package_string(package_info: dict, ascii: bool = True, max_tree_depth: Optional[int] = None) -> str:
    """ Get the string representation of a package.
        It will be formatted as:

        package name: <package_name>
        url: <url>
        versions:
            <versions>
        file tree:
            <file_tree>
        README.md:
            <README.md>
        Makefile:
            <Makefile>
        CMakeLists.txt:
            <CMakeLists.txt>
        setup.py:
            <setup.py>
        requirements.txt:
            <requirements.txt>
        spack package.py:
            <spack_package.py>
    """
    if len(package_info['file_tree']) > 1:
        package_info['file_tree'] = [{
            'name': package_info['package_name'],
            'type': 'directory',
            'contents': package_info['file_tree']
        }]

    def add_if(fname: str, key: str) -> str:
        if key in package_info and fname in package_info[key]:
            return f"{fname}:\n{package_info[key][fname]}\n"
        return ""

    version_str = "\n".join(" - ".join(v) for v in package_info['versions'])

    output = ""
    output += f"package name: {package_info['package_name']}\n"
    output += f"url: {package_info['source_url']}\n"
    output += f"versions:\n{version_str}\n"
    output += f"file tree:\n"
    output += "\n".join(get_file_tree_string(package_info['file_tree'], ascii=ascii, max_depth=max_tree_depth)) + "\n"
    output += add_if('README.md', 'markdown_files')
    output += add_if('Makefile', 'build_files')
    output += add_if('CMakeLists.txt', 'build_files')
    output += add_if('setup.py', 'build_files')
    output += add_if('requirements.txt', 'build_files')
    output += "spack package.py:\n" + remove_top_comment(package_info['package_file']) + "\n"
    return output

def count_lines(s: str) -> int:
    with open(s, 'r') as fp:
        return len(fp.readlines())

def main():
    input_ds = './dataset.jsonl'
    output_ds = './dataset-clm.jsonl'
    text_column = 'text'
    text_unicode_column = 'text_unicode'
    max_depth = None
    num_lines = count_lines(input_ds)
    lengths = []
    tokens = []

    tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf')

    with open(input_ds, 'r') as fp_in, open(output_ds, 'w') as fp_out:
        for line in alive_it(fp_in, title='Processing packages', total=num_lines):
            package_info = json.loads(line)
            output = {
                text_column: get_package_string(package_info, ascii=True, max_tree_depth=max_depth),
                text_unicode_column: get_package_string(package_info, ascii=False, max_tree_depth=max_depth)
            }
            lengths.append(len(output[text_column]))
            tokens.append(len(tokenizer.encode(output[text_column])))
            json.dump(output, fp_out)
            fp_out.write('\n')

    print(f"Average length: {sum(lengths) / len(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Median length: {np.median(lengths)}")

    print(f"Average tokens: {sum(tokens) / len(tokens):.2f}")
    print(f"Max tokens: {max(tokens)}")
    print(f"Min tokens: {min(tokens)}")
    print(f"Median tokens: {np.median(tokens)}")
    print(f"Above 16k tokens: {sum(1 for t in tokens if t > 16000)}")

if __name__ == '__main__':
    main()
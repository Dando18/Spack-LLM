""" Take the outputs from all the spack repos and turn them into strings
    for CLM training.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
import json
from typing import List

# tpl imports
from alive_progress import alive_it


def get_file_tree_string(root: List[dict], prefix: str = "", ascii: bool = False):
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
        if child['type'] == 'directory' and 'contents' in child and len(child['contents']) > 0:
            extension = branch if pointer == tee else space
            yield from get_file_tree_string(child['contents'], prefix + extension, ascii=ascii)


def remove_top_comment(python_src: str) -> str:
    """ Remove lines that start with # from the top of the file.
    """
    lines = python_src.split("\n")
    while len(lines) > 0 and lines[0].startswith("#"):
        lines = lines[1:]
    return "\n".join(lines)

def get_package_string(package_info: dict, ascii: bool = True) -> str:
    """ Get the string representation of a package.
        It will be formatted as:

        package name: <package_name>
        url: <url>
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

    output = ""
    output += f"package name: {package_info['package_name']}\n"
    output += f"url: {package_info['source_url']}\n"
    output += f"file tree:\n"
    output += "\n".join(get_file_tree_string(package_info['file_tree'], ascii=ascii)) + "\n"
    output += add_if('README.md', 'markdown_files')
    output += add_if('Makefile', 'build_files')
    output += add_if('CMakeLists.txt', 'build_files')
    output += add_if('setup.py', 'build_files')
    output += add_if('requirements.txt', 'build_files')
    output += "spack package.py:\n" + remove_top_comment(package_info['package_file']) + "\n"
    return output


def main():
    input_ds = './dataset.jsonl'
    output_ds = './dataset-clm.jsonl'
    text_column = 'text'
    text_unicode_column = 'text_unicode'

    with open(input_ds, 'r') as fp_in, open(output_ds, 'w') as fp_out:
        for line in alive_it(fp_in, title='Processing packages'):
            package_info = json.loads(line)
            output = {
                text_column: get_package_string(package_info, ascii=True),
                text_unicode_column: get_package_string(package_info, ascii=False)
            }
            json.dump(output, fp_out)
            fp_out.write('\n')


if __name__ == '__main__':
    main()
""" Write a spack package based on a given repo.
"""
# std imports
from argparse import ArgumentParser
from glob import glob
import os
from os import PathLike
from pathlib import Path
import requests
import shutil
import subprocess
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List, Mapping, Optional, Tuple
import urllib

# tpl imports
from llama_cpp import Llama

def get_args():
    parser = ArgumentParser(description=__doc__)
    package_group = parser.add_argument_group("Package Data")
    package_group.add_argument("-p", "--package-name", required=True, help="Name of the package to create")
    package_group.add_argument("--max-file-tree-depth", type=int, default=None, help="Maximum depth to print the file tree. Default: None")
    package_group.add_argument("--exclude-cmake", action="store_true", help="Exclude CMakeLists.txt from the output")
    input_group = package_group.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--git", type=str, help="Git repo to clone")
    input_group.add_argument("--url", type=str, help="URL to download and unpack")
    input_group.add_argument("--dir", type=str, help="Directory to use")
    model_group = parser.add_argument_group("Text Generation")
    model_group.add_argument("-m", "--model", type=str, required=True, help="Path to the model to use")
    model_group.add_argument("-t", "--temperature", type=float, default=0.4, help="Temperature to use when sampling. Default: 0.4")
    model_group.add_argument("--top-p", type=float, default=0.95, help="Top p to use when sampling. Default: 0.95")
    model_group.add_argument("--top-k", type=int, default=50, help="Top k to use when sampling. Default: 50")
    model_group.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate. Default: 512")
    model_group.add_argument("--threads", type=int, help="Number of threads to use. Default: None")
    model_group.add_argument("--echo", action="store_true", help="Echo the internal prompt to stdout")
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", type=str, help="Path to the output file. Default: stdout")
    return parser.parse_args()


def clone_git_repo(git_url: str, dir: str) -> str:
    """ Clone a git repo to a directory and return the path to the repo root """
    subprocess.run(["git", "clone", git_url, dir])
    return dir

def download_and_unpack(url: str, dir: str) -> str:
    """ Download and unpack a tarball to a directory and return the path to the repo root """
    # get extension of url
    suffixes = Path(url).suffixes
    suffix = "".join(suffixes[-2:] if len(suffixes) > 1 else suffixes)

    with NamedTemporaryFile(suffix=suffix) as fp:
        urllib.request.urlretrieve(url, fp.name)
        shutil.unpack_archive(fp.name, dir)
    return dir

def get_github_versions(github_url: str) -> List[Tuple[str, str]]:
    """ query github api for all releases """
    # get the repo name
    repo_org = github_url.split("/")[-2]
    repo_name = github_url.split("/")[-1]

    # get all releases
    releases = requests.get(f"https://api.github.com/repos/{repo_org}/{repo_name}/releases").json()
    versions = [(r['tag_name'], r['tarball_url'][1:]) for r in releases]
    return versions

def get_git_versions(git_url: str) -> List[Tuple[str, str]]:
    """ Get the versions of a git repo """
    # check if github url
    parsed = urllib.parse.urlparse(git_url)
    if parsed.netloc == "github.com":
        return get_github_versions(git_url)

    # get all tags
    tags = subprocess.run(["git", "ls-remote", "--tags", git_url], capture_output=True, text=True).stdout.split("\n")
    tags = [t.split("\t")[-1].split("/")[-1] for t in tags if len(t) > 0]

    # get all branches
    branches = subprocess.run(["git", "ls-remote", "--heads", git_url], capture_output=True, text=True).stdout.split("\n")
    branches = [b.split("\t")[-1].split("/")[-1] for b in branches if len(b) > 0]

    # get all versions
    versions = []
    for tag in tags:
        if tag.startswith("v"):
            versions.append((tag, tag[1:]))
    for branch in branches:
        if branch.startswith("release-"):
            versions.append((branch, branch[8:]))
    return versions

def get_file_contents(fpath: PathLike) -> str:
    """ Get the contents of a file. """
    with open(fpath, 'r') as f:
        return f.read()

class SourceInfo:
    root: str
    file_tree: dict
    markdown_files: dict
    txt_files: dict
    build_files: dict

    def __init__(self, root: str):
        self.root = root

    def collect(self):
        # create file tree
        self.file_tree = self._get_file_tree(self.root)
        self.file_tree = self.file_tree['contents']
        if len(self.file_tree) == 1:
            self.file_tree = self.file_tree[0]['contents']

        # get the contents of all markdown files
        self.markdown_files = self._get_contents_by_pattern(self.root, '**/*.md')

        # get the contents of all txt files
        self.txt_files = {}
        for pattern in ['**/*.txt', '**/*.rst', '**/*.rtf']:
            self.txt_files.update(self._get_contents_by_pattern(self.root, pattern))

        # get the contents of all build files
        self.build_files = {}
        for pattern in ['**/makefile', '**/Makefile*', '**/CMakeLists.txt', '**/configure', '**/configure.ac', '**/setup.py', '**/requirements.txt']:
            self.build_files.update(self._get_contents_by_pattern(self.root, pattern))

    def toDict(self) -> dict:
        return {
            'source_url': None,
            'file_tree': self.file_tree,
            'markdown_files': self.markdown_files,
            'txt_files': self.txt_files,
            'build_files': self.build_files
        }

    def _get_file_tree(self, tree_root: PathLike) -> dict:
        # build a tree of all files in the repo
        tree = {"name": os.path.basename(tree_root)}

        if os.path.isdir(tree_root):
            # If the given path is a directory, recursively populate the tree
            tree["type"] = "directory"
            tree["contents"] = [self._get_file_tree(os.path.join(tree_root, item)) for item in os.listdir(tree_root)]
        else:
            # If the given path is a file, record it in the tree
            tree["type"] = "file"
        
        return tree

    def _get_contents_by_pattern(self, root: PathLike, pattern: str) -> Mapping[PathLike, str]:
        """ Get the contents of all files in the repo that match the given pattern. """
        paths_to_contents = {}
        for fpath in glob(os.path.join(root, pattern), recursive=True):
            path_without_root = os.path.relpath(fpath, root)
            paths_to_contents[path_without_root] = get_file_contents(fpath)
        return paths_to_contents


def collect_repo_info(repo_root: str) -> dict:
    """ collect metadata about a repo for the package """
    source_info = SourceInfo(repo_root)
    source_info.collect()
    return source_info.toDict()


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

def get_package_string(package_info: dict, ascii: bool = True, max_tree_depth: Optional[int] = None, exclude_cmake: bool = False) -> str:
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
    if not exclude_cmake:
        output += add_if('CMakeLists.txt', 'build_files')
    output += add_if('setup.py', 'build_files')
    output += add_if('requirements.txt', 'build_files')
    output += "spack package.py:"
    return output


def main():
    args = get_args()

    # collect repo data
    with TemporaryDirectory() as tmpdir:
        # fetch
        if args.git:
            src_str = str(args.git)
            repo_root = clone_git_repo(args.git, tmpdir)
            versions = get_git_versions(args.git)
        elif args.url:
            src_str = str(args.url)
            repo_root = download_and_unpack(args.url, tmpdir)
            versions = [('main', args.url)]
        elif args.dir:
            src_str = None
            repo_root = args.dir
            versions = [('main', args.dir)]

        # collect info
        repo_info = collect_repo_info(repo_root)
        repo_info['source_url'] = src_str
        repo_info['package_name'] = args.package_name
        repo_info['versions'] = versions

    # create prompt from repo info
    prompt = get_package_string(repo_info, ascii=True, max_tree_depth=args.max_file_tree_depth, exclude_cmake=args.exclude_cmake)

    # create model
    model = Llama(model_path=args.model, n_ctx=16384, n_threads=args.threads)

    # generate package: give prompt to model
    generated_output = model(
        prompt,
        echo=args.echo,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stream=args.output is None
    )

    # write output
    if args.output:
        outputs = [o['text'] for o in generated_output['choices']]
        output_str = '\n\n=== Package.py Output ===\n\n'.join(outputs)
        with open(args.output, 'w') as fp:
            fp.write(output_str)
    else:
        for output in generated_output:
            assert len(output['choices']) == 1, "Too many outputs"
            output_str = output['choices'][0]['text']
            print(output_str, flush=True, end='')


if __name__ == "__main__":
    main()
    

#!/usr/bin/env spack-python
""" Gather data from spack package repositories and store it in a database.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from glob import glob
import json
import multiprocessing
import os
from os import PathLike
import requests
import shutil
import subprocess
import tarfile
import tempfile
from typing import List, Mapping, Tuple
from urllib.parse import urlparse

# tpl imports
from alive_progress import alive_it
import spack
import spack.fetch_strategy as fs
from spack.package_base import preferred_version
from spack.stage import Stage


class SourceInfo:
    fetch_strategy: str
    file_tree: dict
    markdown_files: dict
    txt_files: dict
    build_files: dict

    def __init__(self, fetch_strategy: str):
        self.fetch_strategy = fetch_strategy

    def collect(self):
        with tempfile.TemporaryDirectory() as dir:
            # gather source
            #repo_root = self._clone(self.fetch_strategy, dir)
            output_queue = multiprocessing.Queue()
            output = {'repo_root': None}
            output_queue.put(output)
            proc = multiprocessing.Process(target=self._clone, args=(self.fetch_strategy, dir, output_queue))
            proc.start()

            proc.join(30)
            repo_root = output_queue.get()['repo_root']
            if proc.is_alive():
                proc.terminate()
                proc.join()
                raise RuntimeError(f"Fetching {self.fetch_strategy} timed out.")

            # create file tree
            self.file_tree = self._get_file_tree(repo_root)
            self.file_tree = self.file_tree['contents']
            if len(self.file_tree) == 1:
                self.file_tree = self.file_tree[0]['contents']

            # get the contents of all markdown files
            self.markdown_files = self._get_contents_by_pattern(repo_root, '**/*.md')

            # get the contents of all txt files
            self.txt_files = {}
            for pattern in ['**/*.txt', '**/*.rst', '**/*.rtf']:
                self.txt_files.update(self._get_contents_by_pattern(repo_root, pattern))

            # get the contents of all build files
            self.build_files = {}
            for pattern in ['**/makefile', '**/Makefile*', '**/CMakeLists.txt', '**/configure', '**/configure.ac', '**/setup.py', '**/requirements.txt']:
                self.build_files.update(self._get_contents_by_pattern(repo_root, pattern))

    def toDict(self) -> dict:
        return {
            'source_url': str(self.fetch_strategy),
            'file_tree': self.file_tree,
            'markdown_files': self.markdown_files,
            'txt_files': self.txt_files,
            'build_files': self.build_files
        }

    def _clone(self, fetch_strategy: str, dest: PathLike, output_queue: multiprocessing.Queue) -> PathLike:
        """ Download the .tar.gz file from the url """
        with Stage(fetch_strategy) as stage:
            fetch_strategy.fetch()
            fetch_strategy.expand()
            shutil.copytree(stage.source_path, dest, dirs_exist_ok=True)
            
            obj = output_queue.get()
            obj = {'repo_root': dest}
            output_queue.put(obj)
        return dest

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


def get_package_files(packages_root: PathLike) -> List[Tuple[str, PathLike]]:
    """ Get all package files in the given directory. """
    package_files = glob(os.path.join(packages_root, '*', 'package.py'))
    package_names = [os.path.basename(os.path.dirname(package_file)) for package_file in package_files]
    return list(zip(package_names, package_files))


def get_file_contents(fpath: PathLike) -> str:
    """ Get the contents of a file. """
    with open(fpath, 'r') as f:
        return f.read()


def get_package_source_info(package_name: str) -> dict:
    
    # Get the repo url from Spack
    spec = spack.spec.Spec(package_name)
    pkg_cls = spack.repo.path.get_pkg_class(spec.name)
    pkg = pkg_cls(spec)
    preferred = preferred_version(pkg) # TODO -- get all version urls
    if pkg.has_code:
        source_url = fs.for_package_version(pkg, preferred)
    else:
        return None

    # get other versions and urls
    other_versions = []
    for version in pkg.versions:
        other_versions.append((str(version), str(fs.for_package_version(pkg, version))))

    # get the package source info with Source
    src = SourceInfo(source_url)
    src.collect()
    out = src.toDict()
    out['package_name'] = package_name
    out['versions'] = other_versions
    return out


def main():
    packages_root = '/home/daniel/spack/var/spack/repos/builtin/packages'

    packages = get_package_files(packages_root)

    with open('dataset.jsonl', 'w') as fp:
        for package_name, package_file in alive_it(packages, title="collecting packages"):
            try:
                package_source_info = get_package_source_info(package_name)
                package_source_info['package_file'] = get_file_contents(package_file)
                json.dump(package_source_info, fp)
                fp.write('\n')
            except tarfile.ReadError as e:
                print(f'Error untarring package {package_name}: {e}')
            except requests.exceptions.InvalidSchema as e:
                print(f"Error fetching package {package_name}: {e}")
            except NotImplementedError as e:
                print(f"Unsupported request type. Error fetching package {package_name}: {e}")
            except Exception as e:
                print(f"Unknown error. Error fetching package {package_name}: {e}")
    


if __name__ == '__main__':
    main()

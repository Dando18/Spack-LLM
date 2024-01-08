# Spack LLM

A large language model trained to generate [Spack](github.com/spack/spack)
packages. This is mostly experimental to see how well LLMs can generate package
definitions based on repository meta-data.

There are two models, a 7B and 13B parameter model, which are CodeLlama models
fine-tuned on package meta-data. They are available on HuggingFace:
[7B](https://huggingface.co/daniellnichols/spack-llama-7b) and
[13B](https://huggingface.co/daniellnichols/spack-llama-13b).

## Download Models

The models are available in safetensor format with fp32 weights. They are also
available in the GGUF format from
[llama.cpp](https://github.com/ggerganov/llama.cpp) with 4, 6, 8, and 16 bit
weights. These can be downloaded with `download.sh`. For example:

```sh
sh download.sh 7B_8bit
```

## Install

To install the requirements in a virtual environment run:

```sh
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Generating Spack Packages

The script `create-package.py` can be used to interface with the model. An
example usage looks like:

```sh
python create-package.py --package-name my-package --git https://github.com/me/my-package --model models/spack-llama-13b/ggml-model-q8_0.gguf
```

You can also direct it to use a local directory or a URL pointing to an archive.
To show all the available options run: `python create-package.py --help`.

```
usage: create-package.py [-h] -p PACKAGE_NAME [--max-file-tree-depth MAX_FILE_TREE_DEPTH] [--exclude-cmake] (--git GIT | --url URL | --dir DIR) -m
                         MODEL [-t TEMPERATURE] [--top-p TOP_P] [--top-k TOP_K] [--max-new-tokens MAX_NEW_TOKENS] [--threads THREADS] [--echo]
                         [-o OUTPUT]

Write a spack package based on a given repo.

optional arguments:
  -h, --help            show this help message and exit

Package Data:
  -p PACKAGE_NAME, --package-name PACKAGE_NAME
                        Name of the package to create
  --max-file-tree-depth MAX_FILE_TREE_DEPTH
                        Maximum depth to print the file tree. Default: None
  --exclude-cmake       Exclude CMakeLists.txt from the output
  --git GIT             Git repo to clone
  --url URL             URL to download and unpack
  --dir DIR             Directory to use

Text Generation:
  -m MODEL, --model MODEL
                        Path to the model to use
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature to use when sampling. Default: 0.4
  --top-p TOP_P         Top p to use when sampling. Default: 0.95
  --top-k TOP_K         Top k to use when sampling. Default: 50
  --max-new-tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate. Default: 512
  --threads THREADS     Number of threads to use. Default: None
  --echo                Echo the internal prompt to stdout

Output:
  -o OUTPUT, --output OUTPUT
                        Path to the output file. Default: stdout
```

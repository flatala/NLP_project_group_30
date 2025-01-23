# RAGs for Open Domain Complex QA

## Requirements
Tested on Python 3.11.10 and 3.11.11 on Unix-based (Ubuntu) systems.

We use `uv` as a package and venv manager. If you have `pip` installed in your global Python environment, you can use that to install `uv`. To install `uv` using `pip`, run the following command:

```bash
pip install uv
```

Else, you can use `curl` to install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Otherwise, refer to [official Astral docs](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

## Installation Instructions

`uv` can handle Python versions automatically. Just create a virtual environment, and run the sync command. This will install all the required packages.

```bash
uv venv
uv sync
```


## Installing dependencies using `pip`

A `requirements.txt` file is also provided in the repository, which is a direct compilation/dump from the `uv` environment. You can use this file to install the dependencies using `pip` as well. For Unix-based systems, you can use the following commands:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Contributors
- Franciszek Latała,
- Razo van Berkel
- Marianna Louizidou
- Anish Deshpande

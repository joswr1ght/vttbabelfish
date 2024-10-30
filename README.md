# VTT Babelfish

Translate VTT subtitle files using Claude API.

Instead of translating one VTT line at a time, this script improves the translation result by parsing the VTT contents into sentences and translating each VTT chunk with the sentence context.

## Installation

Create a virtual environment and install the requirements:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

First, activate the virtual environment:

```bash
source env/bin/activate
```

Then, run the script with the following arguments:

```bash
vttbabelfish.py: Translate VTT subtitle files using Claude API

usage: vttbabelfish.py [-h] [-o OUTPUT] --api-key API_KEY [-e EXCLUDE_FILE]
                       [--debug]
                       input_file target_lang

positional arguments:
  input_file            Input VTT file path
  target_lang           Target language (2-letter or 3-letter code, or BCP-47
                        tag)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (optional)
  --api-key API_KEY     Anthropic API key
  -e EXCLUDE_FILE, --exclude-file EXCLUDE_FILE
                        File with terms not to translate
  --debug               Enable debug logging
```

![GIF of script running](VTTBabelfish.gif)


## Cost Estimate

Using the Claude API to translate a VTT file with 1000 lines from English to Spanish costs approximately $0.75.

## Authors

- Jeff McJunkin @jeffmcjunkin
- Joshua Wright @joswr1ght

VTT Babelfish is written with the assistance of genAI tools.

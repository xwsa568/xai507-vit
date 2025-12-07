# xai507-vit

## Installation

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt

Here are the execution commands in Markdown.

## Run Commands

### 1\. Run with Multi-scale PE (Proposed)

```bash
python main.py --mode multiscale
```

### 2\. Run with Baseline

```bash
python main.py --mode baseline
python main.py --pe-type rope
```

Here is the updated section for the README.

### 4\. Run All Methods

To run all available positional encoding methods sequentially for comparison:

```bash
python main.py --mode all
```

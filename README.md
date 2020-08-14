[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Chinese Word Segmentation

### Architecture
The baseline architecture is based on the work of [Ma et al.](https://aclweb.org/anthology/D18-1529). An improved transformer-based architecture is also implemented.

The dataset can be downloaded from [here](http://sighan.cs.uchicago.edu/bakeoff2005/).

### Results

|  Model | AS | CITYU | MSR | PKU |
| --- | --- | --- | --- | --- |
|  This work | 96.5 | 97.5 | 97.7 | 96.3 |
|  [Ma et al. (2018)](http://aclweb.org/anthology/D18-1529) | 96.2 | 97.2 | 97.4 | 96.1 |

### Run

The train script is in `cws/train.py`. Run this to see all the input parameters

```python
python cws/train.py -h
```

You can generate segmented sentences by running `cws/predictor.py`.

```python
python cws/predictor.py -h
```

The evaluation uses the official scripts, in `scripts/score`.

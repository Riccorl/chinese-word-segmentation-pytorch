[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Chinese Word Segmentation

### Architecture
The baseline architecture is based on the work of [Ma et al.](https://aclweb.org/anthology/D18-1529)

### Params
LR = 0.000524

### Results

#### bert-base-chinese

* pku epoch 2
    ```
    **Adam**
    === TOTAL TRUE WORDS RECALL:    0.958
    === TOTAL TEST WORDS PRECISION: 0.967
    === F MEASURE:  0.963
    0.962667
    ```


#### voidful/albert_chinese_tiny

* pku 
    ```
    simple model
    Adam
    epoch 2
    === TOTAL TRUE WORDS RECALL:    0.936
    === TOTAL TEST WORDS PRECISION: 0.937
    === F MEASURE:  0.937
    0.936667
    ```
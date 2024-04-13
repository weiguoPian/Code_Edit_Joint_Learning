# Code_Edit_Joint_Learning

Code repository of the paper "You Don’t Have to Say Where to Edit! Joint Learning to Localize and Edit Source Code".

## Dataset

The dataset can be downloaded through [link](https://drive.google.com/file/d/19n1iJcdNMHHdJgNH54LSDKC8YFcHEvL1/view?usp=sharing). Please download the dataset and put it in [./data/](./data/)

## Environment

We conduct experiments with Python 3.6.13 and Pytorch 1.5.0.

To setup the environment, please simply run

```
pip install -r requirements.txt
```

## Running

Before running, please modify "base_path=' ';" to your own path.

For CodeBERT, please run
```
sh sub_run_CodeBert.sh
```

For GraphCodeBERT, please run
```
sh sub_run_GraphCodeBert.sh
```

For CodeGPT, please run
```
sh sub_run_CodeGPT.sh
```

For PLBART, please run
```
sh sub_run_PLBART.sh
```

For CodeT5, please run
```
sh sub_run_CodeT5.sh
```

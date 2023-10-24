# HingeABL
Codes for paper "Adaptive Hinge Balance Loss for Document-Level Relation Extraction", Findings of EMNLP 2023.
## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 11.3)
* [PyTorch](http://pytorch.org/) (tested on 1.12.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.20.1)
* numpy (tested on 1.21.6)
* [spacy](https://spacy.io/) (tested on 3.3.3)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* ujson
* tqdm

## Dataset
The [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580) dataset can be downloaded following the instructions at [link](https://github.com/tonytan48/Re-DocRED/tree/main/data).

The expected structure of files is:
```
HingeABL
 |-- dataset
 |    |-- docred
 |    |    |-- train_revised.json        
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |    |-- train_annotated.json  
 |-- meta
 |    |-- rel2id.json   
 |-- scripts
 |-- checkpoint
 |-- log
```
Note: *train_annotated.json*, *rel2id.json* can be obtained through the [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset, which can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data).

## Training and Evaluation
Train the BERT model with HingeABL using the following command:

```bash
>> sh scripts/train_HingeABL.sh  # training
>> sh scripts/test_HingeABL.sh  # evaluation
```
You can select different loss functions by setting the `--loss_type` argument before training. Optional loss types include：`ATL, balance_softmax, AFL, SAT, MeanSAT, HingeABL, AML`.

Note: This code is partially based on the code of [ATLOP](https://github.com/wzhouad/ATLOP).
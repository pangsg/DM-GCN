## **Dynamic and Multi-Channel Graph Convolutional Network for Aspect-Based Sentiment Analysis**

Dataset and code for the paper: **Dynamic and Multi-Channel Graph Convolutional Network for Aspect-Based Sentiment Analysis**. Shiguan Pang, Yun Xue, Zehao Yan, Weihao Huang, Jinhui Feng. ACL Findings 2021.https://aclanthology.org/2021.findings-acl.232.pdf

## Overview

The overall framework of DM-GCN is shown in Figure 1. Inspired by AM-GCM , our key idea is that the proposed model should be available to aggregate syntactical features, semantical features and their combination features becomingly to address the issues mentioned before. For this purpose, firstly, semantic graphs are constructed by the multi-head self-attention mechanism. Secondly, syntactic graphs are transformed from the dependency tree of sentences. Then two specific convolution modules and a common convolution module are used to extract corresponding information respectively. Finally, a trainable parameter is used to fuse extracted information more suitably for our task.

![Figure1](C:\Users\辣鸡胖\Desktop\论文\Figure1.png)

## Requirement

- Python 3.6.7
- PyTorch 1.2.0
- NumPy 1.17.2
- GloVe pre-trained word vectors:
  - Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors).
  - Put [glove.840B.300d.txt](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) into the `dataset/glove/` folder.

- Rported results in the paper are under a fixed random seed, thus results might be unstable under different GPU devices or random seeds. To reproduce the reported results, you can try to train the model for several times under different random seeds such as from 0 to 50. If you want to get the trained models, please contact me by email (psg-nlp@m.scnu.edu.cn)

## Usage

Training the model:

```bash
python train.py --dataset [dataset]
```

Prepare vocabulary files for the dataset:

```bash
python prepare_vocab.py --dataset [dataset]
```

Evaluate trained model
```bash
python eval.py --model_dir [model_file path]
```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{pang-etal-2021-dynamic,
    title = "Dynamic and Multi-Channel Graph Convolutional Networks for Aspect-Based Sentiment Analysis",
    author = "Pang, Shiguan  and
      Xue, Yun  and
      Yan, Zehao  and
      Huang, Weihao  and
      Feng, Jinhui",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.232",
    doi = "10.18653/v1/2021.findings-acl.232",
    pages = "2627--2636",
}
```

## License

MIT

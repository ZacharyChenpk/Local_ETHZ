# Graph Convolutional Network

1700012764 陈智斌

This is a Pytorch implementation of Graph Convolutional Networks for the task of semi-supervised classification of nodes in a graph.



## Based article

Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks. [pdf](https://arxiv.org/pdf/1609.02907.pdf)



## Environment

Ubuntu 16.04, python 3.6.9, numpy 1.17.4, torch 1.3.1



## Data

As Kipf had done, I use the dataset splits provided by [https://github.com/kimiyoung/planetoid/tree/master/data](https://github.com/kimiyoung/planetoid/tree/master/data). To use these dataset splits, change your working directory to **myGCN**, and directly put the data files into ../GCNdata.



## Run

#### Training

```
cd myGCN
python train.py --datastr {ONE OF citeseer,cora,pubmed} --epoch 200 --hidsize 16 --lr 0.01 --dropout 0.5 --weight_decay 5e-4
```

Noted that all parameters are optional.

#### Testing

Change the line 15 in **test.py** to the trained dataset, and 

```
python test.py
```




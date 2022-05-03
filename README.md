# GCFM-pytorch
The implementation of "Fine-Grained Question-Level Deception Detection via Graph-based Learning and Cross-modal Fusion" using pytorch
## Getting Start

### Install
We provide a pytorch implementation. The following are needed:
* torch, torchvision, torchaudio
* tensorflow==1.15.5
* bert-serving-server, bert-serving-client
* librosa==0.8.0
* dgl 
* psutil

You can quick start via cloning repo and installing ``requirements.txt`` in a ``Python >= 3.7.0`` environment.
```
git clone https://github.com/zhjsss12/GCFM-pytorch
cd GCFM-pytorch
conda create -n GCFM python=3.7
conda activate GCFM
pip install -r requirements.txt
```

### Start bert server
Download [BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and unzip in ``GCFM-pytorch/``. Start bert-serving as following:
```
bert-serving-start -pooling_strategy NONE -model_dir ./uncased_L-12_H-768_A-12/ -max_seq_len=20 -num_worker=4 -port=5777 -port_out=5778
```
You can change port if needed.

## Deception Detection

# CADA

## Introduction

This repository contains the official implementation of "C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets".

## Environment

```bash
git clone https://github.com/J4ckLiu/CADA.git
cd CADA
conda create -n cada python=3.9
conda activate cada
pip install -r requirements.txt
```

## Usage

```bash
cd examples
```

### Tuning

```bash
python tune_cifar.py --seed 42 --model densenet121 
```

### Testing

```bash
python test_cifar.py --seed 42 --model densenet121 --alpha 0.01 --conformal thr
```



If you find this work useful, please cite:

```bibtex
@inproceedings{Liu2026CADA,
  title     = {C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets},
  author    = {Liu, Kangdao and Zeng, Hao and Huang, Jianguo and Zhuang, Huiping and Vong, Chi-Man and Wei, Hongxin},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence},
  year      = {2026},
}
```



---

是否希望我帮你再改成**中文版本**（保持同样简洁度、跟 OCS-ARC 一样语气）？

# This repository contains the official implementation of "C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets".

## How to Install

```bash
git clone https://github.com/J4ckLiu/CADA.git
cd CADA
conda create -n cada python=3.9
conda activate cada
pip install -r requirements.txt
```

## How to Run

```bash
cd examples
python tune_cifar.py --seed 42 --model densenet121
python test_cifar.py --seed 42 --model densenet121 --alpha 0.01 --conformal thr
```


## Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{Liu2026CADA,
  title     = {C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets},
  author    = {Liu, Kangdao and Zeng, Hao and Huang, Jianguo and Zhuang, Huiping and Vong, Chi-Man and Wei, Hongxin},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence},
  year      = {2026},
}
```


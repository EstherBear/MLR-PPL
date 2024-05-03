# Multi-label Image Recognition with Partial Positive Labels

Implementation of paper: 
- [Category-Adaptive Label Discovery and Noise Rejection for Multi-label Image Recognition with Partial Positive Labels](https://arxiv.org/abs/2211.07846)  
  Technical Report.  
  Tao Pu*, Qianru Lao*, Hefeng Wu, Tianshui Chen, Liang Lin (* equally contributed)

## Preliminary and Usage
1. Donwload [data.zip](https://1drv.ms/u/s!ArFSFaZzVErwgXMvjwsvLad6x3S5?e=hbtbTp), and unzip it.
2. Modify the lines 16-19 in config.py.
3. Create several folders (i.e., "exp/log", "exp/code", "exp/checkpoint", "exp/summary") to record experiment details.


## Usage
```bash
cd MLR-PPL
vim scripts/Noisy.sh
./scripts/Noisy.sh
```

## Citation
```
@ARTICLE{Pu2024MLR-PPL,
  author={Pu, Tao and Lao, Qianru and Wu, Hefeng and Chen, Tianshui and Tian, Ling and Liu, Jie and Lin, Liang},
  journal={IEEE Transactions on Multimedia}, 
  title={Category-Adaptive Label Discovery and Noise Rejection for Multi-label Recognition with Partial Positive Labels}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2024.3395901}}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* putao537@gmail.com
* estherbear17@gmail.com

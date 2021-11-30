## Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (DAGMM)

## About DAGMM model
[Paper](https://bzong.github.io/doc/iclr18-dagmm.pdf) Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (ICLR,2018)
[Youtube Review](https://youtu.be/byvMpGsl7cE) 발표자: DSBA연구실 이윤승 석사과정(https://github.com/yun-ss97)


## How to train and evaluate
### Train 
```python
python main.py --mode 'train' --data_root {root path} --data_name {data name}
```


### Test (All) 
```python
python main.py --mode 'test_all' --data_root {data root path} --data_name {data name}
```

### Result Analysis
```python
RUN analyze.ipynb
```
- figure 1: plot reconstructed input and original input regard to top1&bottom1 anomaly score

- figure 2: plot reconstructed input and original input per variable for top5&bottom5 anomaly score


### Training Details

**[Hyperparameter]**
| Name | Description |
| ---  |  --- | 
| **Epochs** | 10 |
| **Batch Size** | 256 |
| **Learning Rate** | 1e-4 |
| **lambda_energy** | 0.1 |
| **lambda_cov** | 0.005 |
| **number of gaussian components** | 5 |


Reference: [[code]](https://github.com/lixiangwang/DAGMM-pytorch)

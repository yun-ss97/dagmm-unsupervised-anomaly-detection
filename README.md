## Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (DAGMM)


### How to train and evaluate
#### Train 
```python
python src/main.py --mode 'train' --data_path {dataset path}
```


#### Test 
```python
python main.py --mode 'test_all_point' --data_path {dataset path}
```

#### Result Analysis
```python
RUN draw_plot.ipynb
```
- Qualitative Result: Anomaly score plot for all moment
- Quantative Result: AUROC, AUPRC, Best-F1 Score


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


### About DAGMM model
[[Paper](https://bzong.github.io/doc/iclr18-dagmm.pdf)] Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (ICLR,2018)

[[Youtube Review](https://youtu.be/byvMpGsl7cE)] 발표자: 고려대학교 산업경영공학과 DSBA 연구실 이윤승(https://github.com/yun-ss97)



Reference: [[code]](https://github.com/lixiangwang/DAGMM-pytorch)

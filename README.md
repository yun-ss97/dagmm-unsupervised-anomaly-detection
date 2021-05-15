## Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection (DAGMM)


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

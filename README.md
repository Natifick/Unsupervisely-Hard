# 🥪 Unsupervised Hard and Rare Samples Estimation by Neural Networks

This project "Machine learning" course at Skoltech. The idea is to find hard and rare samples. Prior works try to differentiate such samples by using supervised models and metrics that rely on labels. Main interest of this work is to study if it's possible to do so without labels with unsupervised training. We propose to metrics - Mean Squared Distance (MSD) and Loss Gradient Norm (LGN), study their properties on CIFAR-10 dataset, compare with supervised metrics and provide additional experiments with dataset compression and noisy images detection.

## Structure

Main experiments are located in correspodnging `/notebooks` folder :

- `ss_contrastive.ipynb`/`ss_contratsive_updated` - collection of MSD statistics, analysis, and experiment with removed 30% of data
- `additional_msd_analysis.ipynb` - analysis of MSD and labels, study of discrepancy score, comparison with clustering
- `loss_experiment.ipynb` - collection and analysis of LGN statistics
- `partially_noised_dataset.ipynb` - experiment with synethic noisy images
- `calculate_metrics.ipynb` - comparison with supervised metrics from other works

In `/RHOLossMain` is additional study of [3] comparison with MSD metrics, however, results of these experiments didn't end up being presented. In `\data` folder all collected statistics from different folders are located, so they can be used for analysis without running training of the model again.

## How to run

In order to run experiments all main experiments from report in `\notebooks` you need all standard DL/ML depdendeices (torch, matplotlib, numpy, seaborn, etc.). Also, additional libaries are used:
```
lightly=1.5.1
tslearn=0.6.3
```
To run experiments, simply go to notebook of interest and run all the cells 😃 Seeding is used for reproducablity!

## Related works
This project takes inspiration from these papers:
1. [Characterizing Datapoints via Second-Split Forgetting](https://arxiv.org/pdf/2210.15031.pdf)2
2. [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://arxiv.org/pdf/2009.10795.pdf)
3. [Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://arxiv.org/pdf/2206.07137.pdf)
4. [An Empirical Study of Example Forgetting During Deep Neural Network Learning](https://arxiv.org/pdf/1812.05159.pdf)

# amortized-inference-hci

- This repository contains the code for the CHI'23 paper, "Amortized Inference with User Simulations". This work facilitates deep user inferences, meaningly, the identification of latent user factors from data, by leveraging simulation models.

- This project speeds up inference processes to the millisecond level by training a neural proxy model. We've also prioritized the robustness of inference by estimating the full parameter distribution.

- In this repository, we demonstrate the process of training a neural inference model for three HCI simulations: menu search, point-and-click, and touchscreen typing. For each case, we provide the code for running the simulation, building and training an inference model, and evaluating its performance against empirical datasets.

- Additionally, this repository provides pre-trained inference models. This enables practitioners to perform inferences on their own data, offering the potential to directly apply our models to their research or projects.

## Amortized Inference with User Simulations (CHI'23)

- ### Making deep inferences about users quicker
	- HCI models can explain user behavior by parameter fitting.  
	- HCI models contain theoretically interesting parameters describing cognitive and physiological characteristics of users.  
	- Traditional methods face challenges due to computational costs, taking hours or even days per user.  
	- In amortization, we pretrain a neural proxy model for probabilistic inference. It increases speed of inference and achieves robustness by estimating parameter distribution.  
	- We study the efficiency and accuracy of amorization in three HCI cases: typing, menu selection, and pointing.
- ### [Project page](https://hsmoon121.github.io/projects/chi23-amortized-inference/index.html)
- ### [Presentation video (YouTube link)](https://www.youtube.com/watch?v=Lx9jKuzsASA)

## Examples of usage

- Currently, we provide three example codes:
- [01_getting_started.ipynb](https://github.com/hsmoon121/amortized-inference-hci/blob/main/examples/01_getting_started.ipynb) - Introduction of code implementation.
- [02_sbi_example.ipynb](https://github.com/hsmoon121/amortized-inference-hci/blob/main/examples/02_sbi_example.ipynb) - Demonstration of inverse modeling results using the `sbi` library.
- [03_plot_figures.ipynb](https://github.com/hsmoon121/amortized-inference-hci/blob/main/examples/03_plot_figures.ipynb) - Script for reproducing figures in our paper.

## Citation

- Please cite this paper as follows if you use this code in your research.

```
@inproceedings{moon2023amortized,
title={Amortized Inference with User Simulations},
author={Moon, Hee-Seung and Oulasvirta, Antti and Lee, Byungjoo},
booktitle={Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
year={2023},
publisher = {Association for Computing Machinery},
url = {https://hsmoon121.github.io/projects/chi23-amortized-inference},
doi = {10.1145/3544548.3581439},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

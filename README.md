# amortized-inference-hci

## Amortized Inference with User Simulations (CHI'23)

- ### Making deep inferences about users quicker
	- HCI models can explain user behavior by parameter fitting.  
	- HCI models contain theoretically interesting parameters describing cognitive and physiological characteristics of users.  
	- Traditional methods face challenges due to computational costs, taking hours or even days per user.  
	- In amortization, we pretrain a neural proxy model for probabilistic inference. It increases speed of inference and achieves robustness by estimating parameter distribution.  
	- We study the efficiency and accuracy of amorization in three HCI cases: typing, menu selection, and pointing.
- ### [Project page](https://hsmoon121.github.io/projects/chi23-amortized-inference/index.html)
- ### [Presentation video (YouTube link)](https://www.youtube.com/watch?v=Lx9jKuzsASA)

## Tutorials

- Currently, we provide three tutorial codes:
- [01_getting_started.ipynb](https://github.com/hsmoon121/amortized-inference-hci/examples/01_getting_started.ipynb) - Introduction of code implementation.
- [02_sbi_example.ipynb](https://github.com/hsmoon121/amortized-inference-hci/examples/02_sbi_example.ipynb) - Demonstration of inverse modeling results using the `sbi` library.
- [03_plot_figures.ipynb](https://github.com/hsmoon121/amortized-inference-hci/examples/03_plot_figures.ipynb) - Script for reproducing figures in our paper.

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

# Generalizing-Graph-Network-Models-for-the-Traveling-Salesman-Problem
Code for Generalizing Graph Network Models for the Traveling Salesman Problem, ICONIP23

## Usage
* run  `train.ipynb` to train the GCN model on TSP20 dataset or TSP100 dataset
* run  `test_part.ipynb` to test the model on various problem sizes 
* run  `data_generator.ipynb` to generate trainset or testset of different problem sizes, with or without labels.
* pre-trained models and datasets could be downloaded from:[link](链接：https://pan.baidu.com/s/1eRQ0cD8xK163gW8S79_ytA?pwd=uxq7)

## Dependencies
* Python >= 3.6
* Pytorch
* Numpy
* tqdm
* [pyconcorde](https://github.com/jvkersch/pyconcorde)

## Acknowledgements
* LKH-3.0.7[link](http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz)
  
* The Residual Gated Graph Convolutional Network models are based on [https://github.com/chaitjo/graph-convnet-tsp](https://github.com/chaitjo/graph-convnet-tsp)

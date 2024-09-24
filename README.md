## Inferring latent circuit from the neural trajectories of the RNN

This package allows fitting a latent circuit into the trajectories of the original continuous-time recurrent neural network (See [Langdon et. al for details](https://www.biorxiv.org/content/10.1101/2022.01.23.477431v1)). The latent circuit fitting can be viewed as a model order reduction technique.

<img src="fig/Latent Circuit diagram.svg">

The main idea is to generate the trajectories **x**(t) with a small circuit (n units), embed it with matrix **Q** into the space of the original fitted trajectories **y**(t) (N units), and minimizie the differences Distance(**Qx**(t), **y**(t)), while simultaneuously ensuring that the small circuit performs the intended behavior well:

<img src="https://latex.codecogs.com/svg.image?\text{Cost}=\min_{Q,w_{rec}}\|\mathbf{y}(t)-Q\mathbf{x}(t)\|_2&plus;c_2\|\text{output}(\mathbf{x})-\text{target}\|_2&plus;c_3\:\text{extra&space;penalties}">

Here, the optimization is simultaneuously performed on both the embedding matrix **Q** and the recurrent connectivity of a small circuit **w_{rec}**.

### How to get started

One can start from the following [example Jupyter notebook](/jupyter/inferring latent circuit from CDDM RNN.ipynb)

### Dependencies

This package relies on a [`trainRNNbrain`](link) package, make sure to download it and install it locally.







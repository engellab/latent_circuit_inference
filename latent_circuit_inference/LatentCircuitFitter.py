'''
Class which accepts LatentCircuit, an RNN_torch and a task instances and fits the RNN traces
'''
import numpy as np
import torch
from copy import deepcopy
from geoopt.optim import RiemannianAdam
from scipy.sparse.linalg import lsqr
import geoopt
from geoopt import Stiefel
from sklearn.decomposition import IncrementalPCA as iPCA

def print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss):
    gr_prfx = '\033[92m'
    gr_sfx = '\033[0m'

    train_prfx = gr_prfx if (train_loss <= min_train_loss) else ''
    train_sfx = gr_sfx if (train_loss <= min_train_loss) else ''
    if not (val_loss is None):
        val_prfx = gr_prfx if (val_loss <= min_val_loss) else ''
        val_sfx = gr_sfx if (val_loss <= min_val_loss) else ''
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx},"
              f" validation loss: {val_prfx}{np.round(val_loss, 6)}{val_sfx}")
    else:
        print(f"iteration {iter},"
              f" train loss: {train_prfx}{np.round(train_loss, 6)}{train_sfx}")


class LatentCircuitFitter():
    def __init__(self, LatentCircuit, RNN, Task, N_PCs, max_iter, tol, lr, criterion, encoding,
                 Qinitialization, penalty_type='l2', lambda_w=0.01, lambda_behavior=0.25):
        '''
        :param RNN: LatentCircuit (specific template class)
        :param RNN: pytorch RNN (specific template class)
        :param Task: task (specific template class)
        :param max_iter: maximum number of iterations
        :param tol: float, such that if the cost function reaches tol the optimization terminates
        :param criterion: function to evaluate loss
        :param lambda_ort: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param lambda_w: float, regularization on the circuit weights
        '''
        self.RNN = RNN
        self.LatentCircuit = LatentCircuit
        self.Task = Task
        self.N_PCs = N_PCs
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.criterion = criterion
        self.lambda_w = lambda_w
        self.lambda_behavior = lambda_behavior
        self.encoding = encoding
        self.Qinitialization = Qinitialization
        self.penalty_type = penalty_type
        self.min_weight = 0.0001
        self.device = self.LatentCircuit.device
        print(f"Latent Circuit Fitter device {self.device}")
        input_batch, target_batch, _ = self.Task.get_batch()
        input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.device)
        target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.device)
        # default shape of q-matrix, if there is no projection on lower dimensional space

        with torch.no_grad():
            self.RNN.sigma_rec = torch.tensor(0, device=self.device)
            self.RNN.sigma_inp = torch.tensor(0, device=self.device)
            y, predicted_output_rnn = self.RNN(input_batch)

        if (self.LatentCircuit.N > self.RNN.N):
            raise ValueError("Input dimensionality of the latent circuit has to be lower than dimensionality of RNN!")
        if (self.N_PCs > self.RNN.N) or (self.N_PCs < self.LatentCircuit.N):
            raise ValueError("N_PC parameter has to be greater than the number of latent nodes but <= number of nodes in the RNN!")

        self.set_projection()
        self.q_shape = (self.N_PCs, self.LatentCircuit.N) # - update the shape of q
        y = self.projection(y)

        if self.Qinitialization:
            self.initialize_Q(y, input_batch, target_batch)
        else:
            q = torch.Tensor(np.random.randn(*self.q_shape)).to(self.LatentCircuit.device)
            U, s, Vh = torch.linalg.svd(q)
            k = int(torch.minimum(torch.tensor(U.shape[-1]), torch.tensor(Vh.shape[0])))
            q = (U[:, :k] @ Vh).to(self.LatentCircuit.device)
            manifold = geoopt.Stiefel()
            self.q = geoopt.ManifoldParameter(q, manifold=manifold).to(self.LatentCircuit.device)

        params = list(self.LatentCircuit.parameters()) + [self.q]
        self.optimizer = RiemannianAdam(params, lr=self.lr)

    def set_projection(self):
        print("setting projection of RNN traces on the lower subspace")
        # do the PCA on RNN traces to lower the dimensionality!

        input_batch, target_batch, conditions_batch = self.Task.get_batch()
        sigma_rec = deepcopy(self.RNN.sigma_rec)
        sigma_inp = deepcopy(self.RNN.sigma_inp)
        self.RNN.sigma_rec = self.RNN.sigma_inp = torch.tensor(0, device=self.device)
        y, predicted_output_rnn = self.RNN(torch.from_numpy(input_batch.astype("float32")).to(self.RNN.device))

        Y = y.reshape(self.RNN.N, -1).detach().cpu().numpy().T
        pca = iPCA(n_components=self.N_PCs, batch_size=1000)
        pca.partial_fit(Y)

        self.Pr = torch.Tensor(pca.components_).to(self.RNN.device)
        self.RNN.sigma_rec = torch.tensor(sigma_rec, device=self.device)
        self.RNN.sigma_inp = torch.tensor(sigma_inp, device=self.device)
        def projection_function(y):
            return torch.einsum("ij, jkl->ikl", self.Pr, y)

        self.projection = projection_function
        return None

    def train_step(self, input, y, predicted_output_RNN):
        self.LatentCircuit.train()
        x, predicted_output_lc = self.LatentCircuit(input, w_noise=True)
        penalty_matching_dict = {"l1": 1, "l2": 2}
        p = penalty_matching_dict[self.penalty_type]

        behavioral_loss = self.lambda_behavior * self.criterion(predicted_output_lc, predicted_output_RNN)
        regularization_penalty = self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.recurrent_layer.weight, p)) + \
                                 self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.input_layer.weight, p)) + \
                                 self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.output_layer.weight, p))
        if self.encoding:
            x_emb = torch.einsum("ji, ikp->jkp", self.q, x)
            dynamics_matching_loss = self.criterion(x_emb, self.projection(y))
        else:
            y_pr = torch.einsum("ij, ikp->jkp", self.q, self.projection(y))
            dynamics_matching_loss = self.criterion(x, y_pr)

        loss = dynamics_matching_loss + behavioral_loss + regularization_penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # apply constraints
        self.apply_constraints()
        return loss.item()

    def eval_step(self, input, y, predicted_output_rnn):
        with torch.no_grad():
            self.LatentCircuit.eval()
            x, predicted_output_lc = self.LatentCircuit(input, w_noise=False)
            val_loss = self.criterion(predicted_output_lc, predicted_output_rnn) / torch.var(predicted_output_rnn, unbiased=True)
            return float(val_loss.cpu().numpy())

    def apply_constraints(self):
        # apply constraint with the masks controlling which connections are not allowed
        self.LatentCircuit.recurrent_layer.weight.data *= self.LatentCircuit.rec_connectivity_mask.to(self.device)
        self.LatentCircuit.output_layer.weight.data *= self.LatentCircuit.out_connectivity_mask.to(self.device)
        self.LatentCircuit.input_layer.weight.data *= self.LatentCircuit.inp_connectivity_mask.to(self.device)

        # keeping the W_inp, W_out positive
        self.LatentCircuit.input_layer.weight.data[torch.tensor(self.LatentCircuit.input_layer.weight.data <= 0)] = self.min_weight
        self.LatentCircuit.output_layer.weight.data[torch.tensor(self.LatentCircuit.output_layer.weight.data <= 0)] = self.min_weight

        # # Dale's law
        # if not (self.LatentCircuit.dale_mask is None):
        #     # Dale mask is a vector of +1 and -1. If the sign of w_ij * Dale_j < 0 that means the constraint is violated
        #     incorrect_rec_vals_mask = ((self.LatentCircuit.recurrent_layer.data * self.LatentCircuit.dale_mask) <= 0).to(self.device)
        #     self.LatentCircuit.recurrent_layer.data[incorrect_rec_vals_mask] = self.min_weight
        #
        #     incorrect_out_vals_mask = ((self.LatentCircuit.output_layer.data * self.LatentCircuit.dale_mask) <= 0).to(self.device)
        #     self.LatentCircuit.output_layer.data[incorrect_out_vals_mask] = self.min_weight

        return None

    def run_training(self):
        train_losses = []
        val_losses = []

        min_train_loss = np.inf
        min_val_loss = np.inf
        best_lc_params = deepcopy(self.LatentCircuit.get_params())

        # the same batch is used over and over again!
        input_batch, target_batch, _ = self.Task.get_batch()
        input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.LatentCircuit.device)
        target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.LatentCircuit.device)
        input_val = deepcopy(input_batch)

        with torch.no_grad():
            self.RNN.sigma_rec = torch.tensor(0, device=self.device)
            self.RNN.sigma_inp = torch.tensor(0, device=self.device)
            y, predicted_output_rnn = self.RNN(input_batch)

        for iter in range(self.max_iter):
            train_loss = self.train_step(input_batch, y.detach(), predicted_output_rnn.detach()) #TODO : ???
            # validation
            val_loss = self.eval_step(input_val, y, predicted_output_rnn)
            # keeping track of train and valid losses and printing
            print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                best_lc_params = deepcopy(self.get_params())
            if train_loss <= min_train_loss: min_train_loss = train_loss

            if val_loss <= self.tol:
                self.LatentCircuit.set_params(best_lc_params)
                return self.LatentCircuit, train_losses, val_losses, best_lc_params

        self.LatentCircuit.set_params(best_lc_params)
        return self.LatentCircuit, train_losses, val_losses, best_lc_params

    def initialize_Q(self, y, input_batch, target_batch):
        # get better initialization of Q by initializing it as
        # axes hosting the input and output variables
        # matrix of traces

        # decoding perspective
        A = y.detach().cpu().numpy().reshape(y.shape[0], -1).T
        # columns of inputs and outputs
        b = np.hstack([np.vstack([input_batch[:, :, i].detach().cpu().numpy(), target_batch[:, :, i].detach().cpu().numpy()]) for i in range(input_batch.shape[-1])]).T
        C = np.zeros((self.N_PCs, b.shape[1]))

        for i in range(C.shape[1]):
            C[:, i] = lsqr(A, b[:, i], damp=100)[0]
        q = torch.Tensor(deepcopy(C)).to(self.LatentCircuit.device)
        U, s, Vh = torch.linalg.svd(q)
        k = int(torch.minimum(torch.tensor(U.shape[-1]), torch.tensor(Vh.shape[0])))
        q = (U[:,:k] @ Vh).to(self.LatentCircuit.device)
        manifold = geoopt.Stiefel()
        self.q = geoopt.ManifoldParameter(q, manifold=manifold).to(self.LatentCircuit.device)

        return None


    def get_params(self):
        params = self.LatentCircuit.get_params()
        params["U"] = deepcopy(self.Pr.detach().cpu().numpy())
        params["q"] = deepcopy(self.q.detach().cpu().numpy())
        return params

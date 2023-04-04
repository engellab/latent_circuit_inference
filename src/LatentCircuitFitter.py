'''
Class which accepts LatentCircuit, an RNN_torch and a task instances and fits the RNN traces
'''
import geoopt
import numpy as np
import torch
from copy import deepcopy
from scipy.sparse.linalg import lsqr
from geoopt import Stiefel


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
    def __init__(self, LatentCircuit, RNN, Task, max_iter, tol, criterion, optimizer, lambda_w, Qinitialization):
        '''
        :param RNN: LatentCircuit (specific template class)
        :param RNN: pytorch RNN (specific template class)
        :param Task: task (specific template class)
        :param max_iter: maximum number of iterations
        :param tol: float, such that if the cost function reaches tol the optimization terminates
        :param criterion: function to evaluate loss
        :param optimizer: pytorch optimizer (Adam, SGD, etc.)
        :param lambda_ort: float, regularization softly imposing a pair-wise orthogonality
         on columns of W_inp and rows of W_out
        :param lambda_w: float, regularization on the circuit weights
        '''
        self.RNN = RNN
        self.LatentCircuit = LatentCircuit
        self.Task = Task
        self.max_iter = max_iter
        self.tol = tol
        self.criterion = criterion
        self.optimizer = optimizer
        self.lambda_w = lambda_w
        self.Qinitialization = Qinitialization
        self.device = self.LatentCircuit.device
        if (self.LatentCircuit.N < self.RNN.N):
            self.do_pca = True
        elif (self.LatentCircuit.N == self.RNN.N):
            self.do_pca = False
        else:
            raise ValueError("Input dimensionality of the latent circuit has to be lower than dimensionality of RNN!")

    def train_step(self, input, Uy, predicted_output_rnn):
        self.LatentCircuit.train()
        Qx, predicted_output_lc = self.LatentCircuit(input)

        # mean_out = predicted_output_rnn.reshape(-1, predicted_output_rnn.shape[-1])
        # var_out = torch.mean(torch.mean((mean_out - torch.mean(mean_out, dim=0)) ** 2, dim=0))
        # mean_traces = Uy.reshape(-1, Uy.shape[-1])
        # var_traces = torch.mean(torch.mean((mean_traces - torch.mean(mean_traces, dim=0)) ** 2, dim=0))
        # # / var_out + \ #/ var_traces + \

        loss = self.criterion(predicted_output_lc, predicted_output_rnn) / torch.var(predicted_output_rnn, unbiased=False) + \
               self.criterion(Qx, Uy) / torch.var(Uy, unbiased=False) + \
               self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.recurrent_layer.weight, 2)) + \
               self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.input_layer.weight, 2)) + \
               self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.output_layer.weight, 2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # projection back to constrained space
        device = self.LatentCircuit.device
        self.LatentCircuit.input_layer.weight.data *= self.LatentCircuit.inp_connectivity_mask.to(device)
        self.LatentCircuit.recurrent_layer.weight.data *= self.LatentCircuit.rec_connectivity_mask.to(device)
        self.LatentCircuit.output_layer.weight.data *= self.LatentCircuit.out_connectivity_mask.to(device)
        # self.LatentCircuit.make_orthonormal()
        return loss.item()

    def eval_step(self, input, Uy, predicted_output_rnn):
        with torch.no_grad():
            self.LatentCircuit.eval()
            Qx, predicted_output_lc = self.LatentCircuit(input, w_noise=False)
            val_loss = self.criterion(predicted_output_lc, predicted_output_rnn) + \
                       self.criterion(Qx, Uy) + \
                       self.lambda_w * torch.mean(torch.pow(self.LatentCircuit.recurrent_layer.weight, 2))
            return float(val_loss.cpu().numpy())

    def run_training(self):
        train_losses = []
        val_losses = []
        # runs all trajectories without noise in dynamics and input
        if self.do_pca: self.LatentCircuit.set_projection(self.RNN, self.Task)
        min_train_loss = np.inf
        min_val_loss = np.inf
        best_lc_params = deepcopy(self.LatentCircuit.get_params())

        input_batch, target_batch, _ = self.Task.get_batch()
        input_batch = torch.from_numpy(input_batch.astype("float32")).to(self.LatentCircuit.device)
        target_batch = torch.from_numpy(target_batch.astype("float32")).to(self.LatentCircuit.device)
        input_val = deepcopy(input_batch)
        with torch.no_grad():
            self.RNN.sigma_rec = torch.tensor(0, device=self.device)
            self.RNN.sigma_inp = torch.tensor(0, device=self.device)
            y, predicted_output_rnn = self.RNN(input_batch)
        if self.do_pca: y = torch.swapaxes(self.LatentCircuit.projection(torch.swapaxes(y, 0, -1)), 0, -1)
        if self.Qinitialization: self.initialize_Q(y, input_batch, target_batch)

        for iter in range(self.max_iter):
            train_loss = self.train_step(input_batch, y.detach(), predicted_output_rnn.detach())
            # validation
            val_loss = self.eval_step(input_val, y, predicted_output_rnn)
            # keeping track of train and valid losses and printing
            print_iteration_info(iter, train_loss, min_train_loss, val_loss, min_val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                best_lc_params = deepcopy(self.LatentCircuit.get_params())
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
        A = y.detach().cpu().numpy().reshape(y.shape[0], -1).T
        # columns of inputs and outputs
        b = np.hstack([input_batch.reshape(input_batch.shape[0], -1).T.detach().cpu().numpy(),
                       target_batch.reshape(target_batch.shape[0], -1).detach().cpu().numpy().T])
        C = np.zeros((self.LatentCircuit.N, b.shape[1]))
        for i in range(C.shape[1]):
            C[:, i] = lsqr(A, b[:, i], damp=100)[0]
        q = deepcopy(C)
        q = self.LatentCircuit.make_orthonormal(torch.Tensor(q).to(self.LatentCircuit.device))
        manifold = geoopt.Stiefel()
        self.LatentCircuit.q = geoopt.ManifoldParameter(q, manifold=manifold).to(self.LatentCircuit.device)
        return None


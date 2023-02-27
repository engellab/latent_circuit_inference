import sys
import os
sys.path.insert(0, "../")
import torch
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
from copy import deepcopy
from sklearn.decomposition import IncrementalPCA as iPCA

class LatentCircuit(torch.nn.Module):
    '''
    '''
    def __init__(self, n, N, W_inp, W_out,
                 activation,
                 dt=1, tau=10,
                 num_inputs=6, num_outputs=2,
                 sigma_rec=0.03, sigma_inp=0.03,
                 rec_connectivity_mask=None,
                 inp_connectivity_mask=None,
                 out_connectivity_mask=None,
                 random_generator=None,
                 device=None):
        super(LatentCircuit, self).__init__()
        if (device is None):
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        # self.device = torch.device('mps')
        print(f"Using {self.device} for Latent Circuit!")

        self.n = n
        self.N = N
        self.activation = activation
        self.tau = tau
        self.dt = dt
        self.alpha = torch.tensor((dt/tau), device=self.device)
        self.sigma_rec = torch.tensor(sigma_rec, device=self.device)
        self.sigma_inp = torch.tensor(sigma_inp, device=self.device)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.random_generator = random_generator
        self.input_layer = (torch.nn.Linear(self.num_inputs, self.n, bias=False, device=self.device))
        self.recurrent_layer = torch.nn.Linear(self.n, self.n, bias=False, device=self.device)
        self.output_layer = torch.nn.Linear(self.n, self.num_outputs, bias=False, device=self.device)

        self.input_layer.weight.data = W_inp
        self.output_layer.weight.data = W_out
        self.recurrent_layer.weight.data = torch.zeros((self.n, self.n)).float().to(device=self.device)
        self.inp_connectivity_mask = torch.Tensor(inp_connectivity_mask).to(self.device)
        self.rec_connectivity_mask = torch.Tensor(rec_connectivity_mask).to(self.device)
        self.out_connectivity_mask = torch.Tensor(out_connectivity_mask).to(self.device)
        self.input_layer.weight.data *= (self.inp_connectivity_mask)
        self.recurrent_layer.weight.data *= (self.rec_connectivity_mask)
        self.output_layer.weight.data *= (self.out_connectivity_mask)
        self.x = torch.zeros(self.n, device=self.device)

        # # # if you want to have w_inp and w_out frozen
        # for param in self.input_layer.parameters():
        #     param.requires_grad = False
        # for param in self.output_layer.parameters():
        #     param.requires_grad = False

        q = torch.rand(self.n, self.N, device=self.device)
        q = self.make_orthonormal(q)
        self.q = torch.nn.Parameter(q, requires_grad=True)

    #     self.A = torch.nn.Parameter(torch.rand(self.N, self.N, device=torch.device('cpu'), generator=self.random_generator), requires_grad=True).to(self.device)
    #     self.Q = (torch.eye(self.N, device=self.device) - (self.A - self.A.t()) / 2) @ torch.inverse(torch.eye(self.N, device=self.device) + (self.A - self.A.t()) / 2)
    #     self.q = self.Q[:self.n, :]
    #     self.q = self.q.to(device=self.device)
    #
    # def make_orthonormal(self):
    #     skew = (self.A - self.A.T) / 2
    #     eye = torch.eye(self.N, device=self.device)
    #     o = (eye - skew) @ torch.inverse(eye + skew)
    #     self.q = o[:self.n, :]

    def make_orthonormal(self, q=None):
        if q is None:
            on_self = True
            q = self.q
        else:
            on_self = False
        U, s, Vh = torch.linalg.svd(q)
        S = torch.zeros((U.shape[-1], Vh.shape[0]), device=self.device)
        k = int(torch.minimum(torch.tensor(U.shape[-1]), torch.tensor(Vh.shape[0])))
        S[:k, :k] = torch.eye(k, device=self.device)
        if on_self is True:
            self.q.data = (U @ S @ Vh).to(self.device)
        else:
            return U @ S @ Vh

    def set_projection(self, RNN, Task):
        # do the PCA on RNN traces to lower the dimensionality!
        input_batch, target_batch, conditions_batch = Task.get_batch()
        sigma_rec = deepcopy(RNN.sigma_rec)
        sigma_inp = deepcopy(RNN.sigma_inp)
        RNN.sigma_rec = RNN.sigma_inp = torch.tensor(0, device=self.device)
        y, predicted_output_rnn = RNN(torch.from_numpy(input_batch.astype("float32")).to(RNN.device))

        Y = y.reshape(RNN.N, -1).detach().cpu().numpy().T
        pca = iPCA(n_components=self.N, batch_size=1000)
        pca.partial_fit(Y)

        self.projection = torch.nn.Linear(RNN.N, self.N, bias=False).to(self.device)
        self.projection.weight.data = torch.Tensor(pca.components_).to(RNN.device)
        for param in self.projection.parameters():
            param.requires_grad = False
        RNN.sigma_rec = torch.tensor(sigma_rec, device=self.device)
        RNN.sigma_inp = torch.tensor(sigma_inp, device=self.device)
        return None

    def forward(self, u, w_noise=True):
        '''
        :param u: array of input vectors (self.input_size, T_steps, batch_size)
        :param w_noise: bool, pass forward with or without noise
        :return: the full history of the internal variables and the outputs
        '''
        T_steps = u.shape[1]
        batch_size = u.shape[-1]
        states = torch.zeros(self.n, 1, batch_size, device=self.device)
        states[:, 0, :] = deepcopy(self.x).reshape(-1, 1).repeat(1, batch_size)
        rec_noise = torch.zeros(self.n, T_steps, batch_size, device=self.device)
        inp_noise = torch.zeros(self.num_inputs, T_steps, batch_size, device=self.device)
        if w_noise:
            rec_noise = torch.sqrt((2 / self.alpha) * self.sigma_rec ** 2) \
                        * torch.randn(*rec_noise.shape, generator=self.random_generator, device=self.device)
            inp_noise = torch.sqrt((2 / self.alpha) * self.sigma_inp ** 2) \
                        * torch.randn(*inp_noise.shape, generator=self.random_generator, device=self.device)
        # passing through layers require batch-first shape!
        # that's why we need to reshape the inputs and states!
        states = torch.swapaxes(states, 0, -1)
        u = torch.swapaxes(u, 0, -1).to(self.device)
        rec_noise = torch.swapaxes(rec_noise, 0, -1).to(self.device)
        inp_noise = torch.swapaxes(inp_noise, 0, -1).to(self.device)
        for i in range(T_steps - 1):
            state_new = (1 - self.alpha) * states[:, i, :] + \
                        self.alpha * (
                            self.activation(
                                self.recurrent_layer(states[:, i, :]) +
                                self.input_layer(u[:, i, :] + inp_noise[:, i, :])) +
                                rec_noise[:, i, :]
                        )
            states = torch.cat((states, state_new.unsqueeze_(1)), 1)
        outputs = torch.swapaxes(self.output_layer(states), 0, -1)
        states_embedded = states @ self.q
        states_embedded = torch.swapaxes(states_embedded, 0, -1)
        return states_embedded, outputs

    def get_params(self):
        '''
        Save crucial parameters of the RNN as numpy arrays
        :return: parameter dictionary containing connectivity parameters, initial conditions,
         number of nodes, dt and tau
        '''
        param_dict = {}
        W_out = deepcopy(self.output_layer.weight.data.cpu().detach().numpy())
        W_rec = deepcopy(self.recurrent_layer.weight.data.cpu().detach().numpy())
        W_inp = deepcopy(self.input_layer.weight.data.cpu().detach().numpy())
        try:
            param_dict["U"] = deepcopy(self.projection.weight.data.cpu().detach().numpy())
        except:
            param_dict["U"] = None
        param_dict["q"] = deepcopy(self.q.cpu().detach().numpy())
        # param_dict["A"] = deepcopy(self.A.cpu().detach().numpy())
        # Q = (torch.eye(self.N, device=self.device) - (self.A - self.A.t()) / 2) @ torch.inverse(
        #     torch.eye(self.N, device=self.device) + (self.A - self.A.t()) / 2)
        # q = Q[:self.n, :].to(device=self.device)
        # param_dict["q"] = deepcopy(q.cpu().detach().numpy())
        param_dict["W_out"] = W_out
        param_dict["W_inp"] = W_inp
        param_dict["W_rec"] = W_rec
        param_dict["n"] = self.n
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        return param_dict

    def set_params(self, params):
        self.output_layer.weight.data = torch.from_numpy(params["W_out"]).to(self.device)
        self.input_layer.weight.data = torch.from_numpy(params["W_inp"]).to(self.device)
        self.recurrent_layer.weight.data = torch.from_numpy(params["W_rec"]).to(self.device)
        self.q = torch.nn.Parameter(torch.from_numpy(params["q"]).to(self.device), requires_grad=True).to(self.device)

        # self.A = torch.nn.Parameter(torch.from_numpy(params["A"]).to(self.device), requires_grad=True).to(self.device)
        # self.Q = (torch.eye(self.N, device=self.device) - (self.A - self.A.t()) / 2) @ torch.inverse(
        #     torch.eye(self.N, device=self.device) + (self.A - self.A.t()) / 2)
        # self.q = self.Q[:self.n, :].to(device=self.device)

        try:
            self.projection.weight.data = torch.from_numpy(params["U"]).to(self.device)
        except:
            pass
        return None

# if __name__ == '__main__':
#     n = 8
#     N = 16
#     W_inp = np.zeros((n, 6))
#     W_inp[:6, :6] = np.eye(6)
#     W_out = np.zeros((2, 8))
#     W_out[0, 6] = 1.0
#     W_out[1, 7] = 1.0
#     W_inp = torch.Tensor(W_inp)
#     W_out = torch.Tensor(W_out)
#     rec_connectivity_mask = torch.ones((n, n))
#     activation = lambda x: torch.maximum(torch.tensor(0), x)
#     lc = LatentCircuit(n,N, W_inp, W_out, rec_connectivity_mask, activation)
#
#     u = torch.ones((6, 105, 11))
#     statesQ, outputs = lc.forward(u)
#     print(statesQ.shape)



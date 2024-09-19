import sys
import os
sys.path.insert(0, "../")
import torch
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
from copy import deepcopy

class LatentCircuit(torch.nn.Module):
    '''
    '''
    def __init__(self,
                 N,
                 activation_name,
                 activation_slope,
                 w_inp, w_out,
                 w_rec = None,
                 dt=1, tau=10,
                 num_inputs=6, num_outputs=2,
                 sigma_rec=0.03, sigma_inp=0.03,
                 rec_connectivity_mask=None,
                 inp_connectivity_mask=None,
                 out_connectivity_mask=None,
                 dale_mask = None,
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
        print(f"Using {self.device} for Latent Circuit!")

        self.N = N
        self.activation_slope = torch.tensor(activation_slope).to(self.device)
        self.activation_name = activation_name
        if activation_name == 'relu':
            self.activation = lambda x: torch.maximum(torch.tensor(0.), self.activation_slope * x)
        elif activation_name == 'tanh':
            self.activation = lambda x: torch.tanh(self.activation_slope * x)
        elif activation_name == 'sigmoid':
            self.activation = lambda x: torch.sigmoid(self.activation_slope * x)
        elif activation_name == 'softplus':
            self.activation = lambda x: torch.nn.Softplus(beta=self.activation_slope)(x)

        self.tau = tau
        self.dt = dt
        self.alpha = torch.tensor((dt/tau), device=self.device)
        self.sigma_rec = torch.tensor(sigma_rec, device=self.device)
        self.sigma_inp = torch.tensor(sigma_inp, device=self.device)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.random_generator = random_generator
        self.input_layer = (torch.nn.Linear(self.num_inputs, self.N, bias=False, device=self.device))
        self.recurrent_layer = torch.nn.Linear(self.N, self.N, bias=False, device=self.device)
        self.output_layer = torch.nn.Linear(self.N, self.num_outputs, bias=False, device=self.device)

        self.input_layer.weight.data = torch.from_numpy(np.array(w_inp)).float().to(device=self.device)
        self.output_layer.weight.data = torch.from_numpy(np.array(w_out)).float().to(device=self.device)
        if not (w_rec is None):
            self.recurrent_layer.weight.data = torch.from_numpy(np.array(w_out)).to(device=self.device)
        else:
            self.recurrent_layer.weight.data = torch.zeros((self.N, self.N)).float().to(device=self.device)
        self.inp_connectivity_mask = torch.Tensor(inp_connectivity_mask).to(self.device)
        self.rec_connectivity_mask = torch.Tensor(rec_connectivity_mask).to(self.device)
        self.out_connectivity_mask = torch.Tensor(out_connectivity_mask).to(self.device)

        self.input_layer.weight.data *= (self.inp_connectivity_mask)
        self.recurrent_layer.weight.data *= (self.rec_connectivity_mask)
        self.output_layer.weight.data *= (self.out_connectivity_mask)
        if dale_mask is None:
            self.dale_mask = None
        else:
            self.dale_mask = torch.from_numpy(dale_mask).to(torch.float32).to(self.device)

        self.x = torch.zeros(self.N, device=self.device)

    def forward(self, u, w_noise=True):
        '''
        :param u: array of input vectors (self.input_size, T_steps, batch_size)
        :param w_noise: bool, pass forward with or without noise
        :return: the full history of the internal variables and the outputs
        '''
        T_steps = u.shape[1]
        batch_size = u.shape[-1]
        states = torch.zeros(self.N, 1, batch_size, device=self.device)
        states[:, 0, :] = deepcopy(self.x).reshape(-1, 1).repeat(1, batch_size)
        rec_noise = torch.zeros(self.N, T_steps, batch_size, device=self.device)
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
        states = torch.swapaxes(states, 0, -1)
        return states, outputs

    def get_params(self):
        '''
        Save crucial parameters of the RNN as numpy arrays
        :return: parameter dictionary containing connectivity parameters, initial conditions,
         number of nodes, dt and tau
        '''
        param_dict = {}
        param_dict["activation_name"] = self.activation_name
        param_dict["activation_slope"] = float(self.activation_slope.cpu().detach().numpy())
        w_out = deepcopy(self.output_layer.weight.data.cpu().detach().numpy())
        w_rec = deepcopy(self.recurrent_layer.weight.data.cpu().detach().numpy())
        w_inp = deepcopy(self.input_layer.weight.data.cpu().detach().numpy())
        if self.recurrent_layer.bias is None:
            pass
        else:
            b_rec = deepcopy(self.recurrent_layer.bias.data.cpu().detach().numpy())
            param_dict["b_rec"] = b_rec
        param_dict["W_out"] = w_out
        param_dict["W_inp"] = w_inp
        param_dict["W_rec"] = w_rec
        param_dict["N"] = self.N
        param_dict["dt"] = self.dt
        param_dict["tau"] = self.tau
        return param_dict

    def set_params(self, params):
        self.output_layer.weight.data = torch.from_numpy(params["W_out"].astype("float32")).to(self.device)
        self.input_layer.weight.data = torch.from_numpy(params["W_inp"].astype("float32")).to(self.device)
        self.recurrent_layer.weight.data = torch.from_numpy(params["W_rec"].astype("float32")).to(self.device)
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



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from GCN_layers import GraphConvolution
from control_constrints import lqr


class ControlAttention(nn.Module):
    def __init__(self, input_size):
        super(ControlAttention, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input, R):
        weight = self.fc(R)
        input = input.view(input.size(0), 1)
        weight = (input * weight - (input * weight).max()).exp()
        return weight


class PDECell(nn.Module):
    def __init__(self, wiring, in_features=None, input_mapping="affine", output_mapping="affine", ode_unfolds=6, epsilon=1e-8, density=0.4):
        super(PDECell, self).__init__()

        if in_features is not None:
            wiring.build((None, in_features))
        if not wiring.is_built():
            raise ValueError("Wiring error! Unknown number of input features. Please pass 'in_features' or call 'wiring.build()'.")

        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
            "laplacian": (0.001, 1),
            "control": (0.001, 4),
        }

        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._density = density

        self.gc1 = GraphConvolution(1, 1)
        self.attention = ControlAttention(in_features)

        self._allocate_parameters()

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def density(self):
        return self._density

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.sensory_adjacency_matrix))

    def add_weight(self, name, init_value):
        param = nn.Parameter(init_value)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        return torch.ones(shape) * minval if minval == maxval else torch.rand(*shape) * (maxval - minval) + minval

    def sprandsym(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        density = 0.1
        X = torch.randn(*shape) * (torch.rand_like(torch.randn(*shape)) < density).float()
        A = torch.mm(X, X.t())
        A += torch.diag(0.01 * torch.ones(shape[0]))
        A = (A - A.min()) / (A.max() - A.min())
        A = A * (maxval - minval) + minval
        torch.linalg.cholesky(A)
        return A

    def _allocate_parameters(self):
        self._params = {}
        self._params["laplacian"] = self.add_weight("laplacian", self.sprandsym((self.state_size, self.state_size), "laplacian"))
        self._params["control"] = self.add_weight("control", self._get_init_value((self.state_size,), "control"))
        self._params["gleak"] = self.add_weight("gleak", self._get_init_value((self.state_size,), "gleak"))
        self._params["vleak"] = self.add_weight("vleak", self._get_init_value((self.state_size,), "vleak"))
        self._params["cm"] = self.add_weight("cm", self._get_init_value((self.state_size,), "cm"))
        self._params["sigma"] = self.add_weight("sigma", self._get_init_value((self.state_size, self.state_size), "sigma"))
        self._params["mu"] = self.add_weight("mu", self._get_init_value((self.state_size, self.state_size), "mu"))
        self._params["w"] = self.add_weight("w", self._get_init_value((self.state_size, self.state_size), "w"))
        self._params["erev"] = self.add_weight("erev", torch.Tensor(self._wiring.erev_initializer()))
        self._params["sensory_sigma"] = self.add_weight("sensory_sigma", self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma"))
        self._params["sensory_mu"] = self.add_weight("sensory_mu", self._get_init_value((self.sensory_size, self.state_size), "sensory_mu"))
        self._params["sensory_w"] = self.add_weight("sensory_w", self._get_init_value((self.sensory_size, self.state_size), "sensory_w"))
        self._params["sensory_erev"] = self.add_weight("sensory_erev", torch.Tensor(self._wiring.sensory_erev_initializer()))
        self._params["sparsity_mask"] = torch.Tensor(np.abs(self._wiring.adjacency_matrix))
        self._params["sensory_sparsity_mask"] = torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix))

        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight("input_w", torch.ones((self.sensory_size,)))
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight("input_b", torch.zeros((self.sensory_size,)))
        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight("output_w", torch.ones((self.motor_size,)))
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight("output_b", torch.zeros((self.motor_size,)))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1)
        return torch.sigmoid(sigma * (v_pre - mu))

    def _ode_solver(self, inputs, state, elapsed_time, adj):
        v_pre = state
        shape = inputs.size(0)
        Q, R = torch.eye(shape), torch.eye(shape)
        Q_att = self.attention(inputs, Q).view(shape)
        R_att = self.attention(inputs, R).view(shape)
        Q, R = torch.diag(Q_att), torch.diag(R_att)
        K, _ = lqr(adj, self._params["laplacian"], Q, R)
        control = self._params["control"]
        control = (control - torch.mean((K * inputs), dim=1)) @ self._params["laplacian"]

        sensory_w_act = self._params["sensory_w"] * self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"])
        sensory_w_act *= self._params["sensory_sparsity_mask"].to(sensory_w_act.device)
        sensory_rev_act = sensory_w_act * self._params["sensory_erev"]
        w_num_sensory = torch.sum(sensory_rev_act, dim=1)
        w_den_sensory = torch.sum(sensory_w_act, dim=1)

        inputs = inputs.view(shape, 1)
        diffusion = F.relu(self.gc1(inputs, adj))
        for _ in range(3):
            diffusion = F.relu(self.gc1(diffusion, adj))
        diffusion = diffusion.view(shape)

        cm_t = diffusion + self._params["cm"] / (elapsed_time / self._ode_unfolds) + control

        for _ in range(self._ode_unfolds):
            w_act = self._params["w"] * self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            w_act *= self._params["sparsity_mask"].to(w_act.device)
            rev_act = w_act * self._params["erev"]
            w_num = torch.sum(rev_act, dim=1) + w_num_sensory
            w_den = torch.sum(w_act, dim=1) + w_den_sensory

            numerator = cm_t * v_pre + self._params["gleak"] * self._params["vleak"] + w_num
            denominator = cm_t + self._params["gleak"] + w_den

            v_pre = numerator / (denominator + self._epsilon)
        return v_pre, self._params["laplacian"], control + self._params["control"]

    def _map_inputs(self, inputs):
        if self._input_mapping in ["affine", "linear"]:
            inputs *= self._params["input_w"]
        if self._input_mapping == "affine":
            inputs += self._params["input_b"]
        return inputs

    def _map_outputs(self, state):
        output = state[:, :self.motor_size] if self.motor_size < self.state_size else state
        if self._output_mapping in ["affine", "linear"]:
            output *= self._params["output_w"]
        if self._output_mapping == "affine":
            output += self._params["output_b"]
        return output

    def _clip(self, w):
        return F.relu(w)

    def apply_weight_constraints(self):
        for key in ["w", "sensory_w", "cm", "gleak", "laplacian"]:
            self._params[key].data = self._clip(self._params[key].data)

    def forward(self, inputs, adj, states=None):
        if states is None:
            states = torch.zeros(inputs.size(0))
        inputs = self._map_inputs(inputs)
        next_state, Bmatrix, control_signal = self._ode_solver(inputs, states, 1.0, adj)
        outputs = self._map_outputs(next_state)
        return outputs, next_state, Bmatrix, control_signal

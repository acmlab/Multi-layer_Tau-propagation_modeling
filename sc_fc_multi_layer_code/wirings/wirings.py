# Copyright 2020-2021 Mathias Lechner
# Licensed under the Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0

import numpy as np

class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.input_dim = None
        self.output_dim = None

    def is_built(self):
        return self.input_dim is not None

    def build(self, input_shape):
        input_dim = int(input_shape[1])
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Conflicting input dimensions provided: expected {self.input_dim}, got {input_dim}"
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.adjacency_matrix)

    def sensory_erev_initializer(self, shape=None, dtype=None):
        return np.copy(self.sensory_adjacency_matrix)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros([input_dim, self.units], dtype=np.int32)

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id):
        return "motor" if neuron_id < self.output_dim else "inter"

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units or dest < 0 or dest >= self.units:
            raise ValueError("Invalid synapse indices")
        if polarity not in [-1, 1]:
            raise ValueError("Synapse polarity must be -1 or +1")
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError("build() must be called before adding sensory synapses")
        if src < 0 or src >= self.input_dim or dest < 0 or dest >= self.units:
            raise ValueError("Invalid sensory synapse indices")
        if polarity not in [-1, 1]:
            raise ValueError("Synapse polarity must be -1 or +1")
        self.sensory_adjacency_matrix[src, dest] = polarity

    def get_config(self):
        return {
            "adjacency_matrix": self.adjacency_matrix,
            "sensory_adjacency_matrix": self.sensory_adjacency_matrix,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "units": self.units,
        }

    @classmethod
    def from_config(cls, config):
        wiring = Wiring(config["units"])
        wiring.adjacency_matrix = config["adjacency_matrix"]
        wiring.sensory_adjacency_matrix = config["sensory_adjacency_matrix"]
        wiring.input_dim = config["input_dim"]
        wiring.output_dim = config["output_dim"]
        return wiring


class FullyConnected(Wiring):
    def __init__(self, units, output_dim=None, erev_init_seed=1111, self_connections=True):
        super().__init__(units)
        self.set_output_dim(output_dim or units)
        self.self_connections = self_connections
        self._rng = np.random.default_rng(erev_init_seed)
        for src in range(self.units):
            for dest in range(self.units):
                if src == dest and not self_connections:
                    continue
                polarity = self._rng.choice([-1, 1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        for src in range(self.input_dim):
            for dest in range(self.units):
                polarity = self._rng.choice([-1, 1, 1])
                self.add_sensory_synapse(src, dest, polarity)


class Random(Wiring):
    def __init__(self, units, output_dim=None, sparsity_level=0.0, random_seed=1111):
        super().__init__(units)
        self.set_output_dim(output_dim or units)
        self.sparsity_level = sparsity_level
        if not 0.0 <= sparsity_level < 1.0:
            raise ValueError("sparsity_level must be in [0, 1)")
        self._rng = np.random.default_rng(random_seed)

        num_synapses = int(np.round(units * units * (1 - sparsity_level)))
        all_synapses = [(src, dest) for src in range(units) for dest in range(units)]
        selected = self._rng.choice(all_synapses, size=num_synapses, replace=False)
        for src, dest in selected:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        num_synapses = int(np.round(self.input_dim * self.units * (1 - self.sparsity_level)))
        all_synapses = [(src, dest) for src in range(self.input_dim) for dest in range(self.units)]
        selected = self._rng.choice(all_synapses, size=num_synapses, replace=False)
        for src, dest in selected:
            polarity = self._rng.choice([-1, 1, 1])
            self.add_sensory_synapse(src, dest, polarity)


class NCP(Wiring):
    def __init__(self, inter_neurons, command_neurons, motor_neurons, sensory_fanout, inter_fanout, recurrent_command_synapses, motor_fanin, seed=22222):
        super().__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self._rng = np.random.RandomState(seed)
        self._motor_neurons = list(range(motor_neurons))
        self._command_neurons = list(range(motor_neurons, motor_neurons + command_neurons))
        self._inter_neurons = list(range(motor_neurons + command_neurons, motor_neurons + command_neurons + inter_neurons))
        self._sensory_fanout = sensory_fanout
        self._inter_fanout = inter_fanout
        self._recurrent_command_synapses = recurrent_command_synapses
        self._motor_fanin = motor_fanin

    def get_type_of_neuron(self, neuron_id):
        if neuron_id in self._motor_neurons:
            return "motor"
        if neuron_id in self._command_neurons:
            return "command"
        return "inter"

    def _build_sensory_to_inter(self):
        unconnected = set(self._inter_neurons)
        for src in self._sensory_neurons:
            for dest in self._rng.choice(self._inter_neurons, size=self._sensory_fanout, replace=False):
                unconnected.discard(dest)
                self.add_sensory_synapse(src, dest, self._rng.choice([-1, 1]))
        for dest in unconnected:
            for src in self._rng.choice(self._sensory_neurons, size=1, replace=False):
                self.add_sensory_synapse(src, dest, self._rng.choice([-1, 1]))

    def _build_inter_to_command(self):
        unconnected = set(self._command_neurons)
        for src in self._inter_neurons:
            for dest in self._rng.choice(self._command_neurons, size=self._inter_fanout, replace=False):
                unconnected.discard(dest)
                self.add_synapse(src, dest, self._rng.choice([-1, 1]))
        for dest in unconnected:
            for src in self._rng.choice(self._inter_neurons, size=1, replace=False):
                self.add_synapse(src, dest, self._rng.choice([-1, 1]))

    def _build_command_recurrence(self):
        for _ in range(self._recurrent_command_synapses):
            src = self._rng.choice(self._command_neurons)
            dest = self._rng.choice(self._command_neurons)
            self.add_synapse(src, dest, self._rng.choice([-1, 1]))

    def _build_command_to_motor(self):
        unconnected = set(self._command_neurons)
        for dest in self._motor_neurons:
            for src in self._rng.choice(self._command_neurons, size=self._motor_fanin, replace=False):
                unconnected.discard(src)
                self.add_synapse(src, dest, self._rng.choice([-1, 1]))
        for src in unconnected:
            for dest in self._rng.choice(self._motor_neurons, size=1, replace=False):
                self.add_synapse(src, dest, self._rng.choice([-1, 1]))

    def build(self, input_shape):
        super().build(input_shape)
        self._num_sensory_neurons = self.input_dim
        self._sensory_neurons = list(range(self._num_sensory_neurons))
        self._build_sensory_to_inter()
        self._build_inter_to_command()
        self._build_command_recurrence()
        self._build_command_to_motor()

"""
Vectorized Quantum Circuit — Pure PyTorch (No PennyLane).

Implements quantum operations as batched tensor ops on GPU.
Achieves 50-100x speedup over PennyLane's sequential qnode.

Ported from ECG project's v2/quantum_circuit.py with adaptations
for the EEG sleep staging pipeline.

Features:
- RX, RY, RZ single-qubit rotation gates (batched)
- Circular and full entanglement via CNOT
- Pauli-Z measurement
- Configurable rotation_axes: 'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class QuantumGates:
    """Static methods for batched quantum gate operations."""

    @staticmethod
    def ry_matrix(angle: torch.Tensor) -> torch.Tensor:
        """
        Create batched RY rotation matrices.

        RY(θ) = [[cos(θ/2), -sin(θ/2)],
                 [sin(θ/2),  cos(θ/2)]]

        Args:
            angle: (batch,) rotation angles
        Returns:
            (batch, 2, 2) complex rotation matrices
        """
        cos_half = torch.cos(angle / 2)
        sin_half = torch.sin(angle / 2)

        gate = torch.stack([
            torch.stack([cos_half, -sin_half], dim=-1),
            torch.stack([sin_half, cos_half], dim=-1)
        ], dim=-2)

        return gate.to(torch.complex64)

    @staticmethod
    def rx_matrix(angle: torch.Tensor) -> torch.Tensor:
        """
        Create batched RX rotation matrices.

        RX(θ) = [[cos(θ/2),    -i·sin(θ/2)],
                 [-i·sin(θ/2),  cos(θ/2)   ]]

        Args:
            angle: (batch,) rotation angles
        Returns:
            (batch, 2, 2) complex rotation matrices
        """
        cos_half = torch.cos(angle / 2).to(torch.complex64)
        sin_half = -1j * torch.sin(angle / 2).to(torch.complex64)

        gate = torch.stack([
            torch.stack([cos_half, sin_half], dim=-1),
            torch.stack([sin_half, cos_half], dim=-1)
        ], dim=-2)
        return gate

    @staticmethod
    def rz_matrix(angle: torch.Tensor) -> torch.Tensor:
        """
        Create batched RZ rotation matrices.

        RZ(φ) = [[e^(-iφ/2), 0        ],
                 [0,          e^(iφ/2) ]]

        Args:
            angle: (batch,) rotation angles
        Returns:
            (batch, 2, 2) complex rotation matrices
        """
        angle_c = angle.to(torch.complex64)
        phase_neg = torch.exp(-1j * angle_c / 2)
        phase_pos = torch.exp(1j * angle_c / 2)
        zeros = torch.zeros_like(angle, dtype=torch.complex64)

        gate = torch.stack([
            torch.stack([phase_neg, zeros], dim=-1),
            torch.stack([zeros, phase_pos], dim=-1)
        ], dim=-2)

        return gate


class VectorizedQuantumCircuit(nn.Module):
    """
    Vectorized Variational Quantum Circuit.

    Implements amplitude encoding + variational layers in pure PyTorch.
    All operations are batched and GPU-compatible — no per-sample loops.

    Architecture:
        1. Amplitude encoding: RY(x_i) on each qubit
        2. For each variational layer:
           a. Apply specified rotation gates with trainable parameters
           b. Apply entanglement (circular CNOT or full CNOT)
        3. Return quantum state vector

    Args:
        n_qubits: Number of qubits (default: 8)
        n_layers: Number of variational layers (default: 3)
        rotation_axes: String of axes per layer, e.g. 'y', 'xy', 'xyz'
        entanglement: 'circular' (ring) or 'full' (all-to-all)
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        rotation_axes: str = 'yz',
        entanglement: str = 'circular'
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits
        self.rotation_axes = rotation_axes.lower()
        self.entanglement = entanglement.lower()

        # Trainable rotation parameters — one set per axis
        self.params = nn.ParameterDict()
        for i, axis in enumerate(self.rotation_axes):
            self.params[f'theta_{i}_{axis}'] = nn.Parameter(
                torch.randn(n_layers, n_qubits) * 0.1
            )

        # Precompute Pauli-Z observables (constant tensors)
        self._init_pauli_z()

    def _init_pauli_z(self):
        """Precompute Pauli-Z observables for measurement."""
        pauli_z_list = []
        for qubit in range(self.n_qubits):
            z_obs = self._create_pauli_z(qubit)
            pauli_z_list.append(z_obs)
        self.register_buffer('pauli_z_obs', torch.stack(pauli_z_list, dim=0))

    def _create_pauli_z(self, qubit_idx: int) -> torch.Tensor:
        """
        Create Pauli-Z observable for a specific qubit.

        Builds: I ⊗ ... ⊗ Z ⊗ ... ⊗ I  (Z at position qubit_idx)
        """
        eye2 = torch.eye(2, dtype=torch.complex64)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        result = torch.tensor([1.0], dtype=torch.complex64)

        for i in range(self.n_qubits):
            mat = pauli_z if i == qubit_idx else eye2
            result = torch.kron(
                result.view(-1, 1) if i == 0 else result, mat
            )

        return result.view(self.state_dim, self.state_dim)

    def _apply_single_qubit_gate(
        self,
        state: torch.Tensor,
        qubit_idx: int,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply single-qubit gate using efficient tensor reshaping.

        Instead of building a full 2^n × 2^n matrix, we reshape the state
        vector to expose the target qubit axis and apply the 2×2 gate directly.

        Args:
            state: (batch, state_dim) quantum state
            qubit_idx: Target qubit index
            gate: (batch, 2, 2) gate matrices

        Returns:
            (batch, state_dim) new state
        """
        batch_size = state.shape[0]

        # Reshape to expose qubit structure: (batch, 2, 2, ..., 2)
        shape = [batch_size] + [2] * self.n_qubits
        state = state.view(*shape)

        # Move target qubit axis to position 1
        dims = list(range(self.n_qubits + 1))
        qubit_axis = qubit_idx + 1
        dims[1], dims[qubit_axis] = dims[qubit_axis], dims[1]
        state = state.permute(*dims)

        # Apply gate: (batch, 2, 2) @ (batch, 2, rest)
        rest_size = self.state_dim // 2
        state = state.reshape(batch_size, 2, rest_size)
        state = torch.bmm(gate, state)

        # Reshape back
        shape_after = [batch_size, 2] + [2] * (self.n_qubits - 1)
        state = state.view(*shape_after)

        # Permute back to original qubit ordering
        inverse_dims = list(range(self.n_qubits + 1))
        inverse_dims[1], inverse_dims[qubit_axis] = inverse_dims[qubit_axis], inverse_dims[1]
        state = state.permute(*inverse_dims)

        return state.reshape(batch_size, self.state_dim)

    def _apply_cnot(
        self,
        state: torch.Tensor,
        control: int,
        target: int
    ) -> torch.Tensor:
        """
        Apply CNOT gate using index manipulation.

        Flips target qubit amplitude when control qubit is |1⟩.
        No matrix multiplication needed — just swaps amplitudes.
        """
        batch_size = state.shape[0]

        # Reshape to expose qubit structure
        shape = [batch_size] + [2] * self.n_qubits
        state = state.view(*shape)

        # Build indices for control=1 case
        control_axis = control + 1
        target_axis = target + 1

        idx_c1_t0 = [slice(None)] * (self.n_qubits + 1)
        idx_c1_t1 = [slice(None)] * (self.n_qubits + 1)
        idx_c1_t0[control_axis] = 1
        idx_c1_t0[target_axis] = 0
        idx_c1_t1[control_axis] = 1
        idx_c1_t1[target_axis] = 1

        # Swap amplitudes when control is |1⟩
        state_c1_t0 = state[tuple(idx_c1_t0)].clone()
        state_c1_t1 = state[tuple(idx_c1_t1)].clone()

        state[tuple(idx_c1_t0)] = state_c1_t1
        state[tuple(idx_c1_t1)] = state_c1_t0

        return state.reshape(batch_size, self.state_dim)

    def _apply_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """Apply entanglement gates based on topology setting."""
        if self.entanglement == 'circular':
            # Ring: q0→q1→q2→...→q(n-1)→q0
            for q in range(self.n_qubits):
                target = (q + 1) % self.n_qubits
                state = self._apply_cnot(state, control=q, target=target)
        elif self.entanglement == 'full':
            # All-to-all: every pair
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    state = self._apply_cnot(state, control=i, target=j)
        return state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode features and apply variational circuit.

        Args:
            x: (batch, n_qubits) classical features normalized to [0, π]

        Returns:
            (batch, state_dim) quantum state amplitudes
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize |0⟩^⊗n
        state = torch.zeros(
            batch_size, self.state_dim,
            dtype=torch.complex64, device=device
        )
        state[:, 0] = 1.0 + 0.0j

        # ===== Amplitude Encoding =====
        # Apply RY(x_i) to each qubit for angle encoding
        for i in range(self.n_qubits):
            angle = x[:, i]
            gate = QuantumGates.ry_matrix(angle).to(device)
            state = self._apply_single_qubit_gate(state, i, gate)

        # ===== Variational Layers =====
        for layer in range(self.n_layers):
            for i, axis in enumerate(self.rotation_axes):
                # Apply parameterized rotation for this axis
                for qubit in range(self.n_qubits):
                    angle = self.params[f'theta_{i}_{axis}'][layer, qubit].expand(batch_size)

                    if axis == 'x':
                        gate = QuantumGates.rx_matrix(angle).to(device)
                    elif axis == 'y':
                        gate = QuantumGates.ry_matrix(angle).to(device)
                    elif axis == 'z':
                        gate = QuantumGates.rz_matrix(angle).to(device)
                    else:
                        raise ValueError(f"Unknown rotation axis: {axis}")

                    state = self._apply_single_qubit_gate(state, qubit, gate)

                # Apply entanglement after first rotation in each layer
                if i == 0:
                    state = self._apply_entanglement(state)

        return state

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        """
        Measure Pauli-Z expectation for each qubit.

        Args:
            state: (batch, state_dim) quantum state

        Returns:
            (batch, n_qubits) real expectation values in [-1, 1]
        """
        expectations = []

        for i in range(self.n_qubits):
            obs = self.pauli_z_obs[i].to(state.device)
            # ⟨ψ|Z_i|ψ⟩ = ψ† · Z_i · ψ
            obs_state = torch.matmul(obs, state.T).T
            expectation = torch.real(torch.sum(state.conj() * obs_state, dim=1))
            expectations.append(expectation)

        return torch.stack(expectations, dim=1)


class QuantumMeasurement(nn.Module):
    """
    Standalone measurement module for quantum state.

    Wraps Pauli-Z measurement for modular use.
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits

        # Precompute Pauli-Z observables
        pauli_z_list = []
        for qubit in range(n_qubits):
            z_obs = self._create_pauli_z(qubit)
            pauli_z_list.append(z_obs)
        self.register_buffer('pauli_z_obs', torch.stack(pauli_z_list, dim=0))

    def _create_pauli_z(self, qubit_idx: int) -> torch.Tensor:
        """Create Pauli-Z observable for a specific qubit."""
        eye2 = torch.eye(2, dtype=torch.complex64)
        pauli_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        result = torch.tensor([1.0], dtype=torch.complex64)
        for i in range(self.n_qubits):
            mat = pauli_z if i == qubit_idx else eye2
            result = torch.kron(
                result.view(-1, 1) if i == 0 else result, mat
            )

        return result.view(self.state_dim, self.state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Measure quantum state.

        Args:
            state: (batch, state_dim) quantum state
        Returns:
            (batch, n_qubits) Pauli-Z expectations
        """
        expectations = []
        for i in range(self.n_qubits):
            obs = self.pauli_z_obs[i].to(state.device)
            obs_state = torch.matmul(obs, state.T).T
            exp = torch.real(torch.sum(state.conj() * obs_state, dim=1))
            expectations.append(exp)

        return torch.stack(expectations, dim=1)


# =========================================================================
# Rotation axis mapping for pipeline compatibility
# =========================================================================

# Maps pipeline rotation strings to circuit rotation_axes format
ROTATION_MAP = {
    'RX': 'x',
    'RY': 'y',
    'RZ': 'z',
    'RXY': 'xy',
    'RXZ': 'xz',
    'RYZ': 'yz',
    'RXYZ': 'xyz',
}

ENTANGLEMENT_MAP = {
    'ring': 'circular',
    'full': 'full',
    'circular': 'circular',
    'none': 'none',
}

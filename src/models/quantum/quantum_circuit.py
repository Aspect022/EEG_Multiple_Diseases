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

        # Precompute measurement signs (constant tensors)
        self._init_measurement_signs()

    def _init_measurement_signs(self):
        """Precompute expectation measurement sign vectors for O(2^n) measurement."""
        signs = []
        for qubit in range(self.n_qubits):
            qubit_signs = []
            for k in range(self.state_dim):
                bit = (k >> (self.n_qubits - 1 - qubit)) & 1
                qubit_signs.append(1.0 if bit == 0 else -1.0)
            signs.append(qubit_signs)
        self.register_buffer('measurement_signs', torch.tensor(signs, dtype=torch.float32))

    def _apply_single_qubit_gate(
        self,
        state: torch.Tensor,
        qubit_idx: int,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply single-qubit gate using highly optimized dynamic einsum contraction.
        """
        batch_size = state.shape[0]
        state_reshaped = state.view([batch_size] + [2] * self.n_qubits)
        
        # Build einsum subscripts dynamically
        state_subs = [chr(97 + i) for i in range(self.n_qubits)]
        target_char = state_subs[qubit_idx]
        gate_subs = ['B', 'X', target_char]
        
        out_subs = list(state_subs)
        out_subs[qubit_idx] = 'X'
        
        subscripts = f"B{gate_subs[1]}{gate_subs[2]},B{''.join(state_subs)}->B{''.join(out_subs)}"
        state_out = torch.einsum(subscripts, gate, state_reshaped)
        return state_out.reshape(batch_size, self.state_dim)

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

        # Swap amplitudes when control is |1⟩ on a cloned tensor to remain out-of-place for autograd
        new_state = state.clone()
        new_state[tuple(idx_c1_t0)] = state[tuple(idx_c1_t1)]
        new_state[tuple(idx_c1_t1)] = state[tuple(idx_c1_t0)]

        return new_state.reshape(batch_size, self.state_dim)

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
        Optimized to run in O(2^n) without full matrix multiplications.
        """
        probs = (state.conj() * state).real
        if probs.ndim == 3 and probs.shape[-1] == 1:
            probs = probs.squeeze(-1)
        return torch.matmul(probs, self.measurement_signs.to(state.device).T)


class QuantumMeasurement(nn.Module):
    """
    Standalone measurement module for quantum state.
    Optimized Pauli-Z expectation calculation.
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self._init_measurement_signs()

    def _init_measurement_signs(self):
        """Precompute expectation measurement sign vectors for O(2^n) measurement."""
        signs = []
        for qubit in range(self.n_qubits):
            qubit_signs = []
            for k in range(self.state_dim):
                bit = (k >> (self.n_qubits - 1 - qubit)) & 1
                qubit_signs.append(1.0 if bit == 0 else -1.0)
            signs.append(qubit_signs)
        self.register_buffer('measurement_signs', torch.tensor(signs, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        probs = (state.conj() * state).real
        if probs.ndim == 3 and probs.shape[-1] == 1:
            probs = probs.squeeze(-1)
        return torch.matmul(probs, self.measurement_signs.to(state.device).T)


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

import torch
import torch.nn as nn
import numpy as np

class VQCLayer(nn.Module):
    """
    Differentiable VQC with:
      - n_qubits qubit registers (default 4)
      - 3 rotation layers (RX, RY, RZ per qubit)
      - Ring-topology CNOT entanglement between layers
      - Pure PyTorch: all ops are autograd-compatible
      - Optimized using dynamic einsum for single-qubit gates

    Input:  complex state vector (B, 2^n_qubits, 1)
    Output: complex state vector (B, 2^n_qubits, 1)
    """

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

        # Learnable rotation parameters: [3 layers × n_qubits × 3 gates (RX,RY,RZ)]
        self.theta = nn.Parameter(torch.randn(3, n_qubits, 3) * 0.1)

        # Precompute ring CNOT matrices and register as buffers
        for i in range(n_qubits):
            ctrl = i
            tgt  = (i + 1) % n_qubits
            self.register_buffer(f'cnot_{i}', self._build_cnot(ctrl, tgt))

    def _build_cnot(self, ctrl: int, tgt: int) -> torch.Tensor:
        """Build full CNOT matrix for n-qubit system."""
        dim = self.dim
        cnot = torch.eye(dim, dtype=torch.complex64)
        for i in range(dim):
            # flip target bit if control bit is 1
            ctrl_val = (i >> (self.n_qubits - 1 - ctrl)) & 1
            if ctrl_val == 1:
                j = i ^ (1 << (self.n_qubits - 1 - tgt))
                cnot[i, i] = 0
                cnot[i, j] = 1
                cnot[j, i] = 1
                cnot[j, j] = 0
        return cnot

    def _rx(self, theta: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        # Create a 2x2 gate for a single angle parameter
        gate = torch.zeros(2, 2, dtype=torch.complex64, device=theta.device)
        gate[0, 0] = c
        gate[0, 1] = -1j * s
        gate[1, 0] = -1j * s
        gate[1, 1] = c
        return gate

    def _ry(self, theta: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        gate = torch.zeros(2, 2, dtype=torch.complex64, device=theta.device)
        gate[0, 0] = c
        gate[0, 1] = -s
        gate[1, 0] = s
        gate[1, 1] = c
        return gate

    def _rz(self, theta: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(theta/2), torch.sin(theta/2)
        gate = torch.zeros(2, 2, dtype=torch.complex64, device=theta.device)
        gate[0, 0] = torch.complex(c, -s)
        gate[1, 1] = torch.complex(c, s)
        return gate

    def _apply_single_gate(self, state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply single-qubit gate using dynamic einsum."""
        batch_size = state.shape[0]
        # state is (batch_size, dim, 1). Squeeze to (batch_size, dim)
        state_squeezed = state.squeeze(-1)
        state_reshaped = state_squeezed.view([batch_size] + [2] * self.n_qubits)
        
        # Build einsum subscripts dynamically
        state_subs = [chr(97 + idx) for idx in range(self.n_qubits)]
        target_char = state_subs[qubit]
        
        # Gate subscripts: 'x' (output), target qubit char (input)
        # Since gate is 2x2, but we need batch dimensions if gate is batched.
        # Here gate is 2x2 (same for all batch elements). So subscripts is "xy, ...y... -> ...x..."
        out_subs = list(state_subs)
        out_subs[qubit] = 'x'
        
        subscripts = f"xy,B{''.join(state_subs)}->B{''.join(out_subs)}"
        state_out = torch.einsum(subscripts, gate, state_reshaped)
        return state_out.reshape(batch_size, self.dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, dim, 1) complex tensor
        returns: (batch, dim, 1) complex tensor
        """
        dev = state.device
        gate_fns = [self._rx, self._ry, self._rz]

        for layer in range(3):
            # Rotation layer
            for q in range(self.n_qubits):
                for g, fn in enumerate(gate_fns):
                    gate = fn(self.theta[layer, q, g])
                    state = self._apply_single_gate(state, gate, q)
            # Entanglement layer (ring CNOT)
            for i in range(self.n_qubits):
                cnot = getattr(self, f'cnot_{i}').to(dev)
                state = torch.matmul(cnot.unsqueeze(0), state)

        return state

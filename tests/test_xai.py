import unittest
import torch
import numpy as np
from qspikexai.models.qspikexai_net import QSpikeXAINet
from qspikexai.xai.signal_xai import input_saliency, integrated_gradients
from qspikexai.xai.quantum_xai import quantum_gate_attribution

class TestXAIModules(unittest.TestCase):
    
    def setUp(self):
        # Set up a lightweight task (sleep_apnea has 1 channel, seq_len 15360)
        # Or MCI which has 19 channels, seq_len 2048
        self.task = 'mci'
        self.channels = 19
        self.seq_len = 2048
        self.model = QSpikeXAINet(task=self.task, n_qubits=4, n_heads_vqc=2)
        self.model.eval()
        
    def test_input_saliency_shape(self):
        x_raw = torch.randn(1, self.channels, self.seq_len)
        sal = input_saliency(self.model, x_raw, target_class=0)
        self.assertEqual(sal.shape, (1, self.channels, self.seq_len))
        
    def test_integrated_gradients_shape(self):
        x_raw = torch.randn(1, self.channels, self.seq_len)
        ig = integrated_gradients(self.model, x_raw, steps=4) # Use small steps for fast test
        self.assertEqual(ig.shape, (1, self.channels, self.seq_len))
        
    def test_quantum_gate_attribution_shape(self):
        x_raw = torch.randn(1, self.channels, self.seq_len)
        # Pre-compute a random dummy scalogram: (B, C, F, T) -> (1, 19, 40, 512)
        x_scal = torch.randn(1, self.channels, 40, self.seq_len // 4)
        
        # Calculate gate attribution
        attr_dict = quantum_gate_attribution(
            self.model, 
            x_raw, 
            x_scal, 
            target_class=0
        )
        
        theta_attr = attr_dict['theta_attribution']
        # VQC parameters theta has shape (3, n_qubits, 3)
        self.assertEqual(theta_attr.shape, (3, 4, 3))
        self.assertGreaterEqual(np.min(theta_attr), 0.0, "Attributions should be non-negative magnitude gradients")
        
        # Verify aggregates
        self.assertEqual(len(attr_dict['layer_scores']), 3)
        self.assertEqual(len(attr_dict['qubit_scores']), 4)
        self.assertIn('RX', attr_dict['gate_type_scores'])
        self.assertIn('RY', attr_dict['gate_type_scores'])
        self.assertIn('RZ', attr_dict['gate_type_scores'])

if __name__ == '__main__':
    unittest.main()
Class: TestXAIModules

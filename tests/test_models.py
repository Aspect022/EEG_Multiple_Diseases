import unittest
import torch
from qspikexai.models.qspikexai_net import QSpikeXAINet
from qspikexai.models.baselines.eegnet import EEGNet
from qspikexai.models.baselines.eeg_tcnet import EEGTCNet
from qspikexai.models.baselines.resnet1d import ResNet1D
from qspikexai.models.baselines.vit1d import ViT1D

class TestModelShapes(unittest.TestCase):
    
    def test_qspikexai_net_shapes(self):
        tasks = ['sleep_apnea', 'schizophrenia', 'mci', 'depression']
        
        # Batch size = 2
        for task in tasks:
            # Determine channel counts and sequence lengths
            from qspikexai.models.qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
            channels = TASK_CHANNELS[task]
            seq_len = TASK_SEQ_LEN[task]
            n_classes = 4 if task == 'sleep_apnea' else 2
            
            print(f"Testing QSpikeXAINet on {task}: channels={channels}, seq_len={seq_len}")
            model = QSpikeXAINet(task=task, n_qubits=4, n_heads_vqc=4)
            model.eval()
            
            # Input raw signal: shape (B, C, T)
            x_raw = torch.randn(2, channels, seq_len)
            
            # Forward pass (computes CWT on-the-fly)
            with torch.no_grad():
                logits = model(x_raw)
                
            self.assertEqual(logits.shape, (2, n_classes), f"Incorrect output logits shape for task {task}")
            
    def test_baselines_shapes(self):
        tasks = ['sleep_apnea', 'schizophrenia', 'mci', 'depression']
        baselines = ['eegnet', 'eeg_tcnet', 'resnet1d', 'vit1d']
        
        for task in tasks:
            from qspikexai.models.qspikexai_net import TASK_CHANNELS, TASK_SEQ_LEN
            channels = TASK_CHANNELS[task]
            seq_len = TASK_SEQ_LEN[task]
            n_classes = 4 if task == 'sleep_apnea' else 2
            
            for b_name in baselines:
                print(f"Testing baseline {b_name} on {task}")
                
                # Dynamic baseline instantiation
                if b_name == 'eegnet':
                    model = EEGNet(task=task)
                elif b_name == 'eeg_tcnet':
                    model = EEGTCNet(task=task)
                elif b_name == 'resnet1d':
                    model = ResNet1D(task=task)
                elif b_name == 'vit1d':
                    model = ViT1D(task=task, patch_size=64)
                    
                model.eval()
                x_raw = torch.randn(2, channels, seq_len)
                
                with torch.no_grad():
                    logits = model(x_raw)
                    
                self.assertEqual(logits.shape, (2, n_classes), f"Incorrect output logits shape for baseline {b_name} on {task}")

if __name__ == '__main__':
    unittest.main()

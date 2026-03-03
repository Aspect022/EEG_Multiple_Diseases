"""
Benchmarking Utilities for ECG Classification Models.

This module provides tools to measure and compare model performance metrics:
- Inference Time (ms)
- VRAM/GPU Memory Usage (MB)
- FLOPs (Floating Point Operations)
- Parameter Count

Reference:
- fvcore for FLOPs: https://github.com/facebookresearch/fvcore
- thop for FLOPs: https://github.com/Lyken17/pytorch-OpCounter
"""

import time
import gc
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    model_name: str
    input_shape: Tuple[int, ...]
    
    # Timing metrics
    inference_time_ms: float = 0.0
    inference_time_std_ms: float = 0.0
    warmup_time_ms: float = 0.0
    
    # Memory metrics
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    vram_peak_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    
    # Compute metrics
    flops: int = 0
    macs: int = 0  # Multiply-Accumulate operations
    params_total: int = 0
    params_trainable: int = 0
    
    # Additional info
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    num_runs: int = 100
    
    def __str__(self) -> str:
        """Pretty print benchmark results."""
        lines = [
            f"\n{'='*60}",
            f"Benchmark Results: {self.model_name}",
            f"{'='*60}",
            f"Input Shape: {self.input_shape}",
            f"Device: {self.device} | Dtype: {self.dtype}",
            f"Batch Size: {self.batch_size} | Runs: {self.num_runs}",
            f"{'-'*60}",
            f"TIMING:",
            f"  Inference Time: {self.inference_time_ms:.3f} ± {self.inference_time_std_ms:.3f} ms",
            f"  Throughput: {1000/self.inference_time_ms:.1f} samples/sec" if self.inference_time_ms > 0 else "",
            f"{'-'*60}",
            f"MEMORY:",
            f"  VRAM Allocated: {self.vram_allocated_mb:.2f} MB",
            f"  VRAM Peak: {self.vram_peak_mb:.2f} MB",
        ]
        
        if self.flops > 0:
            lines.extend([
                f"{'-'*60}",
                f"COMPUTE:",
                f"  FLOPs: {self._format_number(self.flops)}",
                f"  MACs: {self._format_number(self.macs)}",
            ])
        
        lines.extend([
            f"{'-'*60}",
            f"PARAMETERS:",
            f"  Total: {self._format_number(self.params_total)}",
            f"  Trainable: {self._format_number(self.params_trainable)}",
            f"{'='*60}",
        ])
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_number(n: int) -> str:
        """Format large numbers with K/M/G suffixes."""
        if n >= 1e9:
            return f"{n/1e9:.2f}G"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'inference_time_ms': self.inference_time_ms,
            'inference_time_std_ms': self.inference_time_std_ms,
            'vram_allocated_mb': self.vram_allocated_mb,
            'vram_peak_mb': self.vram_peak_mb,
            'flops': self.flops,
            'macs': self.macs,
            'params_total': self.params_total,
            'params_trainable': self.params_trainable,
            'device': self.device,
            'batch_size': self.batch_size,
        }


class ModelBenchmark:
    """
    Comprehensive model benchmarking utility.
    
    Measures inference time, memory usage, FLOPs, and parameter counts
    for PyTorch models.
    
    Args:
        model: PyTorch model to benchmark.
        device: Device to run benchmarks on. Default: auto-detect.
        dtype: Data type for inputs. Default: torch.float32.
        
    Example:
        >>> model = MyModel()
        >>> benchmark = ModelBenchmark(model)
        >>> result = benchmark.run(input_shape=(1, 3, 224, 224))
        >>> print(result)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.dtype = dtype
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device).to(dtype)
        self.model.eval()
        
        # Model name
        self.model_name = model.__class__.__name__
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.
        
        Returns:
            Tuple of (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
    
    def measure_inference_time(
        self,
        input_tensor: torch.Tensor,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Measure inference time.
        
        Args:
            input_tensor: Input tensor for forward pass.
            num_warmup: Number of warmup iterations.
            num_runs: Number of timed iterations.
            
        Returns:
            Tuple of (mean_ms, std_ms, warmup_ms).
        """
        input_tensor = input_tensor.to(self.device).to(self.dtype)
        
        # Warmup
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        warmup_time = (time.perf_counter() - warmup_start) * 1000 / num_warmup
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(input_tensor)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                times.append((time.perf_counter() - start) * 1000)
        
        return np.mean(times), np.std(times), warmup_time
    
    def measure_memory(
        self,
        input_tensor: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """
        Measure GPU memory usage.
        
        Args:
            input_tensor: Input tensor for forward pass.
            
        Returns:
            Tuple of (allocated_mb, reserved_mb, peak_mb, cpu_mb).
        """
        if self.device.type != 'cuda':
            return 0.0, 0.0, 0.0, 0.0
        
        # Clear cache and reset stats
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        input_tensor = input_tensor.to(self.device).to(self.dtype)
        
        # Measure before
        torch.cuda.synchronize()
        
        # Run inference
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        torch.cuda.synchronize()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        return allocated, reserved, peak, 0.0
    
    def count_flops(
        self,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, int]:
        """
        Count FLOPs and MACs for the model.
        
        Tries multiple backends (fvcore, thop) and falls back gracefully.
        
        Args:
            input_shape: Shape of input tensor.
            
        Returns:
            Tuple of (flops, macs).
        """
        input_tensor = torch.randn(input_shape, device=self.device, dtype=self.dtype)
        
        # Try fvcore first
        try:
            from fvcore.nn import FlopCountAnalysis, parameter_count
            
            flops_analyzer = FlopCountAnalysis(self.model, input_tensor)
            flops = flops_analyzer.total()
            macs = flops // 2  # Approximate
            return int(flops), int(macs)
        except ImportError:
            pass
        except Exception as e:
            print(f"fvcore FLOPs counting failed: {e}")
        
        # Try thop
        try:
            from thop import profile
            
            macs, params = profile(self.model, inputs=(input_tensor,), verbose=False)
            flops = int(macs * 2)  # MACs to FLOPs
            return flops, int(macs)
        except ImportError:
            pass
        except Exception as e:
            print(f"thop FLOPs counting failed: {e}")
        
        # Try ptflops
        try:
            from ptflops import get_model_complexity_info
            
            macs, params = get_model_complexity_info(
                self.model,
                input_shape[1:],  # Without batch dimension
                as_strings=False,
                print_per_layer_stat=False,
            )
            flops = int(macs * 2)
            return flops, int(macs)
        except ImportError:
            pass
        except Exception as e:
            print(f"ptflops FLOPs counting failed: {e}")
        
        # Unable to count FLOPs
        print("Warning: No FLOPs counting library available. Install fvcore, thop, or ptflops.")
        return 0, 0
    
    def run(
        self,
        input_shape: Tuple[int, ...],
        num_warmup: int = 10,
        num_runs: int = 100,
        count_flops: bool = True,
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark.
        
        Args:
            input_shape: Shape of input tensor (including batch dimension).
            num_warmup: Number of warmup iterations.
            num_runs: Number of timed iterations.
            count_flops: Whether to count FLOPs.
            
        Returns:
            BenchmarkResult with all metrics.
        """
        # Create input tensor
        input_tensor = torch.randn(input_shape, device=self.device, dtype=self.dtype)
        
        # Count parameters
        params_total, params_trainable = self.count_parameters()
        
        # Measure inference time
        time_mean, time_std, warmup_time = self.measure_inference_time(
            input_tensor, num_warmup, num_runs
        )
        
        # Measure memory
        vram_alloc, vram_reserved, vram_peak, cpu_mem = self.measure_memory(input_tensor)
        
        # Count FLOPs
        flops, macs = 0, 0
        if count_flops:
            flops, macs = self.count_flops(input_shape)
        
        return BenchmarkResult(
            model_name=self.model_name,
            input_shape=input_shape,
            inference_time_ms=time_mean,
            inference_time_std_ms=time_std,
            warmup_time_ms=warmup_time,
            vram_allocated_mb=vram_alloc,
            vram_reserved_mb=vram_reserved,
            vram_peak_mb=vram_peak,
            cpu_memory_mb=cpu_mem,
            flops=flops,
            macs=macs,
            params_total=params_total,
            params_trainable=params_trainable,
            device=str(self.device),
            dtype=str(self.dtype).split('.')[-1],
            batch_size=input_shape[0],
            num_runs=num_runs,
        )


class ComparativeBenchmark:
    """
    Compare multiple models on the same benchmarks.
    
    Args:
        models: Dict mapping model names to model instances.
        device: Device for benchmarking.
        
    Example:
        >>> models = {
        ...     'ResNet18': resnet18(),
        ...     'Swin-T': swin_tiny(),
        ... }
        >>> benchmark = ComparativeBenchmark(models)
        >>> results = benchmark.run(input_shape=(1, 3, 224, 224))
        >>> benchmark.print_comparison(results)
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: Optional[str] = None,
    ):
        self.models = models
        self.device = device
    
    def run(
        self,
        input_shape: Tuple[int, ...],
        **kwargs,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmarks on all models.
        
        Args:
            input_shape: Input tensor shape.
            **kwargs: Arguments passed to ModelBenchmark.run().
            
        Returns:
            Dict mapping model names to BenchmarkResults.
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"\nBenchmarking: {name}")
            benchmark = ModelBenchmark(model, device=self.device)
            results[name] = benchmark.run(input_shape, **kwargs)
        
        return results
    
    def print_comparison(self, results: Dict[str, BenchmarkResult]):
        """Print side-by-side comparison table."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        # Header
        headers = ["Model", "Time (ms)", "VRAM (MB)", "FLOPs", "Params"]
        row_format = "{:<20} {:>12} {:>12} {:>12} {:>12}"
        print(row_format.format(*headers))
        print("-" * 80)
        
        # Rows
        for name, result in results.items():
            print(row_format.format(
                name[:20],
                f"{result.inference_time_ms:.2f}",
                f"{result.vram_peak_mb:.1f}",
                BenchmarkResult._format_number(result.flops),
                BenchmarkResult._format_number(result.params_total),
            ))
        
        print("=" * 80)
    
    def to_dataframe(self, results: Dict[str, BenchmarkResult]):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd
            
            data = [r.to_dict() for r in results.values()]
            return pd.DataFrame(data)
        except ImportError:
            print("pandas not installed. Install with: pip install pandas")
            return None


@contextmanager
def measure_time(name: str = "Operation"):
    """
    Context manager for quick timing measurements.
    
    Example:
        >>> with measure_time("Forward pass"):
        ...     output = model(input)
        Forward pass: 12.34 ms
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    yield
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f} ms")


def quick_benchmark(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    device: str = 'cuda',
    num_runs: int = 50,
) -> BenchmarkResult:
    """
    Quick one-liner benchmark.
    
    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape.
        device: Device to use.
        num_runs: Number of timed runs.
        
    Returns:
        BenchmarkResult.
    """
    benchmark = ModelBenchmark(model, device=device)
    return benchmark.run(input_shape, num_runs=num_runs)

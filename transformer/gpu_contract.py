"""Shared host interface for the prospective GPU experiment.

This interface specifies ownership and observable behavior, not numerical kernels.
Backend-specific values and datasets remain opaque and GPU resident. CPU source
and prior experiment contracts are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

GPU_ENGINES = ("mps", "mlx", "metal")
GPU_CONTRACT = "gpu-device-batch-v1"


class GPUBackend(Protocol):
    """All floating tensors, parameters, gradients, and moments are stored FP32.

    Constructors accept ``initial_flat, max_batch=16, max_length=32``. Every
    floating result is a backend-specific device value until explicitly read.
    Integer input uploads must reject malformed shapes, ranges, and types.
    Uploaded datasets/schedules remain alive throughout the fit. The caller
    serializes execution; one backend owns all its workspaces and pending work.
    """

    def upload_inputs(self, tokens: np.ndarray, targets: np.ndarray | None = None) -> Any:
        """Upload int32 [B,T] tokens and optional equally shaped targets."""
        ...

    def upload_dataset(self, rows: np.ndarray) -> Any:
        """Upload int32 [N,T+1] rows; no model arithmetic occurs on the CPU."""
        ...

    def upload_indices(self, indices: np.ndarray) -> Any:
        """Upload an int32, one-dimensional schedule or evaluation index list."""
        ...

    def gather_batch(self, dataset: Any, index_buffer: Any, offset: int, count: int) -> Any:
        """Gather tokens/targets on the GPU using a resident schedule slice."""
        ...

    def forward(self, batch: Any, return_trace: bool = False) -> Any:
        """Forward logits [B,T,128]; an audit trace may accompany the logits."""
        ...

    def loss(self, batch: Any) -> Any:
        """Mean next-character cross entropy; no gradients or optimizer update."""
        ...

    def evaluate(self, batch: Any) -> tuple[Any, Any]:
        """Return (loss, logits) from one forward, without backward or update."""
        ...

    def loss_and_grad(self, batch: Any) -> tuple[Any, Any]:
        """Return scalar loss and flat parameter gradients without updating state."""
        ...

    def train_step(self, batch: Any) -> Any:
        """Forward/backward/Adam; return scalar loss and increment step once."""
        ...

    def check_finite(self, value: Any | None = None) -> bool:
        """GPU finite reduction with a scalar host result.

        With no value, cover last logits, loss, gradients, parameters and both
        moments. With a value, check that device value. The runner requires this
        after every training step and for evaluation outputs/loss. Full tensors
        must not be copied to CPU to perform this check.
        """
        ...

    def to_numpy(self, value: Any) -> np.ndarray:
        """Wait for completion and copy/read a device value into a host array."""
        ...

    def synchronize(self) -> None:
        """Wait for all work contributing to measured results to complete."""
        ...

    def export_state(self) -> dict[str, Any]:
        """Return independent flat FP32 parameters/m/v arrays and integer step."""
        ...

    def export_parameters(self) -> np.ndarray:
        """Read only flat FP32 parameters, without copying optimizer moments."""
        ...

    def load_state(self, state: dict[str, Any]) -> None:
        """Validate and restore parameters, m, v and step, with exact bytes."""
        ...

    def memory_stats(self) -> dict[str, Any]:
        """Report backend-defined current/peak counters with explicit units/scope."""
        ...

    def capabilities(self) -> dict[str, Any]:
        """Record device, dispatch/compilation mode, math settings and host scalars."""
        ...

    def close(self) -> None:
        """Finish outstanding work and release backend-owned resources."""
        ...

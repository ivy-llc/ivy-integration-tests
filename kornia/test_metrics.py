from helpers import _test_function
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_accuracy(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0, 1, 0]]),
        torch.tensor([[1]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[0, 0.8, 0.2], [0, 0.4, 0.6]]),
        torch.tensor([1, 2]),
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.accuracy,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_confusion_matrix(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 0]),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 1]),
        3,
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.confusion_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_mean_iou(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 0]),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 1]),
        3,
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.mean_iou,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_mean_average_precision(target_framework, mode, backend_compile):
    trace_args = (
        [torch.tensor([[100, 50, 150, 100.]])],
        [torch.tensor([1.])],
        [torch.tensor([0.7])],
        [torch.tensor([[100, 50, 150, 100.]])],
        [torch.tensor([1.])],
        2,
    )
    trace_kwargs = {}
    test_args = (
        [torch.tensor([[50, 25, 75, 50], [100, 50, 150, 100.]])],
        [torch.tensor([1, 2.])],
        [torch.tensor([0.6, 0.8])],
        [torch.tensor([[50, 25, 75, 50], [100, 50, 150, 100.]])],
        [torch.tensor([1, 2.])],
        3,
    )
    kornia.metrics.mean_average_precision(*trace_args)
    kornia.metrics.mean_average_precision(*test_args)
    test_kwargs = {}

    # NOTE: this test fails due to the use of dynamic control flow; skipping
    _test_function(
        kornia.metrics.mean_average_precision,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
        skip=True,
    )


def test_mean_iou_bbox(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]]),
        torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[20, 20, 40, 40], [10, 30, 30, 50]]),
        torch.tensor([[20, 30, 40, 50], [10, 20, 20, 30]])
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.mean_iou_bbox,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_psnr(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
        1.2 * torch.rand(1, 1, 4, 4),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 3, 4, 4),
        1.1 * torch.rand(2, 3, 4, 4),
        2.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.psnr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_ssim(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 1, 5, 5),
        5,
    )
    trace_kwargs = {}
    test_args = (
        # TODO: changing the channels dim here doesn't work with the transpiled function;
        # perhaps due to the looping implementation of conv_general_dilated in the ivy numpy backend?
        # (ivy/functional/backends/numpy/layers.py::conv_general_dilated)
        torch.rand(2, 1, 5, 5),
        torch.rand(2, 1, 5, 5),
        5,
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.ssim,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_ssim3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        torch.rand(1, 1, 5, 5, 5),
        5,
    )
    trace_kwargs = {}
    test_args = (
        # TODO: same as `test_ssim`: changing the channels dim doesn't work with the transpiled function
        torch.rand(2, 1, 5, 5, 5),
        torch.rand(2, 1, 5, 5, 5),
        5,
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.ssim3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_aepe(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(4, 4, 2),
        1.2 * torch.rand(4, 4, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4, 2),
        torch.rand(5, 4, 4, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.metrics.aepe,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )

from helpers import (
    _nest_torch_tensor_to_new_framework,
    _test_function,
    _to_numpy_and_allclose,
    _to_numpy_and_shape_allclose,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_gftt_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.gftt_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_harris_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'k': 0.04, 'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'k': 0.04, 'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.harris_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_hessian_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.hessian_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dog_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.dog_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dog_response_single(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {'sigma1': 1.0, 'sigma2': 1.6}
    test_args = (
        torch.rand(5, 1, 5, 5),
    )
    test_kwargs = {'sigma1': 0.5, 'sigma2': 1.2}
    _test_function(
        kornia.feature.dog_response_single,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laf_descriptors(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 32, 32),
        torch.rand(1, 3, 2, 3),
        kornia.feature.OriNet(pretrained=False),
    )
    trace_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
    test_args = (
        torch.rand(5, 1, 32, 32),
        torch.rand(5, 3, 2, 3),
        kornia.feature.OriNet(pretrained=False),
    )
    test_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
    class_info = {
        'trace_args': {
            2: {'object': kornia.feature.OriNet, 'kwargs': {'pretrained': False}}
        },
        'test_args': {
            2: {'object': kornia.feature.OriNet, 'kwargs': {'pretrained': False}}
        }
    }
    _test_function(
        kornia.feature.get_laf_descriptors,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
        class_info=class_info,
    )

# DCF: our torch.cdist implementation uses a for-loop. https://github.com/ivy-llc/ivy/blob/65548817e99a396461c1eec5cf4eb9453c125cde/ivy/functional/frontends/torch/miscellaneous_ops.py#L100
def test_match_nn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.match_nn,
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


def test_match_mnn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.match_mnn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_match_snn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {'th': 0.8}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {'th': 0.8}
    _test_function(
        kornia.feature.match_snn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_match_smnn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {'th': 0.95}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {'th': 0.95}
    _test_function(
        kornia.feature.match_smnn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_match_fginn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 128),
        torch.rand(3, 128),
        torch.rand(1, 3, 2, 3),
        torch.rand(1, 3, 2, 3),
    )
    trace_kwargs = {'th': 0.8, 'spatial_th': 10.0, 'mutual': False}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'th': 0.8, 'spatial_th': 10.0, 'mutual': False}
    _test_function(
        kornia.feature.match_fginn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_match_adalam(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 128),
        torch.rand(3, 128),
        torch.rand(1, 3, 2, 3),
        torch.rand(1, 3, 2, 3),
    )
    trace_kwargs = {'config': None, 'hw1': None, 'hw2': None}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'config': None, 'hw1': None, 'hw2': None}
    _test_function(
        kornia.feature.match_adalam,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_extract_patches_from_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'PS': 32}
    test_args = (
        torch.rand(1, 3, 64, 64),  # TODO: changing the batch size of these causes the trace_graph test to fail
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'PS': 16}
    _test_function(
        kornia.feature.extract_patches_from_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_extract_patches_simple(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'PS': 32, 'normalize_lafs_before_extraction': True}
    test_args = (
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 5, 2, 3),
    )
    test_kwargs = {'PS': 16, 'normalize_lafs_before_extraction': False}
    _test_function(
        kornia.feature.extract_patches_simple,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_normalize_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 2, 3),
        torch.rand(2, 3, 64, 64),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.normalize_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_denormalize_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 2, 3),
        torch.rand(2, 3, 64, 64),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.denormalize_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laf_to_boundary_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'n_pts': 50}
    test_args = (
        torch.rand(2, 5, 2, 3),
    )
    test_kwargs = {'n_pts': 100}
    _test_function(
        kornia.feature.laf_to_boundary_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_ellipse_to_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 10, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.ellipse_to_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_make_upright(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'eps': 1e-9}
    test_args = (
        torch.rand(2, 5, 2, 3),
    )
    test_kwargs = {'eps': 1e-6}
    _test_function(
        kornia.feature.make_upright,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )

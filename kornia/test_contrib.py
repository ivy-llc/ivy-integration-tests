from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _test_function,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #


def test_compute_padding(target_framework, mode, backend_compile):
    trace_args = (
        (4, 3),
        (3, 3),
    )
    trace_kwargs = {}
    test_args = (
        (8, 5),
        (4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.contrib.compute_padding,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_extract_tensor_patches(target_framework, mode, backend_compile):
    trace_args = (torch.arange(16).view(1, 1, 4, 4),)
    trace_kwargs = {
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    test_args = (torch.flip(torch.arange(32), (0,)).view(2, 1, 4, 4),)
    test_kwargs = {
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    _test_function(
        kornia.contrib.extract_tensor_patches,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_combine_tensor_patches(target_framework, mode, backend_compile):
    trace_args = (
        kornia.contrib.extract_tensor_patches(
            torch.arange(16).view(1, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    trace_kwargs = {
        "original_size": (4, 4),
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    test_args = (
        kornia.contrib.extract_tensor_patches(
            torch.flip(torch.arange(32), (0,)).view(2, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    test_kwargs = {
        "original_size": (4, 4),
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    _test_function(
        kornia.contrib.combine_tensor_patches,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_distance_transform(target_framework, mode, backend_compile):
    trace_args = (torch.randn(1, 1, 5, 5),)
    trace_kwargs = {"kernel_size": 3, "h": 0.35}
    test_args = (torch.randn(5, 1, 5, 5),)
    test_kwargs = {"kernel_size": 3, "h": 0.5}
    _test_function(
        kornia.contrib.distance_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_diamond_square(target_framework, mode, backend_compile):
    trace_args = ((1, 1, 8, 8),)
    trace_kwargs = {
        "roughness": 0.5,
        "random_scale": 1.0,
        "normalize_range": (0.0, 1.0),
        "random_fn": torch.ones,
    }
    test_args = ((5, 1, 8, 8),)
    test_kwargs = {
        "roughness": 0.7,
        "random_scale": 0.9,
        "normalize_range": (-1.0, 1.0),
        "random_fn": torch.ones,
    }
    _test_function(
        kornia.contrib.diamond_square,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_EdgeDetector(target_framework, mode, backend_compile):
    print("kornia.contrib.EdgeDetector")

    if backend_compile:
        pytest.skip()

    TranspiledEdgeDetector = ivy.transpile(kornia.contrib.EdgeDetector, source="torch", target=target)

    torch_detector = kornia.contrib.EdgeDetector()
    transpiled_detector = TranspiledEdgeDetector()

    torch_args = (
        torch.rand(1, 3, 320, 320),
    )
    torch_out = torch_detector(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_detector(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_allclose(orig_np, transpiled_np)


def test_FaceDetector(target_framework, mode, backend_compile):
    print("kornia.contrib.FaceDetector")

    if backend_compile:
        pytest.skip()

    TranspiledFaceDetector = ivy.transpile(kornia.contrib.FaceDetector, source="torch", target=target)

    torch_detector = kornia.contrib.FaceDetector()
    transpiled_detector = TranspiledFaceDetector()

    torch_args = (
        torch.rand(1, 3, 320, 320),
    )
    torch_out = torch_detector(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_detector(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_allclose(orig_np, transpiled_np)


def test_VisionTransformer(target_framework, mode, backend_compile):
    print("kornia.contrib.VisionTransformer")

    if backend_compile:
        pytest.skip()

    TranspiledVisionTransformer = ivy.transpile(kornia.contrib.VisionTransformer, source="torch", target=target)

    torch_vit = kornia.contrib.VisionTransformer()
    transpiled_vit = TranspiledVisionTransformer()

    torch_args = (
        torch.rand(1, 3, 320, 320),
    )
    torch_out = torch_vit(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_vit(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_KMeans(target_framework, mode, backend_compile):
    print("kornia.contrib.KMeans")

    if backend_compile:
        pytest.skip()

    TranspiledKMeans = ivy.transpile(kornia.contrib.KMeans, source="torch", target=target)

    torch_kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, 0)
    transpiled_kmeans = TranspiledKMeans(3, None, 10e-4, 100, 0)
    
    torch_x1 = torch.rand((1000, 5))
    torch_x2 = torch.rand((10, 5))
    transpiled_x1 = _array_to_new_backend(torch_x1, target_framework)
    transpiled_x2 = _array_to_new_backend(torch_x2, target_framework)

    torch_kmeans.fit(torch_x1)
    torch_predictions = torch_kmeans.predict(torch_x2)

    transpiled_kmeans.fit(transpiled_x1)
    transpiled_predictions = transpiled_kmeans.predict(transpiled_x2)

    orig_np = _nest_array_to_numpy(torch_predictions)
    transpiled_np = _nest_array_to_numpy(transpiled_predictions)
    _check_shape_allclose(orig_np, transpiled_np)


# TODO: there are more classes in kornia.contrib that are not being tested yet

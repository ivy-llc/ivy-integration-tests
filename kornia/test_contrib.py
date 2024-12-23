from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
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

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_detector = kornia.contrib.EdgeDetector()
    transpiled_detector = transpiled_kornia.contrib.EdgeDetector()

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

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_detector = kornia.contrib.FaceDetector()
    transpiled_detector = transpiled_kornia.contrib.FaceDetector()

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

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_vit = kornia.contrib.VisionTransformer()
    transpiled_vit = transpiled_kornia.contrib.VisionTransformer()

    torch_args = (
        torch.rand(1, 3, 224, 224),
    )
    torch_out = torch_vit(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_vit(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_KMeans(target_framework, mode, backend_compile):
    print("kornia.contrib.KMeans")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_kmeans = kornia.contrib.KMeans(3, None, 10e-4, 100, 0)
    transpiled_kmeans = transpiled_kornia.contrib.KMeans(3, None, 10e-4, 100, 0)

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


def test_ExtractTensorPatches(target_framework, mode, backend_compile):
    print("kornia.contrib.ExtractTensorPatches")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.arange(9.).view(1, 1, 3, 3)
    torch_out = kornia.contrib.ExtractTensorPatches(window_size=(2, 3))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.contrib.ExtractTensorPatches(window_size=(2, 3))(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CombineTensorPatches(target_framework, mode, backend_compile):
    print("kornia.contrib.CombineTensorPatches")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    patches = kornia.contrib.ExtractTensorPatches(window_size=(2, 2), stride=(2, 2))(torch.arange(16).view(1, 1, 4, 4))
    torch_out = kornia.contrib.CombineTensorPatches(original_size=(4, 4), window_size=(2, 2), stride=(2, 2))(patches)

    transpiled_patches = _nest_torch_tensor_to_new_framework(patches, target_framework)
    transpiled_out = transpiled_kornia.contrib.CombineTensorPatches(original_size=(4, 4), window_size=(2, 2), stride=(2, 2))(transpiled_patches)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_ClassificationHead(target_framework, mode, backend_compile):
    print("kornia.contrib.ClassificationHead")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(1, 256, 256)
    torch_out = kornia.contrib.ClassificationHead(256, 10)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.contrib.ClassificationHead(256, 10)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_ImageStitcher(target_framework, mode, backend_compile):
    print("kornia.contrib.ImageStitcher")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    torch_matcher = kornia.feature.LoFTR(pretrained='outdoor')
    transpiled_matcher = transpiled_kornia.feature.LoFTR(pretrained='outdoor')

    img_left = torch.rand(1, 3, 256, 256)
    img_right = torch.rand(1, 3, 256, 256)
    torch_out = kornia.contrib.ImageStitcher(torch_matcher)(img_left, img_right)

    transpiled_img_left = _nest_torch_tensor_to_new_framework(img_left, target_framework)
    transpiled_img_right = _nest_torch_tensor_to_new_framework(img_right, target_framework)
    transpiled_out = transpiled_kornia.contrib.ImageStitcher(transpiled_matcher)(transpiled_img_left, transpiled_img_right)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_Lambda(target_framework, mode, backend_compile):
    print("kornia.contrib.Lambda")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.rand(1, 3, 5, 5)
    torch_out = kornia.contrib.Lambda(lambda x: kornia.color.rgb_to_grayscale(x))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.contrib.Lambda(lambda x: transpiled_kornia.color.rgb_to_grayscale(x))(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DistanceTransform(target_framework, mode, backend_compile):
    print("kornia.contrib.DistanceTransform")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, target=target_framework)

    x = torch.zeros(1, 1, 5, 5)
    x[:,:, 1, 2] = 1
    torch_out = kornia.contrib.DistanceTransform()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = transpiled_kornia.contrib.DistanceTransform()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)

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
        torch.rand(1, 3, 2, 2),
        kornia.feature.HardNet8(pretrained=False),
    )
    trace_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
    test_args = (
        torch.rand(5, 1, 32, 32),
        torch.rand(5, 3, 2, 2),
        kornia.feature.HardNet8(pretrained=False),
    )
    test_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
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


def test_scale_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 2, 3),
        2.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.scale_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laf_scale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_scale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laf_center(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_center,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rotate_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 2, 3)),
        torch.randn((1, 5, 1)),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((2, 10, 2, 3)),
        torch.randn((2, 10, 1)),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.rotate_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laf_orientation(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 2, 3)),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((2, 10, 2, 3)),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_orientation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_set_laf_orientation(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 2, 3)),
        torch.randn((1, 5, 1)),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((2, 10, 2, 3)),
        torch.randn((2, 10, 1)),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.set_laf_orientation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laf_from_center_scale_ori(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.randn(1, 5, 1, 1),
        torch.randn(1, 5, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 2),
        torch.randn(5, 10, 1, 1),
        torch.randn(5, 10, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.laf_from_center_scale_ori,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laf_is_inside_image(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 1, 32, 32),
    )
    trace_kwargs = {'border': 0}
    test_args = (
        torch.rand(2, 10, 2, 3),
        torch.rand(2, 1, 64, 64),
    )
    test_kwargs = {'border': 1}
    _test_function(
        kornia.feature.laf_is_inside_image,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laf_to_three_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.laf_to_three_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laf_from_three_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 6),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 6),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.laf_from_three_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_perspective_transform_lafs(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(3).repeat(1, 1, 1),
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.eye(3).repeat(2, 1, 1),
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.perspective_transform_lafs,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_DenseSIFTDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.DenseSIFTDescriptor")

    if backend_compile:
        pytest.skip()

    TranspiledDenseSIFTDescriptor = ivy.transpile(kornia.feature.DenseSIFTDescriptor, source="torch", target=target_framework)

    x = torch.rand(2, 1, 200, 300)
    torch_out = kornia.feature.DenseSIFTDescriptor()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDenseSIFTDescriptor()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SIFTDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTDescriptor")

    if backend_compile:
        pytest.skip()

    TranspiledSIFTDescriptor = ivy.transpile(kornia.feature.SIFTDescriptor, source="torch", target=target_framework)

    x = torch.rand(23, 1, 41, 41)
    torch_out = kornia.feature.SIFTDescriptor()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSIFTDescriptor()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_MKDDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.MKDDescriptor")

    if backend_compile:
        pytest.skip()

    TranspiledMKDDescriptor = ivy.transpile(kornia.feature.MKDDescriptor, source="torch", target=target_framework)

    x = torch.rand(23, 1, 32, 32)
    torch_out = kornia.feature.MKDDescriptor()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMKDDescriptor()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_HardNet(target_framework, mode, backend_compile):
    print("kornia.feature.HardNet")

    if backend_compile:
        pytest.skip()

    TranspiledHardNet = ivy.transpile(kornia.feature.HardNet, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HardNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHardNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_HardNet8(target_framework, mode, backend_compile):
    print("kornia.feature.HardNet8")

    if backend_compile:
        pytest.skip()

    TranspiledHardNet8 = ivy.transpile(kornia.feature.HardNet8, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HardNet8()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHardNet8()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_HyNet(target_framework, mode, backend_compile):
    print("kornia.feature.HyNet")

    if backend_compile:
        pytest.skip()

    TranspiledHyNet = ivy.transpile(kornia.feature.HyNet, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HyNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHyNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_TFeat(target_framework, mode, backend_compile):
    print("kornia.feature.TFeat")

    if backend_compile:
        pytest.skip()

    TranspiledTFeat = ivy.transpile(kornia.feature.TFeat, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.TFeat()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledTFeat()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_SOSNet(target_framework, mode, backend_compile):
    print("kornia.feature.SOSNet")

    if backend_compile:
        pytest.skip()

    TranspiledSOSNet = ivy.transpile(kornia.feature.SOSNet, source="torch", target=target_framework)

    x = torch.rand(8, 1, 32, 32)
    torch_out = kornia.feature.SOSNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSOSNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_BlobHessian(target_framework, mode, backend_compile):
    print("kornia.feature.BlobHessian")

    if backend_compile:
        pytest.skip()

    TranspiledBlobHessian = ivy.transpile(kornia.feature.BlobHessian, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobHessian()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobHessian()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerGFTT(target_framework, mode, backend_compile):
    print("kornia.feature.CornerGFTT")

    if backend_compile:
        pytest.skip()

    TranspiledCornerGFTT = ivy.transpile(kornia.feature.CornerGFTT, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerGFTT()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledCornerGFTT()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_CornerHarris(target_framework, mode, backend_compile):
    print("kornia.feature.CornerHarris")

    if backend_compile:
        pytest.skip()

    TranspiledCornerHarris = ivy.transpile(kornia.feature.CornerHarris, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.CornerHarris(x)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledCornerHarris(transpiled_x)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoG(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoG")

    if backend_compile:
        pytest.skip()

    TranspiledBlobDoG = ivy.transpile(kornia.feature.BlobDoG, source="torch", target=target_framework)

    x = torch.rand(2, 3, 3, 32, 32)
    torch_out = kornia.feature.BlobDoG()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobDoG()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_BlobDoGSingle(target_framework, mode, backend_compile):
    print("kornia.feature.BlobDoGSingle")

    if backend_compile:
        pytest.skip()

    TranspiledBlobDoGSingle = ivy.transpile(kornia.feature.BlobDoGSingle, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.BlobDoGSingle()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledBlobDoGSingle()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNet")

    if backend_compile:
        pytest.skip()

    TranspiledKeyNet = ivy.transpile(kornia.feature.KeyNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.KeyNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_MultiResolutionDetector(target_framework, mode, backend_compile):
    print("kornia.feature.MultiResolutionDetector")

    if backend_compile:
        pytest.skip()

    TranspiledMultiResolutionDetector = ivy.transpile(kornia.feature.MultiResolutionDetector, source="torch", target=target_framework)
    TranspiledKeyNet = ivy.transpile(kornia.feature.KeyNet, source="torch", target=target_framework)

    model = kornia.feature.KeyNet()
    transpiled_model = TranspiledKeyNet()

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.MultiResolutionDetector(model)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledMultiResolutionDetector(transpiled_model)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_ScaleSpaceDetector(target_framework, mode, backend_compile):
    print("kornia.feature.ScaleSpaceDetector")

    if backend_compile:
        pytest.skip()

    TranspiledScaleSpaceDetector = ivy.transpile(kornia.feature.ScaleSpaceDetector, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32) * 10.
    torch_out = kornia.feature.ScaleSpaceDetector()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledScaleSpaceDetector()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetDetector(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetDetector")

    if backend_compile:
        pytest.skip()

    TranspiledKeyNetDetector = ivy.transpile(kornia.feature.KeyNetDetector, source="torch", target=target_framework)

    x = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.KeyNetDetector()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNetDetector()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.LAFDescriptor")

    if backend_compile:
        pytest.skip()

    TranspiledLAFDescriptor = ivy.transpile(kornia.feature.LAFDescriptor, source="torch", target=target_framework)

    x = torch.rand(1, 1, 64, 64)
    lafs = torch.rand(1, 2, 2, 3)
    torch_out = kornia.feature.LAFDescriptor()(x, lafs)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_lafs = _nest_torch_tensor_to_new_framework(lafs, target_framework)
    transpiled_out = TranspiledLAFDescriptor()(transpiled_x, transpiled_lafs)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SOLD2(target_framework, mode, backend_compile):
    print("kornia.feature.SOLD2")

    if backend_compile:
        pytest.skip()

    TranspiledSOLD2 = ivy.transpile(kornia.feature.SOLD2, source="torch", target=target_framework)

    x = torch.rand(1, 1, 512, 512)
    torch_out = kornia.feature.SOLD2(pretrained=False)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSOLD2(pretrained=False)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LocalFeature(target_framework, mode, backend_compile):
    print("kornia.feature.LocalFeature")

    if backend_compile:
        pytest.skip()

    TranspiledKeyNetDetector = ivy.transpile(
        kornia.feature.KeyNetDetector, source="torch", target=target_framework
    )
    TranspiledLAFDescriptor = ivy.transpile(
        kornia.feature.LAFDescriptor, source="torch", target=target_framework
    )
    TranspiledLocalFeature = ivy.transpile(
        kornia.feature.LocalFeature, source="torch", target=target_framework
    )

    torch_detector = kornia.feature.KeyNetDetector()
    torch_descriptor = kornia.feature.LAFDescriptor()
    transpiled_detector = TranspiledKeyNetDetector()
    transpiled_descriptor = TranspiledLAFDescriptor()

    x = torch.rand(1, 1, 128, 128)
    torch_out = kornia.feature.LocalFeature(torch_detector, torch_descriptor)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledLocalFeature(transpiled_detector, transpiled_descriptor)(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_SOLD2_detector(target_framework, mode, backend_compile):
    print("kornia.feature.SOLD2_detector")

    if backend_compile:
        pytest.skip()

    TranspiledSOLD2Detector = ivy.transpile(kornia.feature.SOLD2_detector, source="torch", target=target_framework)

    x = torch.rand(1, 1, 512, 512)
    torch_out = kornia.feature.SOLD2_detector(pretrained=False)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSOLD2Detector(pretrained=False)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DeDoDe(target_framework, mode, backend_compile):
    print("kornia.feature.DeDoDe")

    if backend_compile:
        pytest.skip()

    TranspiledDeDoDe = ivy.transpile(kornia.feature.DeDoDe, source="torch", target=target_framework)

    x = torch.rand(1, 3, 256, 256)
    torch_out = kornia.feature.DeDoDe(amp_dtype=torch.float32)(x)

    ivy.set_backend(target_framework)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDeDoDe(amp_dtype=ivy.as_native_dtype("float32"))(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_DISK(target_framework, mode, backend_compile):
    print("kornia.feature.DISK")

    if backend_compile:
        pytest.skip()

    TranspiledDISK = ivy.transpile(kornia.feature.DISK, source="torch", target=target_framework)

    x = torch.rand(1, 3, 256, 256)
    torch_out = kornia.feature.DISK()(x)[0]

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDISK()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out.keypoints, transpiled_out.keypoints)
    _to_numpy_and_shape_allclose(torch_out.descriptors, transpiled_out.descriptors)
    _to_numpy_and_shape_allclose(torch_out.detection_scores, transpiled_out.detection_scores)


def test_DISKFeatures(target_framework, mode, backend_compile):
    print("kornia.feature.DISKFeatures")

    if backend_compile:
        pytest.skip()

    keypoints = torch.rand(100, 2)
    descriptors = torch.rand(100, 128)
    detection_scores = torch.rand(100)
    torch_out = kornia.feature.DISKFeatures(keypoints, descriptors, detection_scores)

    transpiled_keypoints = _nest_torch_tensor_to_new_framework(keypoints, target_framework)
    transpiled_descriptors = _nest_torch_tensor_to_new_framework(descriptors, target_framework)
    transpiled_detection_scores = _nest_torch_tensor_to_new_framework(detection_scores, target_framework)
    transpiled_out = kornia.feature.DISKFeatures(transpiled_keypoints, transpiled_descriptors, transpiled_detection_scores)

    _to_numpy_and_shape_allclose(torch_out.keypoints, transpiled_out.keypoints)
    _to_numpy_and_shape_allclose(torch_out.descriptors, transpiled_out.descriptors)
    _to_numpy_and_shape_allclose(torch_out.detection_scores, transpiled_out.detection_scores)


def test_SIFTFeature(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTFeature")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledSIFTFeature = ivy.transpile(kornia.feature.SIFTFeature, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    x = torch.rand(1, 1, 256, 256)
    torch_out = kornia.feature.SIFTFeature(num_features=5000)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSIFTFeature(num_features=5000)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SIFTFeatureScaleSpace(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTFeatureScaleSpace")

    if backend_compile:
        pytest.skip()

    TranspiledSIFTFeatureScaleSpace = ivy.transpile(kornia.feature.SIFTFeatureScaleSpace, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    torch_out = kornia.feature.SIFTFeatureScaleSpace(num_features=5000)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledSIFTFeatureScaleSpace(num_features=5000)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_GFTTAffNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.GFTTAffNetHardNet")

    if backend_compile:
        pytest.skip()

    TranspiledGFTTAffNetHardNet = ivy.transpile(kornia.feature.GFTTAffNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    torch_out = kornia.feature.GFTTAffNetHardNet(num_features=5000)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledGFTTAffNetHardNet(num_features=5000)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetAffNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetAffNetHardNet")

    if backend_compile:
        pytest.skip()

    TranspiledKeyNetAffNetHardNet = ivy.transpile(kornia.feature.KeyNetAffNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    torch_out = kornia.feature.KeyNetAffNetHardNet(num_features=5000)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNetAffNetHardNet(num_features=5000)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetHardNet")

    if backend_compile:
        pytest.skip()

    TranspiledKeyNetHardNet = ivy.transpile(kornia.feature.KeyNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    torch_out = kornia.feature.KeyNetHardNet(num_features=5000)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledKeyNetHardNet(num_features=5000)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DescriptorMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.DescriptorMatcher")

    if backend_compile:
        pytest.skip()

    TranspiledDescriptorMatcher = ivy.transpile(kornia.feature.DescriptorMatcher, source="torch", target=target_framework)

    x1 = torch.rand(2, 256)
    x2 = torch.rand(2, 256)
    torch_out = kornia.feature.DescriptorMatcher('snn', 0.8)(x1, x2)

    transpiled_x1 = _nest_torch_tensor_to_new_framework(x1, target_framework)
    transpiled_x2 = _nest_torch_tensor_to_new_framework(x2, target_framework)
    transpiled_out = TranspiledDescriptorMatcher('snn', 0.8)(transpiled_x1, transpiled_x2)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_GeometryAwareDescriptorMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.GeometryAwareDescriptorMatcher")

    if backend_compile:
        pytest.skip()

    TranspiledGeometryAwareDescriptorMatcher = ivy.transpile(kornia.feature.GeometryAwareDescriptorMatcher, source="torch", target=target_framework)

    torch_args = (
        torch.rand(2, 256),
        torch.rand(2, 256),
        torch.rand(2, 2, 2, 3),
        torch.rand(2, 2, 2, 3),
    )
    torch_out = kornia.feature.GeometryAwareDescriptorMatcher('fginn')(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = TranspiledGeometryAwareDescriptorMatcher('fginn')(*transpiled_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LocalFeatureMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.LocalFeatureMatcher")

    if backend_compile:
        pytest.skip()

    TranspiledGFTTAffNetHardNet = ivy.transpile(kornia.feature.GFTTAffNetHardNet, source="torch", target=target_framework)
    TranspiledDescriptorMatcher = ivy.transpile(kornia.feature.DescriptorMatcher, source="torch", target=target_framework)
    TranspiledLocalFeatureMatcher = ivy.transpile(kornia.feature.LocalFeatureMatcher, source="torch", target=target_framework)

    data = {
        "image0": torch.rand(1, 1, 320, 200),
        "image1": torch.rand(1, 1, 128, 128),
    }
    torch_local_feature = kornia.feature.GFTTAffNetHardNet(10)
    torch_matcher = kornia.feature.DescriptorMatcher('snn', 0.8)
    torch_out = kornia.feature.LocalFeatureMatcher(torch_local_feature, torch_matcher)(data)

    transpiled_local_feature = TranspiledGFTTAffNetHardNet(10)
    transpiled_matcher = TranspiledDescriptorMatcher('snn', 0.8)
    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)
    transpiled_out = TranspiledLocalFeatureMatcher(transpiled_local_feature, transpiled_matcher)(transpiled_data)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LightGlueMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.LightGlueMatcher")

    if backend_compile:
        pytest.skip()

    TranspiledLightGlueMatcher = ivy.transpile(kornia.feature.LightGlueMatcher, source="torch", target=target_framework)

    torch_args = (
        torch.rand(2, 128),
        torch.rand(5, 128),
        torch.rand(1, 2, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    torch_out = kornia.feature.LightGlueMatcher('disk')(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = TranspiledLightGlueMatcher('disk')(*transpiled_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LightGlue(target_framework, mode, backend_compile):
    print("kornia.feature.LightGlue")

    if backend_compile:
        pytest.skip()

    TranspiledLightGlue = ivy.transpile(kornia.feature.LightGlue, source="torch", target=target_framework)

    data = {
        "image0": {
            "keypoints": torch.rand(1, 100, 2),
            "descriptors": torch.rand(1, 100, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
        "image1": {
            "keypoints": torch.rand(1, 120, 2),
            "descriptors": torch.rand(1, 120, 256),
            "image_size": torch.tensor([[640, 480]]),
        }
    }
    torch_out = kornia.feature.LightGlue(features='superpoint')(data)

    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)
    transpiled_out = TranspiledLightGlue(features='superpoint')(transpiled_data)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LoFTR(target_framework, mode, backend_compile):
    print("kornia.feature.LoFTR")

    if backend_compile:
        pytest.skip()

    TranspiledLoFTR = ivy.transpile(kornia.feature.LoFTR, source="torch", target=target_framework)

    data = {"image0": torch.rand(1, 1, 320, 200), "image1": torch.rand(1, 1, 128, 128)}
    torch_out = kornia.feature.LoFTR(None)(data)

    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)
    transpiled_out = TranspiledLoFTR(None)(transpiled_data)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PassLAF(target_framework, mode, backend_compile):
    print("kornia.feature.PassLAF")

    if backend_compile:
        pytest.skip()

    TranspiledPassLAF = ivy.transpile(kornia.feature.PassLAF, source="torch", target=target_framework)

    laf = torch.rand(1, 2, 3, 3)
    img = torch.rand(1, 3, 32, 32)
    torch_out = kornia.feature.PassLAF()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledPassLAF()(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PatchAffineShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.PatchAffineShapeEstimator")

    if backend_compile:
        pytest.skip()

    TranspiledPatchAffineShapeEstimator = ivy.transpile(kornia.feature.PatchAffineShapeEstimator, source="torch", target=target_framework)

    patch = torch.rand(1, 1, 19, 19)
    torch_out = kornia.feature.PatchAffineShapeEstimator()(patch)

    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    transpiled_out = TranspiledPatchAffineShapeEstimator()(transpiled_patch)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LAFAffineShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.LAFAffineShapeEstimator")

    if backend_compile:
        pytest.skip()

    TranspiledLAFAffineShapeEstimator = ivy.transpile(kornia.feature.LAFAffineShapeEstimator, source="torch", target=target_framework)

    laf = torch.rand(1, 2, 2, 3)
    img = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.LAFAffineShapeEstimator()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledLAFAffineShapeEstimator()(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LAFOrienter(target_framework, mode, backend_compile):
    print("kornia.feature.LAFOrienter")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledLAFOrienter = ivy.transpile(kornia.feature.LAFOrienter, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    laf = torch.rand(1, 2, 2, 3)
    img = torch.rand(1, 1, 32, 32)
    torch_out = kornia.feature.LAFOrienter()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledLAFOrienter()(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PatchDominantGradientOrientation(target_framework, mode, backend_compile):
    print("kornia.feature.PatchDominantGradientOrientation")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledPatchDominantGradientOrientation = ivy.transpile(kornia.feature.PatchDominantGradientOrientation, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    patch = torch.rand(10, 1, 32, 32)
    torch_out = kornia.feature.PatchDominantGradientOrientation()(patch)

    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    transpiled_out = TranspiledPatchDominantGradientOrientation()(transpiled_patch)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_OriNet(target_framework, mode, backend_compile):
    print("kornia.feature.OriNet")

    if backend_compile:
        pytest.skip()

    TranspiledOriNet = ivy.transpile(kornia.feature.OriNet, source="torch", target=target_framework)

    patch = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.OriNet()(patch)

    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    transpiled_out = TranspiledOriNet()(transpiled_patch)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LAFAffNetShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.LAFAffNetShapeEstimator")

    if backend_compile:
        pytest.skip()

    import os
    flag = os.environ.get("APPLY_TRANSPOSE_OPTIMIZATION")
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "false"

    TranspiledLAFAffNetShapeEstimator = ivy.transpile(kornia.feature.LAFAffNetShapeEstimator, source="torch", target=target_framework)

    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = flag

    laf = torch.rand(10, 2, 2, 3)
    img = torch.rand(10, 1, 32, 32)
    torch_out = kornia.feature.LAFAffNetShapeEstimator()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledLAFAffNetShapeEstimator()(transpiled_laf, transpiled_img)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_FilterResponseNorm2d(target_framework, mode, backend_compile):
    print("kornia.feature.FilterResponseNorm2d")

    if backend_compile:
        pytest.skip()

    TranspiledFilterResponseNorm2d = ivy.transpile(kornia.feature.FilterResponseNorm2d, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.FilterResponseNorm2d(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledFilterResponseNorm2d(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_TLU(target_framework, mode, backend_compile):
    print("kornia.feature.TLU")

    if backend_compile:
        pytest.skip()

    TranspiledTLU = ivy.transpile(kornia.feature.TLU, source="torch", target=target_framework)

    x = torch.rand(1, 3, 8, 8)
    torch_out = kornia.feature.TLU(3)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledTLU(3)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DeFMO(target_framework, mode, backend_compile):
    print("kornia.feature.DeFMO")

    if backend_compile:
        pytest.skip()

    TranspiledDeFMO = ivy.transpile(kornia.feature.DeFMO, source="torch", target=target_framework)

    x = torch.rand(2, 6, 240, 320)
    torch_out = kornia.feature.DeFMO()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledDeFMO()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)

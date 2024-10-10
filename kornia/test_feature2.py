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

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledDenseSIFTDescriptor = ivy.transpile(kornia.feature.DenseSIFTDescriptor, source="torch", target=target_framework)

    x = torch.rand(2, 1, 200, 300)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.DenseSIFTDescriptor()
    torch_out = model(x)

    transpiled_model = TranspiledDenseSIFTDescriptor()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SIFTDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTDescriptor")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledSIFTDescriptor = ivy.transpile(kornia.feature.SIFTDescriptor, source="torch", target=target_framework)

    x = torch.rand(23, 1, 41, 41)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.SIFTDescriptor()
    torch_out = model(x)

    transpiled_model = TranspiledSIFTDescriptor()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_MKDDescriptor(target_framework, mode, backend_compile):
    print("kornia.feature.MKDDescriptor")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledMKDDescriptor = ivy.transpile(kornia.feature.MKDDescriptor, source="torch", target=target_framework)

    x = torch.rand(23, 1, 32, 32)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.MKDDescriptor()
    torch_out = model(x)

    transpiled_model = TranspiledMKDDescriptor()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_HardNet(target_framework, mode, backend_compile):
    print("kornia.feature.HardNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledHardNet = ivy.transpile(kornia.feature.HardNet, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HardNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHardNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_HardNet8(target_framework, mode, backend_compile):
    print("kornia.feature.HardNet8")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledHardNet8 = ivy.transpile(kornia.feature.HardNet8, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HardNet8()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHardNet8()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_HyNet(target_framework, mode, backend_compile):
    print("kornia.feature.HyNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledHyNet = ivy.transpile(kornia.feature.HyNet, source="torch", target=target_framework)

    x = torch.rand(16, 1, 32, 32)
    torch_out = kornia.feature.HyNet()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHyNet()(transpiled_x)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)

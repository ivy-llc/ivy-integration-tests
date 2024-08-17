from helpers import (
    _check_allclose,
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

def test_Keypoints(target_framework, mode, backend_compile):
    print("kornia.geometry.keypoints.Keypoints")

    if backend_compile:
        pytest.skip()

    TranspiledKeypoints = ivy.transpile(kornia.geometry.keypoints.Keypoints, source="torch", target=target_framework)

    torch_init_args = (
        torch.rand((10, 3, 2)),
    )
    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)

    torch_keypoints = kornia.geometry.keypoints.Keypoints(*torch_init_args)
    transpiled_keypoints = TranspiledKeypoints(*transpiled_init_args)

    # test .data
    orig_np = _nest_array_to_numpy(torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)

    # test .to_tensor
    orig_np = _nest_array_to_numpy(torch_keypoints.to_tensor())
    transpiled_np = _nest_array_to_numpy(transpiled_keypoints.to_tensor())
    _check_allclose(orig_np, transpiled_np)

    # test .transform_keypoints()
    torch_args = (
        torch.rand((10, 3, 3)),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    new_torch_keypoints = torch_keypoints.transform_keypoints(*torch_args)
    new_transpiled_keypoints = transpiled_keypoints.transform_keypoints(*transpiled_args)

    orig_np = _nest_array_to_numpy(new_torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(new_transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)

    # test .pad()
    torch_args = (
        torch.tensor([[1, 2, 3, 4]]).repeat(10, 1),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    new_torch_keypoints = torch_keypoints.pad(*torch_args)
    new_transpiled_keypoints = transpiled_keypoints.pad(*transpiled_args)

    orig_np = _nest_array_to_numpy(new_torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(new_transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)


def test_Keypoints3D(target_framework, mode, backend_compile):
    print("kornia.geometry.keypoints.Keypoints3D")

    if backend_compile:
        pytest.skip()

    TranspiledKeypoints3D = ivy.transpile(kornia.geometry.keypoints.Keypoints3D, source="torch", target=target_framework)

    torch_init_args = (
        torch.rand((10, 5, 3)),
    )
    transpiled_init_args = _nest_torch_tensor_to_new_framework(torch_init_args, target_framework)

    torch_keypoints = kornia.geometry.keypoints.Keypoints3D(*torch_init_args)
    transpiled_keypoints = TranspiledKeypoints3D(*transpiled_init_args)

    # test .data
    orig_np = _nest_array_to_numpy(torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)

    # test .to_tensor
    orig_np = _nest_array_to_numpy(torch_keypoints.to_tensor())
    transpiled_np = _nest_array_to_numpy(transpiled_keypoints.to_tensor())
    _check_allclose(orig_np, transpiled_np)

    # test .transform_keypoints()
    torch_args = (
        torch.rand((10, 3, 3)),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    new_torch_keypoints = torch_keypoints.transform_keypoints(*torch_args)
    new_transpiled_keypoints = transpiled_keypoints.transform_keypoints(*transpiled_args)

    orig_np = _nest_array_to_numpy(new_torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(new_transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)

    # test .pad()
    torch_args = (
        torch.tensor([[1, 2, 3, 4, 5, 6]]).repeat(10, 1),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    new_torch_keypoints = torch_keypoints.pad(*torch_args)
    new_transpiled_keypoints = transpiled_keypoints.pad(*transpiled_args)

    orig_np = _nest_array_to_numpy(new_torch_keypoints.data)
    transpiled_np = _nest_array_to_numpy(new_transpiled_keypoints.data)
    _check_allclose(orig_np, transpiled_np)

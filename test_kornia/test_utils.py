from helpers import (
    _nest_torch_tensor_to_new_framework,
    _test_function,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import numpy as np
import pytest
import tempfile
import torch


# Tests #
# ----- #

def test_draw_line(target_framework, mode, backend_compile):
    trace_args = (
        torch.zeros((1, 8, 8)),
        torch.tensor([6, 4]),
        torch.tensor([1, 4]),
        torch.tensor([255]),
    )
    trace_kwargs = {}
    test_args = (
        torch.zeros((1, 8, 8)),
        torch.tensor([0, 2]),
        torch.tensor([5, 1]),
        torch.tensor([255]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_line,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_draw_rectangle(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 10, 12),
        torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3, 10, 12),
        torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]], [[2, 2, 6, 6]]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_rectangle,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_draw_convex_polygon(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 12, 16),
        torch.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]]]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 3, 12, 16),
        torch.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]], [[3, 3], [10, 3], [10, 7], [3, 7]]]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_convex_polygon,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_create_meshgrid(target_framework, mode, backend_compile):
    trace_args = (2, 2)
    trace_kwargs = {}
    test_args = (4, 4)
    test_kwargs = {}
    _test_function(
        kornia.utils.create_meshgrid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_create_meshgrid3d(target_framework, mode, backend_compile):
    trace_args = (2, 2, 2)
    trace_kwargs = {}
    test_args = (4, 4, 4)
    test_kwargs = {}
    _test_function(
        kornia.utils.create_meshgrid3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_one_hot(target_framework, mode, backend_compile):
    trace_args = (
        torch.LongTensor([[[0, 1], [2, 0]]]),
        3,
        torch.device('cpu'),
        torch.int64,
    )
    trace_kwargs = {}
    test_args = (
        torch.LongTensor([[[1, 2], [0, 1]]]),
        5,
        torch.device('cpu'),
        torch.int64,
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.one_hot,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_tensor_to_image(target_framework, mode, backend_compile):
    print("kornia.utils.tensor_to_image")

    if backend_compile:
        pytest.skip()

    transpiled_func = ivy.transpile(kornia.utils.tensor_to_image, source="torch", target=target_framework)

    tensor = torch.ones(1, 3, 3)
    transpiled_tensor = _nest_torch_tensor_to_new_framework(tensor, target_framework)
    torch_image = kornia.utils.tensor_to_image(tensor)
    transpiled_image = transpiled_func(transpiled_tensor)
    _to_numpy_and_allclose(torch_image, transpiled_image)

    tensor = torch.ones(3, 4, 4)
    transpiled_tensor = _nest_torch_tensor_to_new_framework(tensor, target_framework)
    torch_image = kornia.utils.tensor_to_image(tensor, force_contiguous=True)
    transpiled_image = transpiled_func(transpiled_tensor, force_contiguous=True)
    _to_numpy_and_allclose(torch_image, transpiled_image)


def test_image_to_tensor(target_framework, mode, backend_compile):
    print("kornia.utils.image_to_tensor")

    if backend_compile:
        pytest.skip()

    transpiled_func = ivy.transpile(kornia.utils.image_to_tensor, source="torch", target=target_framework)

    image = np.ones((3, 3))
    torch_tensor = kornia.utils.image_to_tensor(image)
    transpiled_tensor = transpiled_func(image)
    _to_numpy_and_allclose(torch_tensor, transpiled_tensor)

    image = np.ones((4, 4, 3))
    torch_tensor = kornia.utils.image_to_tensor(image, keepdim=False)
    transpiled_tensor = transpiled_func(image, keepdim=False)
    _to_numpy_and_allclose(torch_tensor, transpiled_tensor)


def test_image_list_to_tensor(target_framework, mode, backend_compile):
    print("kornia.utils.image_list_to_tensor")

    if backend_compile:
        pytest.skip()

    images = [np.ones((4, 4, 1)), np.zeros((4, 4, 1))]

    transpiled_func = ivy.transpile(kornia.utils.image_list_to_tensor, source="torch", target=target_framework)
    torch_tensor = kornia.utils.image_list_to_tensor(images)
    transpiled_tensor = transpiled_func(images)

    _to_numpy_and_allclose(torch_tensor, transpiled_tensor)


def test_image_to_string(target_framework, mode, backend_compile):
    print("kornia.utils.image_to_string")

    if backend_compile:
        pytest.skip()

    image = torch.rand(3, 16, 16)
    transpiled_image = _nest_torch_tensor_to_new_framework(image, target_framework)

    transpiled_func = ivy.transpile(kornia.utils.image_to_string, source="torch", target=target_framework)
    torch_str = kornia.utils.image_to_string(image)
    transpiled_str = transpiled_func(transpiled_image)

    assert torch_str == transpiled_str, "strings mismatched"


# commented due to capsys not allowing fn name to be correctly logged
# def test_print_image(target_framework, mode, backend_compile, capsys):
#     print("kornia.utils.print_image")

#     if backend_compile:
#         pytest.skip()

#     image = torch.rand(3, 16, 16)
#     transpiled_image = _nest_torch_tensor_to_new_framework(image, target_framework)

#     transpiled_func = ivy.transpile(kornia.utils.print_image, source="torch", target=target_framework)

#     # Capture the output of print_image, and check they are the same
#     kornia.utils.print_image(image)
#     torch_output = capsys.readouterr().out

#     transpiled_func(transpiled_image)
#     transpiled_output = capsys.readouterr().out

#     assert torch_output == transpiled_output


def test_save_pointcloud_ply(target_framework, mode, backend_compile):
    print("kornia.utils.save_pointcloud_ply")

    if backend_compile:
        pytest.skip()

    pointcloud = torch.rand(100, 3)
    transpiled_pointcloud = _nest_torch_tensor_to_new_framework(pointcloud, target_framework)

    with tempfile.NamedTemporaryFile(suffix=".ply") as temp_file:
        filename = temp_file.name

        transpiled_save_pointcloud = ivy.transpile(kornia.utils.save_pointcloud_ply, source="torch", target=target_framework)
        transpiled_load_pointcloud = ivy.transpile(kornia.utils.load_pointcloud_ply, source="torch", target=target_framework)

        # Save and load pointcloud to ensure both steps work correctly
        transpiled_save_pointcloud(filename, transpiled_pointcloud)
        loaded_pointcloud = transpiled_load_pointcloud(filename)
        _to_numpy_and_allclose(pointcloud, loaded_pointcloud)


# Note: Commenting the following tests out because these contain torch util code
# like retreiving torch.cuda related attributes or torch.backends.mps device related
# stuff which we choose to not transpile unless there's a strong case made
# against it in which case we can come back and uncomment these out
# def test_get_cuda_device_if_available(target_framework, mode, backend_compile):
#     print("kornia.utils.get_cuda_device_if_available")

#     if backend_compile:
#         pytest.skip()

#     transpiled_get_cuda_device_if_available = ivy.transpile(kornia.utils.get_cuda_device_if_available, source="torch", target=target_framework)

#     torch_device = kornia.utils.get_cuda_device_if_available()
#     transpiled_device = transpiled_get_cuda_device_if_available()

#     assert torch_device
#     assert transpiled_device


# def test_get_mps_device_if_available(target_framework, mode, backend_compile):
#     print("kornia.utils.get_mps_device_if_available")

#     if backend_compile:
#         pytest.skip()

#     transpiled_get_mps_device_if_available = ivy.transpile(kornia.utils.get_mps_device_if_available, source="torch", target=target_framework)

#     torch_device = kornia.utils.get_mps_device_if_available()
#     transpiled_device = transpiled_get_mps_device_if_available()

#     assert torch_device
#     assert transpiled_device


# def test_get_cuda_or_mps_device_if_available(target_framework, mode, backend_compile):
#     print("kornia.utils.get_cuda_or_mps_device_if_available")

#     if backend_compile:
#         pytest.skip()

#     transpiled_get_cuda_or_mps_device_if_available = ivy.transpile(kornia.utils.get_cuda_or_mps_device_if_available, source="torch", target=target_framework)

#     torch_device = kornia.utils.get_cuda_or_mps_device_if_available()
#     transpiled_device = transpiled_get_cuda_or_mps_device_if_available()

#     assert torch_device
#     assert transpiled_device


def test_map_location_to_cpu(target_framework, mode, backend_compile):
    print("kornia.utils.map_location_to_cpu")

    if backend_compile:
        pytest.skip()

    tensor = torch.rand(3, 3)
    transpiled_tensor = _nest_torch_tensor_to_new_framework(tensor, target_framework)

    transpiled_func = ivy.transpile(kornia.utils.map_location_to_cpu, source="torch", target=target_framework)

    torch_mapped = kornia.utils.map_location_to_cpu(tensor)
    transpiled_mapped = transpiled_func(transpiled_tensor)

    _to_numpy_and_allclose(torch_mapped, transpiled_mapped)


def test_is_autocast_enabled(target_framework, mode, backend_compile):
    print("kornia.utils.is_autocast_enabled")

    if backend_compile:
        pytest.skip()

    transpiled_func = ivy.transpile(kornia.utils.is_autocast_enabled, source="torch", target=target_framework)

    torch_autocast_enabled = kornia.utils.is_autocast_enabled()
    transpiled_autocast_enabled = transpiled_func()

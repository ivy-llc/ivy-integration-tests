from helpers import (
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
    _to_numpy_and_shape_allclose,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_fit_line(target_framework, mode, backend_compile):
    print("kornia.geometry.line.fit_line")

    if backend_compile:
        pytest.skip()

    transpiled_fit_line = ivy.transpile(kornia.geometry.line.fit_line, source="torch", target=target_framework)

    torch_args = (
        torch.rand(2, 10, 3),
        torch.ones(2, 10),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_line = kornia.geometry.line.fit_line(*torch_args)
    transpiled_line = transpiled_fit_line(*transpiled_args)

    # NOTE: numerical instability in svd()/lu() leads to logits not being allclose
    deterministic = False
    tolerance = 1e-3

    if deterministic:
        _to_numpy_and_allclose(torch_line.origin, transpiled_line.origin, tolerance=tolerance)
        _to_numpy_and_allclose(torch_line.direction, transpiled_line.direction, tolerance=tolerance)
    else:
        _to_numpy_and_shape_allclose(torch_line.origin, transpiled_line.origin, tolerance=tolerance)
        _to_numpy_and_shape_allclose(torch_line.direction, transpiled_line.direction, tolerance=tolerance)


def test_ParametrizedLine(target_framework, mode, backend_compile):
    print("kornia.geometry.line.ParametrizedLine")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    origin = torch.tensor([0.0, 0.0], requires_grad=True)
    direction = torch.tensor([1.0, 1.0], requires_grad=True)
    torch_line = kornia.geometry.line.ParametrizedLine(origin, direction)

    transpiled_origin = _nest_torch_tensor_to_new_framework(origin, target_framework)
    transpiled_direction = _nest_torch_tensor_to_new_framework(direction, target_framework)
    transpiled_line = transpiled_kornia.geometry.line.ParametrizedLine(transpiled_origin, transpiled_direction)

    # Test .dim()
    torch_dim = torch_line.dim()
    transpiled_dim = transpiled_line.dim()
    assert torch_dim == transpiled_dim

    # Test .direction property
    _to_numpy_and_allclose(torch_line.direction, transpiled_line.direction)

    # Test .origin property
    _to_numpy_and_allclose(torch_line.origin, transpiled_line.origin)

    # Test .distance()
    point = torch.tensor([1.0, 0.0], requires_grad=True)
    transpiled_point = _nest_torch_tensor_to_new_framework(point, target_framework)

    torch_distance = torch_line.distance(point)
    transpiled_distance = transpiled_line.distance(transpiled_point)
    _to_numpy_and_allclose(torch_distance, transpiled_distance)

    # Test .point_at()
    t_value = torch.tensor(0.5)
    transpiled_t_value = _nest_torch_tensor_to_new_framework(t_value, target_framework)

    torch_point_at = torch_line.point_at(t_value)
    transpiled_point_at = transpiled_line.point_at(transpiled_t_value)
    _to_numpy_and_allclose(torch_point_at, transpiled_point_at)

    # Test .projection()
    torch_projection = torch_line.projection(point)
    transpiled_projection = transpiled_line.projection(transpiled_point)
    _to_numpy_and_allclose(torch_projection, transpiled_projection)

    # Test .squared_distance()
    torch_squared_distance = torch_line.squared_distance(point)
    transpiled_squared_distance = transpiled_line.squared_distance(transpiled_point)
    _to_numpy_and_allclose(torch_squared_distance, transpiled_squared_distance)

    # Test class method .through()
    p0 = torch.tensor([0.0, 0.0], requires_grad=True)
    p1 = torch.tensor([1.0, 1.0], requires_grad=True)
    transpiled_p0 = _nest_torch_tensor_to_new_framework(p0, target_framework)
    transpiled_p1 = _nest_torch_tensor_to_new_framework(p1, target_framework)

    torch_line_through = kornia.geometry.line.ParametrizedLine.through(p0, p1)
    transpiled_line_through = transpiled_kornia.geometry.line.ParametrizedLine.through(transpiled_p0, transpiled_p1)
    _to_numpy_and_allclose(torch_line_through.origin, transpiled_line_through.origin)
    _to_numpy_and_allclose(torch_line_through.direction, transpiled_line_through.direction)

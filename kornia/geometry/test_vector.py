from helpers import (
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_Vector3(target_framework, mode, backend_compile):
    print("kornia.geometry.vector.Vector3")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    # Initialize a Vector3 with a tensor
    vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    torch_vector3 = kornia.geometry.vector.Vector3(vector)

    # Transpile the Vector3 class
    TranspiledVector3 = ivy.transpile(kornia.geometry.vector.Vector3, source="torch", target=target_framework)

    transpiled_vector = _nest_torch_tensor_to_new_framework(vector, target_framework)
    transpiled_vector3 = TranspiledVector3(transpiled_vector)

    # Test .x, .y, .z properties
    _to_numpy_and_allclose(torch_vector3.x, transpiled_vector3.x)
    _to_numpy_and_allclose(torch_vector3.y, transpiled_vector3.y)
    _to_numpy_and_allclose(torch_vector3.z, transpiled_vector3.z)

    # Test .normalized()
    torch_normalized = torch_vector3.normalized()
    transpiled_normalized = transpiled_vector3.normalized()
    _to_numpy_and_allclose(torch_normalized.data, transpiled_normalized.data)

    # Test .dot()
    another_vector = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    transpiled_another_vector = _nest_torch_tensor_to_new_framework(another_vector, target_framework)

    torch_dot = torch_vector3.dot(kornia.geometry.vector.Vector3(another_vector))
    transpiled_dot = transpiled_vector3.dot(TranspiledVector3(transpiled_another_vector))
    _to_numpy_and_allclose(torch_dot.data, transpiled_dot.data)

    # Test .squared_norm()
    torch_squared_norm = torch_vector3.squared_norm()
    transpiled_squared_norm = transpiled_vector3.squared_norm()
    _to_numpy_and_allclose(torch_squared_norm.data, transpiled_squared_norm.data)

    # Test class method .random()
    torch_random_vector3 = kornia.geometry.vector.Vector3.random()
    transpiled_random_vector3 = TranspiledVector3.random()
    _check_shape_allclose(torch_random_vector3.data.numpy(), _nest_array_to_numpy(transpiled_random_vector3.data))

    # Test class method .from_coords()
    torch_vector3_from_coords = kornia.geometry.vector.Vector3.from_coords(1.0, 2.0, 3.0)
    transpiled_vector3_from_coords = TranspiledVector3.from_coords(1.0, 2.0, 3.0)
    _to_numpy_and_allclose(torch_vector3_from_coords.data, transpiled_vector3_from_coords.data)


def test_Vector2(target_framework, mode, backend_compile):
    print("kornia.geometry.vector.Vector2")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    # Initialize a Vector2 with a tensor
    vector = torch.tensor([1.0, 2.0], requires_grad=True)
    torch_vector2 = kornia.geometry.vector.Vector2(vector)

    # Transpile the Vector2 class
    TranspiledVector2 = ivy.transpile(kornia.geometry.vector.Vector2, source="torch", target=target_framework)

    transpiled_vector = _nest_torch_tensor_to_new_framework(vector, target_framework)
    transpiled_vector2 = TranspiledVector2(transpiled_vector)

    # Test .x, .y properties
    _to_numpy_and_allclose(torch_vector2.x, transpiled_vector2.x)
    _to_numpy_and_allclose(torch_vector2.y, transpiled_vector2.y)

    # Test .normalized()
    torch_normalized = torch_vector2.normalized()
    transpiled_normalized = transpiled_vector2.normalized()
    _to_numpy_and_allclose(torch_normalized.data, transpiled_normalized.data)

    # Test .dot()
    another_vector = torch.tensor([3.0, 4.0], requires_grad=True)
    transpiled_another_vector = _nest_torch_tensor_to_new_framework(another_vector, target_framework)

    torch_dot = torch_vector2.dot(kornia.geometry.vector.Vector2(another_vector))
    transpiled_dot = transpiled_vector2.dot(TranspiledVector2(transpiled_another_vector))
    _to_numpy_and_allclose(torch_dot.data, transpiled_dot.data)

    # Test .squared_norm()
    torch_squared_norm = torch_vector2.squared_norm()
    transpiled_squared_norm = transpiled_vector2.squared_norm()
    _to_numpy_and_allclose(torch_squared_norm.data, transpiled_squared_norm.data)

    # Test class method .random()
    torch_random_vector2 = kornia.geometry.vector.Vector2.random()
    transpiled_random_vector2 = TranspiledVector2.random()
    _check_shape_allclose(torch_random_vector2.data.numpy(), _nest_array_to_numpy(transpiled_random_vector2.data))

    # Test class method .from_coords()
    torch_vector2_from_coords = kornia.geometry.vector.Vector2.from_coords(1.0, 2.0)
    transpiled_vector2_from_coords = TranspiledVector2.from_coords(1.0, 2.0)
    _to_numpy_and_allclose(torch_vector2_from_coords.data, transpiled_vector2_from_coords.data)

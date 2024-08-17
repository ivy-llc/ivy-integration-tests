from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
)

import ivy
from kornia.geometry.quaternion import Quaternion
import pytest
import torch


# Tests #
# ----- #

def test_Quaternion(target_framework, mode, backend_compile):
    print("kornia.geometry.quaternion.Quaternion")

    if backend_compile:
        pytest.skip()

    TranspiledQuaternion = ivy.transpile(Quaternion, source="torch", target=target_framework)

    # test Quaternion.identity

    torch_q = Quaternion.identity(batch_size=4)
    transpiled_q = TranspiledQuaternion.identity(batch_size=4)

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.__add__

    torch_q1 = Quaternion.identity()
    torch_q2 = Quaternion(torch.tensor([2., 0., 1., 1.]))
    torch_q3 = torch_q1 + torch_q2
    transpiled_q1 = TranspiledQuaternion.identity()
    transpiled_q2 = TranspiledQuaternion(_array_to_new_backend(torch.tensor([2., 0., 1., 1.])))
    transpiled_q3 = transpiled_q1 + transpiled_q2

    orig_np = _nest_array_to_numpy(torch_q3.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q3.data)
    _check_allclose(orig_np, transpiled_np)
    
    
    # test Quaternion.__init__()

    torch_q = Quaternion(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])))

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.__neg__()

    torch_q = -Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = -TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.__pow__()

    torch_q = Quaternion(torch.tensor([1., .5, 0., 0.])) ** 2
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., .5, 0., 0.]))) ** 2

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.__repr__()

    torch_q = Quaternion.identity()
    transpiled_q = TranspiledQuaternion.identity()

    torch_repr = repr(torch_q)
    transpiled_repr = repr(transpiled_q)
    assert torch_repr == transpiled_repr


    # test Quaternion.__sub__()

    torch_q1 = Quaternion(torch.tensor([2., 0., 1., 1.]))
    torch_q2 = Quaternion.identity()
    torch_q3 = torch_q1 - torch_q2
    transpiled_q1 = TranspiledQuaternion(_array_to_new_backend(torch.tensor([2., 0., 1., 1.])))
    transpiled_q2 = TranspiledQuaternion.identity()
    transpiled_q3 = transpiled_q1 - transpiled_q2

    orig_np = _nest_array_to_numpy(torch_q3.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q3.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.coeffs

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    torch_coeffs = _nest_array_to_numpy(torch_q.coeffs)
    transpiled_coeffs = _nest_array_to_numpy(transpiled_q.coeffs)
    _check_allclose(torch_coeffs, transpiled_coeffs)


    # test Quaternion.data

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.from_axis_angle()

    torch_q = Quaternion.from_axis_angle(torch.tensor([[1., 0., 0.]]))
    transpiled_q = TranspiledQuaternion.from_axis_angle(_array_to_new_backend(torch.tensor([[1., 0., 0.]])))

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.from_coeffs()

    torch_q = Quaternion.from_coeffs(1., 0., 0., 0.)
    transpiled_q = TranspiledQuaternion.from_coeffs(1., 0., 0., 0.)

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.from_euler()

    roll, pitch, yaw = torch.tensor(0), torch.tensor(1), torch.tensor(0)
    torch_q = Quaternion.from_euler(roll, pitch, yaw)
    transpiled_q = TranspiledQuaternion.from_euler(
        _array_to_new_backend(roll), _array_to_new_backend(pitch), _array_to_new_backend(yaw)
    )

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.from_matrix()

    torch_m = torch.eye(3)[None]
    transpiled_m = _array_to_new_backend(torch.eye(3)[None])

    torch_q = Quaternion.from_matrix(torch_m)
    transpiled_q = TranspiledQuaternion.from_matrix(transpiled_m)

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.matrix()

    torch_q = Quaternion.identity()
    transpiled_q = TranspiledQuaternion.identity()

    torch_matrix = _nest_array_to_numpy(torch_q.matrix())
    transpiled_matrix = _nest_array_to_numpy(transpiled_q.matrix())
    _check_allclose(torch_matrix, transpiled_matrix)


    # test Quaternion.polar_angle

    torch_q = Quaternion.identity()
    transpiled_q = TranspiledQuaternion.identity()

    torch_polar_angle = _nest_array_to_numpy(torch_q.polar_angle)
    transpiled_polar_angle = _nest_array_to_numpy(transpiled_q.polar_angle)
    _check_allclose(torch_polar_angle, transpiled_polar_angle)


    # test Quaternion.q

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.q)
    transpiled_np = _nest_array_to_numpy(transpiled_q.q)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.random()

    torch_q = Quaternion.random(batch_size=2)
    transpiled_q = TranspiledQuaternion.random(batch_size=2)

    orig_np = _nest_array_to_numpy(torch_q.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q.data)
    _check_shape_allclose(orig_np, transpiled_np)


    # test Quaternion.real

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.real)
    transpiled_np = _nest_array_to_numpy(transpiled_q.real)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.scalar

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.scalar)
    transpiled_np = _nest_array_to_numpy(transpiled_q.scalar)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.shape

    torch_q = Quaternion(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])))

    torch_shape = torch_q.shape
    transpiled_shape = transpiled_q.shape
    assert torch_shape == transpiled_shape


    # test Quaternion.slerp()

    torch_q0 = Quaternion.identity()
    torch_q1 = Quaternion(torch.tensor([1., .5, 0., 0.]))
    torch_q2 = torch_q0.slerp(torch_q1, .3)

    transpiled_q0 = TranspiledQuaternion.identity()
    transpiled_q1 = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., .5, 0., 0.])))
    transpiled_q2 = transpiled_q0.slerp(transpiled_q1, .3)

    orig_np = _nest_array_to_numpy(torch_q2.data)
    transpiled_np = _nest_array_to_numpy(transpiled_q2.data)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.to_euler()

    torch_q = Quaternion(torch.tensor([2., 0., 1., 1.]))
    roll, pitch, yaw = torch_q.to_euler()

    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([2., 0., 1., 1.])))
    transpiled_roll, transpiled_pitch, transpiled_yaw = transpiled_q.to_euler()

    _check_allclose(_nest_array_to_numpy(roll), _nest_array_to_numpy(transpiled_roll))
    _check_allclose(_nest_array_to_numpy(pitch), _nest_array_to_numpy(transpiled_pitch))
    _check_allclose(_nest_array_to_numpy(yaw), _nest_array_to_numpy(transpiled_yaw))


    # test Quaternion.vec

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.vec)
    transpiled_np = _nest_array_to_numpy(transpiled_q.vec)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.w

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.w)
    transpiled_np = _nest_array_to_numpy(transpiled_q.w)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.x

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.x)
    transpiled_np = _nest_array_to_numpy(transpiled_q.x)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.y

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.y)
    transpiled_np = _nest_array_to_numpy(transpiled_q.y)
    _check_allclose(orig_np, transpiled_np)


    # test Quaternion.z

    torch_q = Quaternion(torch.tensor([1., 0., 0., 0.]))
    transpiled_q = TranspiledQuaternion(_array_to_new_backend(torch.tensor([1., 0., 0., 0.])))

    orig_np = _nest_array_to_numpy(torch_q.z)
    transpiled_np = _nest_array_to_numpy(transpiled_q.z)
    _check_allclose(orig_np, transpiled_np)

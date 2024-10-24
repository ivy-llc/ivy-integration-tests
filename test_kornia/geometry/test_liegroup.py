from helpers import (
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_So3(target_framework, mode, backend_compile):
    print("kornia.geometry.liegroup.So3")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    # Initialize a Quaternion and create an So3 object
    quaternion_data = torch.tensor([1., 0., 0., 0.])
    torch_quaternion = kornia.geometry.quaternion.Quaternion(quaternion_data)
    torch_so3 = kornia.geometry.liegroup.So3(torch_quaternion)

    # Transpile the So3 class
    transpiled_quaternion = transpiled_kornia.geometry.quaternion.Quaternion(_nest_torch_tensor_to_new_framework(quaternion_data, target_framework))
    transpiled_so3 = transpiled_kornia.geometry.liegroup.So3(transpiled_quaternion)

    # Test .matrix()
    torch_matrix = torch_so3.matrix()
    transpiled_matrix = transpiled_so3.matrix()
    _to_numpy_and_allclose(torch_matrix, transpiled_matrix)

    # Test .inverse()
    torch_inverse = torch_so3.inverse()
    transpiled_inverse = transpiled_so3.inverse()
    _to_numpy_and_allclose(torch_inverse.q.data, transpiled_inverse.q.data)

    # Test .log()
    torch_log = torch_so3.log()
    transpiled_log = transpiled_so3.log()
    _to_numpy_and_allclose(torch_log, transpiled_log)

    # Test .__mul__()
    other_quaternion_data = torch.tensor([0., 1., 0., 0.])
    other_torch_quaternion = kornia.geometry.quaternion.Quaternion(other_quaternion_data)
    other_torch_so3 = kornia.geometry.liegroup.So3(other_torch_quaternion)

    transpiled_other_quaternion = _nest_torch_tensor_to_new_framework(other_quaternion_data, target_framework)
    transpiled_other_quaternion_obj = transpiled_kornia.geometry.quaternion.Quaternion(transpiled_other_quaternion)
    transpiled_other_so3 = transpiled_kornia.geometry.liegroup.So3(transpiled_other_quaternion_obj)

    torch_composed_so3 = torch_so3 * other_torch_so3
    transpiled_composed_so3 = transpiled_so3 * transpiled_other_so3
    _to_numpy_and_allclose(torch_composed_so3.q.data, transpiled_composed_so3.q.data)

    # Test .adjoint()
    torch_adjoint = torch_so3.adjoint()
    transpiled_adjoint = transpiled_so3.adjoint()
    _to_numpy_and_allclose(torch_adjoint, transpiled_adjoint)

    # Test .from_matrix()
    rotation_matrix = torch.eye(3)
    transpiled_rotation_matrix = _nest_torch_tensor_to_new_framework(rotation_matrix, target_framework)

    torch_from_matrix = kornia.geometry.liegroup.So3.from_matrix(rotation_matrix)
    transpiled_from_matrix = transpiled_kornia.geometry.liegroup.So3.from_matrix(transpiled_rotation_matrix)
    _to_numpy_and_allclose(torch_from_matrix.q.data, transpiled_from_matrix.q.data)

    # Test .exp()
    exp_vector = torch.tensor([0., 0., 0.])
    transpiled_exp_vector = _nest_torch_tensor_to_new_framework(exp_vector, target_framework)

    torch_exp = kornia.geometry.liegroup.So3.exp(exp_vector)
    transpiled_exp = transpiled_kornia.geometry.liegroup.So3.exp(transpiled_exp_vector)
    _to_numpy_and_allclose(torch_exp.q.data, transpiled_exp.q.data)


def test_Se3(target_framework, mode, backend_compile):
    print("kornia.geometry.liegroup.Se3")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    quaternion_data = torch.tensor([1., 0., 0., 0.])
    translation_data = torch.tensor([1., 1., 1.])
    torch_quaternion = kornia.geometry.quaternion.Quaternion(quaternion_data)
    torch_se3 = kornia.geometry.liegroup.Se3(torch_quaternion, translation_data)

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    transpiled_translation = _nest_torch_tensor_to_new_framework(translation_data, target_framework)
    transpiled_quaternion = transpiled_kornia.geometry.quaternion.Quaternion(_nest_torch_tensor_to_new_framework(quaternion_data, target_framework))
    transpiled_se3 = transpiled_kornia.geometry.liegroup.Se3(transpiled_quaternion, transpiled_translation)

    # Test .matrix()
    torch_matrix = torch_se3.matrix()
    transpiled_matrix = transpiled_se3.matrix()
    _to_numpy_and_allclose(torch_matrix, transpiled_matrix)

    # Test .inverse()
    torch_inverse = torch_se3.inverse()
    transpiled_inverse = transpiled_se3.inverse()
    _to_numpy_and_allclose(torch_inverse.r.q.data, transpiled_inverse.r.q.data)
    _to_numpy_and_allclose(torch_inverse.t, transpiled_inverse.t)

    # Test .log()
    torch_log = torch_se3.log()
    transpiled_log = transpiled_se3.log()
    _to_numpy_and_allclose(torch_log, transpiled_log)

    # Test .__mul__()
    other_quaternion_data = torch.tensor([0., 1., 0., 0.])
    other_translation_data = torch.tensor([2., 2., 2.])
    other_torch_quaternion = kornia.geometry.quaternion.Quaternion(other_quaternion_data)
    other_torch_se3 = kornia.geometry.liegroup.Se3(other_torch_quaternion, other_translation_data)

    transpiled_other_quaternion = _nest_torch_tensor_to_new_framework(other_quaternion_data, target_framework)
    transpiled_other_translation = _nest_torch_tensor_to_new_framework(other_translation_data, target_framework)
    transpiled_other_quaternion_obj = transpiled_kornia.geometry.quaternion.Quaternion(transpiled_other_quaternion)
    transpiled_other_se3 = transpiled_kornia.geometry.liegroup.Se3(transpiled_other_quaternion_obj, transpiled_other_translation)

    torch_composed_se3 = torch_se3 * other_torch_se3
    transpiled_composed_se3 = transpiled_se3 * transpiled_other_se3
    _to_numpy_and_allclose(torch_composed_se3.r.q.data, transpiled_composed_se3.r.q.data)
    _to_numpy_and_allclose(torch_composed_se3.t, transpiled_composed_se3.t)

    # Test .adjoint()
    torch_adjoint = torch_se3.adjoint()
    transpiled_adjoint = transpiled_se3.adjoint()
    _to_numpy_and_allclose(torch_adjoint, transpiled_adjoint)

    # Test .from_matrix()
    rotation_translation_matrix = torch.eye(4)
    transpiled_rotation_translation_matrix = _nest_torch_tensor_to_new_framework(rotation_translation_matrix, target_framework)

    torch_from_matrix = kornia.geometry.liegroup.Se3.from_matrix(rotation_translation_matrix)
    transpiled_from_matrix = transpiled_kornia.geometry.liegroup.Se3.from_matrix(transpiled_rotation_translation_matrix)
    _to_numpy_and_allclose(torch_from_matrix.r.q.data, transpiled_from_matrix.r.q.data)
    _to_numpy_and_allclose(torch_from_matrix.t, transpiled_from_matrix.t)

    # Test .exp()
    exp_vector = torch.tensor([0., 0., 0., 0., 0., 0.])
    transpiled_exp_vector = _nest_torch_tensor_to_new_framework(exp_vector, target_framework)

    torch_exp = kornia.geometry.liegroup.Se3.exp(exp_vector)
    transpiled_exp = transpiled_kornia.geometry.liegroup.Se3.exp(transpiled_exp_vector)
    _to_numpy_and_allclose(torch_exp.r.q.data, transpiled_exp.r.q.data)
    _to_numpy_and_allclose(torch_exp.t, transpiled_exp.t)


def test_So2(target_framework, mode, backend_compile):
    print("kornia.geometry.liegroup.So2")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    real_part = torch.tensor([1.0], requires_grad=True)
    imaginary_part = torch.tensor([2.0], requires_grad=True)
    complex_number = torch.complex(real_part, imaginary_part)
    torch_so2 = kornia.geometry.liegroup.So2(complex_number)

    transpiled_complex_number = _nest_torch_tensor_to_new_framework(complex_number, target_framework)
    transpiled_so2 = transpiled_kornia.geometry.liegroup.So2(transpiled_complex_number)

    # Test .matrix()
    torch_matrix = torch_so2.matrix()
    transpiled_matrix = transpiled_so2.matrix()
    _to_numpy_and_allclose(torch_matrix, transpiled_matrix)

    # Test .inverse()
    torch_inverse = torch_so2.inverse()
    transpiled_inverse = transpiled_so2.inverse()
    _to_numpy_and_allclose(torch_inverse.z, transpiled_inverse.z)

    # Test .log()
    torch_log = torch_so2.log()
    transpiled_log = transpiled_so2.log()
    _to_numpy_and_allclose(torch_log, transpiled_log)

    # Test .__mul__()
    other_real_part = torch.tensor([0.5], requires_grad=True)
    other_imaginary_part = torch.tensor([0.5], requires_grad=True)
    other_complex_number = torch.complex(other_real_part, other_imaginary_part)
    other_torch_so2 = kornia.geometry.liegroup.So2(other_complex_number)

    transpiled_other_complex_number = _nest_torch_tensor_to_new_framework(other_complex_number, target_framework)
    transpiled_other_so2 = transpiled_kornia.geometry.liegroup.So2(transpiled_other_complex_number)

    torch_composed_so2 = torch_so2 * other_torch_so2
    transpiled_composed_so2 = transpiled_so2 * transpiled_other_so2
    _to_numpy_and_allclose(torch_composed_so2.z, transpiled_composed_so2.z)

    # Test .adjoint()
    torch_adjoint = torch_so2.adjoint()
    transpiled_adjoint = transpiled_so2.adjoint()
    _to_numpy_and_allclose(torch_adjoint, transpiled_adjoint)

    # Test .from_matrix()
    rotation_matrix = torch.eye(2)
    transpiled_rotation_matrix = _nest_torch_tensor_to_new_framework(rotation_matrix, target_framework)

    torch_from_matrix = kornia.geometry.liegroup.So2.from_matrix(rotation_matrix)
    transpiled_from_matrix = transpiled_kornia.geometry.liegroup.So2.from_matrix(transpiled_rotation_matrix)
    _to_numpy_and_allclose(torch_from_matrix.z, transpiled_from_matrix.z)

    # Test .exp()
    theta = torch.tensor([3.1415 / 2])
    transpiled_theta = _nest_torch_tensor_to_new_framework(theta, target_framework)

    torch_exp = kornia.geometry.liegroup.So2.exp(theta)
    transpiled_exp = transpiled_kornia.geometry.liegroup.So2.exp(transpiled_theta)
    _to_numpy_and_allclose(torch_exp.z, transpiled_exp.z)

    # Test .identity()
    torch_identity = kornia.geometry.liegroup.So2.identity()
    transpiled_identity = transpiled_kornia.geometry.liegroup.So2.identity()
    _to_numpy_and_allclose(torch_identity.z, transpiled_identity.z)

    # Test .hat() and .vee()
    hat_theta = torch.tensor([3.1415 / 2])
    transpiled_hat_theta = _nest_torch_tensor_to_new_framework(hat_theta, target_framework)

    torch_hat = kornia.geometry.liegroup.So2.hat(hat_theta)
    transpiled_hat = transpiled_kornia.geometry.liegroup.So2.hat(transpiled_hat_theta)
    _to_numpy_and_allclose(torch_hat, transpiled_hat)

    torch_vee = kornia.geometry.liegroup.So2.vee(torch_hat)
    transpiled_vee = transpiled_kornia.geometry.liegroup.So2.vee(transpiled_hat)
    _to_numpy_and_allclose(torch_vee, transpiled_vee)


def test_Se2(target_framework, mode, backend_compile):
    print("kornia.geometry.liegroup.Se2")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target_framework)

    so2_rotation = kornia.geometry.liegroup.So2.identity(1)
    translation_vector = torch.ones((1, 2), requires_grad=True)
    torch_se2 = kornia.geometry.liegroup.Se2(so2_rotation, translation_vector)

    transpiled_so2_rotation = transpiled_kornia.geometry.liegroup.So2.identity(1)
    transpiled_translation_vector = _nest_torch_tensor_to_new_framework(translation_vector, target_framework)
    transpiled_se2 = transpiled_kornia.geometry.liegroup.Se2(transpiled_so2_rotation, transpiled_translation_vector)

    # Test .matrix()
    torch_matrix = torch_se2.matrix()
    transpiled_matrix = transpiled_se2.matrix()
    _to_numpy_and_allclose(torch_matrix, transpiled_matrix)

    # Test .inverse()
    torch_inverse = torch_se2.inverse()
    transpiled_inverse = transpiled_se2.inverse()
    _to_numpy_and_allclose(torch_inverse.rotation.z, transpiled_inverse.rotation.z)
    _to_numpy_and_allclose(torch_inverse.translation, transpiled_inverse.translation)

    # Test .log()
    torch_log = torch_se2.log()
    transpiled_log = transpiled_se2.log()
    _to_numpy_and_allclose(torch_log, transpiled_log)

    # Test .__mul__()
    other_so2_rotation = kornia.geometry.liegroup.So2.identity(1)
    other_translation_vector = torch.tensor([[0.5, 0.5]], requires_grad=True)
    other_torch_se2 = kornia.geometry.liegroup.Se2(other_so2_rotation, other_translation_vector)

    transpiled_other_so2_rotation = transpiled_kornia.geometry.liegroup.So2.identity(1)
    transpiled_other_translation_vector = _nest_torch_tensor_to_new_framework(other_translation_vector, target_framework)
    transpiled_other_se2 = transpiled_kornia.geometry.liegroup.Se2(transpiled_other_so2_rotation, transpiled_other_translation_vector)

    torch_composed_se2 = torch_se2 * other_torch_se2
    transpiled_composed_se2 = transpiled_se2 * transpiled_other_se2
    _to_numpy_and_allclose(torch_composed_se2.rotation.z, transpiled_composed_se2.rotation.z)
    _to_numpy_and_allclose(torch_composed_se2.translation, transpiled_composed_se2.translation)

    # Test .adjoint()
    torch_adjoint = torch_se2.adjoint()
    transpiled_adjoint = transpiled_se2.adjoint()
    _to_numpy_and_allclose(torch_adjoint, transpiled_adjoint)

    # Test .from_matrix()
    rotation_matrix = torch.eye(3).repeat(2, 1, 1)
    transpiled_rotation_matrix = _nest_torch_tensor_to_new_framework(rotation_matrix, target_framework)

    torch_from_matrix = kornia.geometry.liegroup.Se2.from_matrix(rotation_matrix)
    transpiled_from_matrix = transpiled_kornia.geometry.liegroup.Se2.from_matrix(transpiled_rotation_matrix)
    _to_numpy_and_allclose(torch_from_matrix.rotation.z, transpiled_from_matrix.rotation.z)
    _to_numpy_and_allclose(torch_from_matrix.translation, transpiled_from_matrix.translation)

    # Test .exp()
    v = torch.ones((1, 3))
    transpiled_v = _nest_torch_tensor_to_new_framework(v, target_framework)

    torch_exp = kornia.geometry.liegroup.Se2.exp(v)
    transpiled_exp = transpiled_kornia.geometry.liegroup.Se2.exp(transpiled_v)
    _to_numpy_and_allclose(torch_exp.rotation.z, transpiled_exp.rotation.z)
    _to_numpy_and_allclose(torch_exp.translation, transpiled_exp.translation)

    # Test .identity()
    torch_identity = kornia.geometry.liegroup.Se2.identity(1)
    transpiled_identity = transpiled_kornia.geometry.liegroup.Se2.identity(1)
    _to_numpy_and_allclose(torch_identity.rotation.z, transpiled_identity.rotation.z)
    _to_numpy_and_allclose(torch_identity.translation.data, transpiled_identity.translation.data)

    # Test .hat() and .vee()
    hat_v = torch.ones((1, 3))
    transpiled_hat_v = _nest_torch_tensor_to_new_framework(hat_v, target_framework)

    torch_hat = kornia.geometry.liegroup.Se2.hat(hat_v)
    transpiled_hat = transpiled_kornia.geometry.liegroup.Se2.hat(transpiled_hat_v)
    _to_numpy_and_allclose(torch_hat, transpiled_hat)

    torch_vee = kornia.geometry.liegroup.Se2.vee(torch_hat)
    transpiled_vee = transpiled_kornia.geometry.liegroup.Se2.vee(transpiled_hat)
    _to_numpy_and_allclose(torch_vee, transpiled_vee)

    # Test properties .rotation and .translation
    _to_numpy_and_allclose(torch_se2.rotation.z, transpiled_se2.rotation.z)
    _to_numpy_and_allclose(torch_se2.translation, transpiled_se2.translation)

    # Test class methods .trans(), .trans_x(), and .trans_y()
    x_translation = torch.tensor([1.0])
    y_translation = torch.tensor([2.0])

    transpiled_x_translation = _nest_torch_tensor_to_new_framework(x_translation, target_framework)
    transpiled_y_translation = _nest_torch_tensor_to_new_framework(y_translation, target_framework)

    torch_trans_se2 = kornia.geometry.liegroup.Se2.trans(x_translation, y_translation)
    transpiled_trans_se2 = transpiled_kornia.geometry.liegroup.Se2.trans(transpiled_x_translation, transpiled_y_translation)
    _to_numpy_and_allclose(torch_trans_se2.translation, transpiled_trans_se2.translation)

    torch_trans_x_se2 = kornia.geometry.liegroup.Se2.trans_x(x_translation)
    transpiled_trans_x_se2 = transpiled_kornia.geometry.liegroup.Se2.trans_x(transpiled_x_translation)
    _to_numpy_and_allclose(torch_trans_x_se2.translation[:, 0], transpiled_trans_x_se2.translation[:, 0])

    torch_trans_y_se2 = kornia.geometry.liegroup.Se2.trans_y(y_translation)
    transpiled_trans_y_se2 = transpiled_kornia.geometry.liegroup.Se2.trans_y(transpiled_y_translation)
    _to_numpy_and_allclose(torch_trans_y_se2.translation[:, 1], transpiled_trans_y_se2.translation[:, 1])

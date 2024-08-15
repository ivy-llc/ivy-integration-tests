from helpers import (
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
import torch


# Tests #
# ----- #

def test_AutoAugment(target_framework, mode, backend_compile):
    print("kornia.augmentation.auto.AutoAugment")
    
    TranspiledAutoAugment = ivy.transpile(
        kornia.augmentation.auto.AutoAugment,
        source="torch",
        target=target_framework,
    )

    torch_aug = kornia.augmentation.auto.AutoAugment()
    transpiled_aug = TranspiledAutoAugment()
    
    torch_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_out = torch_aug(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)

    torch_inverse_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_inverse_out = torch_aug.inverse(*torch_args)

    transpiled_inverse_args = _nest_torch_tensor_to_new_framework(torch_inverse_args, target_framework)
    transpiled_inverse_out = transpiled_aug.inverse(*transpiled_inverse_args)

    orig_inverse_np = _nest_array_to_numpy(torch_inverse_out)
    transpiled_inverse_np = _nest_array_to_numpy(transpiled_inverse_out)
    _check_shape_allclose(orig_inverse_np, transpiled_inverse_np)


def test_RandAugment(target_framework, mode, backend_compile):
    print("kornia.augmentation.auto.RandAugment")

    TranspiledRandAugment = ivy.transpile(
        kornia.augmentation.auto.RandAugment,
        source="torch",
        target=target_framework,
    )

    torch_aug = kornia.augmentation.auto.RandAugment(n=2, m=10)
    transpiled_aug = TranspiledRandAugment(n=2, m=10)
    
    torch_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_out = torch_aug(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)

    torch_inverse_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_inverse_out = torch_aug.inverse(*torch_args)

    transpiled_inverse_args = _nest_torch_tensor_to_new_framework(torch_inverse_args, target_framework)
    transpiled_inverse_out = transpiled_aug.inverse(*transpiled_inverse_args)

    orig_inverse_np = _nest_array_to_numpy(torch_inverse_out)
    transpiled_inverse_np = _nest_array_to_numpy(transpiled_inverse_out)
    _check_shape_allclose(orig_inverse_np, transpiled_inverse_np)


def test_TrivialAugment(target_framework, mode, backend_compile):
    print("kornia.augmentation.auto.TrivialAugment")

    TranspiledTrivialAugment = ivy.transpile(
        kornia.augmentation.auto.TrivialAugment,
        source="torch",
        target=target_framework,
    )

    torch_aug = kornia.augmentation.auto.TrivialAugment()
    transpiled_aug = TranspiledTrivialAugment()
    
    torch_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_out = torch_aug(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)

    torch_inverse_args = (
        torch.rand(5, 3, 30, 30),
    )
    torch_inverse_out = torch_aug.inverse(*torch_args)

    transpiled_inverse_args = _nest_torch_tensor_to_new_framework(torch_inverse_args, target_framework)
    transpiled_inverse_out = transpiled_aug.inverse(*transpiled_inverse_args)

    orig_inverse_np = _nest_array_to_numpy(torch_inverse_out)
    transpiled_inverse_np = _nest_array_to_numpy(transpiled_inverse_out)
    _check_shape_allclose(orig_inverse_np, transpiled_inverse_np)

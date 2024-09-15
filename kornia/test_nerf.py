from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
from kornia.nerf import nerf_model, nerf_solver, volume_renderer, samplers
import pytest
import torch


# Tests #
# ----- #

def test_NerfModel(target_framework, mode, backend_compile):
    print("kornia.nerf.nerf_model.NerfModel")

    if backend_compile:
        pytest.skip()

    TranspiledNerfModel = ivy.transpile(nerf_model.NerfModel, source="torch", target=target_framework)

    torch_args = (
        torch.rand(5, 3),
        torch.rand(5, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_nerf = nerf_model.NerfModel(num_ray_points=32)
    transpiled_nerf = TranspiledNerfModel(num_ray_points=32)

    torch_out = torch_nerf(*torch_args)
    transpiled_out = transpiled_nerf(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_NerfModelRenderer(target_framework, mode, backend_compile):
    print("kornia.nerf.nerf_model.NerfModelRenderer")

    if backend_compile:
        pytest.skip()

    TranspiledPinholeCamera = ivy.transpile(kornia.geometry.camera.pinhole.PinholeCamera, source="torch", target=target_framework)
    TranspiledNerfModel = ivy.transpile(nerf_model.NerfModel, source="torch", target=target_framework)
    TranspiledNerfModelRenderer = ivy.transpile(nerf_model.NerfModelRenderer, source="torch", target=target_framework)

    torch_nerf_model = nerf_model.NerfModel(num_ray_points=32)
    transpiled_nerf_model = TranspiledNerfModel(num_ray_points=32)

    torch_camera_args = (
        torch.rand(1, 4, 4),
        torch.rand(1, 4, 4),
        torch.tensor([256]),
        torch.tensor([256]),
    )
    transpiled_camera_args = _nest_torch_tensor_to_new_framework(torch_camera_args)

    torch_camera = kornia.geometry.camera.pinhole.PinholeCamera(*torch_camera_args)
    transpiled_camera = TranspiledPinholeCamera(*transpiled_camera_args)

    torch_renderer = nerf_model.NerfModelRenderer(torch_nerf_model, image_size=(128, 128), device="cpu", dtype=torch.float32)
    transpiled_renderer = TranspiledNerfModelRenderer(transpiled_nerf_model, image_size=(128, 128), device="cpu", dtype=torch.float32)

    torch_out = torch_renderer.render_view(torch_camera)
    transpiled_out = transpiled_renderer.render_view(transpiled_camera)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_MLP(target_framework, mode, backend_compile):
    print("kornia.nerf.nerf_model.MLP")

    if backend_compile:
        pytest.skip()

    TranspiledMLP = ivy.transpile(nerf_model.MLP, source="torch", target=target_framework)

    torch_args = (
        torch.rand(5, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_mlp = nerf_model.MLP(num_dims=3)
    transpiled_mlp = TranspiledMLP(num_dims=3)

    torch_out = torch_mlp(*torch_args)
    transpiled_out = transpiled_mlp(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


# skipping due to the presence of torch.optim.Adam 
# def test_NerfSolver(target_framework, mode, backend_compile):
#     print("kornia.nerf.nerf_solver.NerfSolver")

#     if backend_compile:
#         pytest.skip()
    
#     TranspiledPinholeCamera = ivy.transpile(kornia.geometry.camera.pinhole.PinholeCamera, source="torch", target=target_framework)
#     TranspiledNerfSolver = ivy.transpile(nerf_solver.NerfSolver, source="torch", target=target_framework)

#     torch_camera_args = (
#         torch.rand(1, 4, 4),
#         torch.rand(1, 4, 4),
#         torch.tensor([256]),
#         torch.tensor([256]),
#     )
#     transpiled_camera_args = _nest_torch_tensor_to_new_framework(torch_camera_args)
#     torch_camera = kornia.geometry.camera.pinhole.PinholeCamera(*torch_camera_args)
#     transpiled_camera = TranspiledPinholeCamera(*transpiled_camera_args)

#     imgs = [torch.rand(3, 256, 256) for _ in range(1)]

#     torch_solver = nerf_solver.NerfSolver(device="cpu", dtype=torch.float32)
#     torch_solver.setup_solver(torch_camera, 0.1, 10.0, True, imgs, 100, 16, 32)

#     transpiled_solver = TranspiledNerfSolver(device="cpu", dtype=torch.float32)
#     transpiled_solver.setup_solver(transpiled_camera, 0.1, 10.0, True, imgs, 100, 16, 32)

#     torch_solver.run(1)
#     transpiled_solver.run(1)


def test_IrregularRenderer(target_framework, mode, backend_compile):
    print("kornia.nerf.volume_renderer.IrregularRenderer")

    if backend_compile:
        pytest.skip()

    TranspiledIrregularRenderer = ivy.transpile(volume_renderer.IrregularRenderer, source="torch", target=target_framework)

    torch_args = (
        torch.rand(5, 32, 3),
        torch.rand(5, 32, 1),
        torch.rand(5, 32, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_renderer = volume_renderer.IrregularRenderer(shift=1)
    transpiled_renderer = TranspiledIrregularRenderer(shift=1)

    torch_out = torch_renderer(*torch_args)
    transpiled_out = transpiled_renderer(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_RegularRenderer(target_framework, mode, backend_compile):
    print("kornia.nerf.volume_renderer.RegularRenderer")

    if backend_compile:
        pytest.skip()

    TranspiledRegularRenderer = ivy.transpile(volume_renderer.RegularRenderer, source="torch", target=target_framework)

    torch_args = (
        torch.rand(5, 32, 3),
        torch.rand(5, 32, 1),
        torch.rand(5, 32, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_renderer = volume_renderer.RegularRenderer(shift=1)
    transpiled_renderer = TranspiledRegularRenderer(shift=1)

    torch_out = torch_renderer(*torch_args)
    transpiled_out = transpiled_renderer(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


# TODO: error in standard kornia call?
def test_RaySampler(target_framework, mode, backend_compile):
    print("kornia.nerf.samplers.RaySampler")

    if backend_compile:
        pytest.skip()

    TranspiledPinholeCamera = ivy.transpile(kornia.geometry.camera.pinhole.PinholeCamera, source="torch", target=target_framework)
    TranspiledRaySampler = ivy.transpile(samplers.RaySampler, source="torch", target=target_framework)

    torch_camera_args = (
        torch.rand(1, 4, 4),
        torch.rand(1, 4, 4),
        torch.tensor([256]),
        torch.tensor([256]),
    )
    transpiled_camera_args = _nest_torch_tensor_to_new_framework(torch_camera_args, target_framework)

    torch_camera = kornia.geometry.camera.pinhole.PinholeCamera(*torch_camera_args)
    transpiled_camera = TranspiledPinholeCamera(*transpiled_camera_args)

    torch_sampler = samplers.RaySampler(min_depth=0.1, max_depth=10.0, ndc=True, device="cpu", dtype=torch.float32)
    transpiled_sampler = TranspiledRaySampler(min_depth=0.1, max_depth=10.0, ndc=True, device="cpu", dtype=torch.float32)

    torch_out = torch_sampler.transform_ray_params_world_to_ndc(torch_camera)
    transpiled_out = transpiled_sampler.transform_ray_params_world_to_ndc(transpiled_camera)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_RandomRaySampler(target_framework, mode, backend_compile):
    print("kornia.nerf.samplers.RandomRaySampler")

    if backend_compile:
        pytest.skip()

    TranspiledPinholeCamera = ivy.transpile(kornia.geometry.camera.pinhole.PinholeCamera, source="torch", target=target_framework)
    TranspiledRandomRaySampler = ivy.transpile(samplers.RandomRaySampler, source="torch", target=target_framework)

    torch_camera_args = (
        torch.rand(1, 4, 4),
        torch.rand(1, 4, 4),
        torch.tensor([256]),
        torch.tensor([256]),
    )
    transpiled_camera_args = _nest_torch_tensor_to_new_framework(torch_camera_args, target_framework)

    torch_camera = kornia.geometry.camera.pinhole.PinholeCamera(*torch_camera_args)
    transpiled_camera = TranspiledPinholeCamera(*transpiled_camera_args)

    heights, widths = torch.tensor([256]), torch.tensor([256])
    transpiled_heights = _array_to_new_backend(heights, target_framework)
    transpiled_widths = _array_to_new_backend(widths, target_framework)

    torch_sampler = samplers.RandomRaySampler(min_depth=0.1, max_depth=10.0, ndc=True, device="cpu", dtype=torch.float32)
    transpiled_sampler = TranspiledRandomRaySampler(min_depth=0.1, max_depth=10.0, ndc=True, device="cpu", dtype=torch.float32)

    torch_sampler.calc_ray_params(torch_camera, torch.tensor([1]))
    transpiled_sampler.calc_ray_params(transpiled_camera, _array_to_new_backend(torch.tensor([1]), target_framework))
    torch_sampler.sample_points_2d(heights, widths, torch.tensor([3]))
    transpiled_sampler.sample_points_2d(transpiled_heights, transpiled_widths, _array_to_new_backend(torch.tensor([3]), target_framework))

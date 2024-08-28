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

def test_warp_perspective(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.eye(3).unsqueeze(0),
        (4, 4),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.eye(3).unsqueeze(0),
        (4, 4),  # TODO: changing this fails the test
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_perspective,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_perspective3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5, 5),
        torch.eye(4).unsqueeze(0),
        (4, 4, 4),
    )
    trace_kwargs = {'flags': 'bilinear', 'border_mode': 'zeros', 'align_corners': False}
    test_args = (
        torch.rand(1, 3, 6, 7, 6),
        torch.eye(4).unsqueeze(0),
        (2, 2, 2),
    )
    test_kwargs = {'flags': 'bilinear', 'border_mode': 'zeros', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.warp_perspective3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.eye(2, 3).unsqueeze(0),
        (4, 4),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(1, 4, 5, 6),
        torch.eye(2, 3).unsqueeze(0),
        (4, 4),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_affine3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5, 5),
        torch.eye(3, 4).unsqueeze(0),
        (4, 4, 4),
    )
    trace_kwargs = {'flags': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(1, 4, 5, 6, 5),
        torch.eye(3, 4).unsqueeze(0),
        (4, 4, 4),
    )
    test_kwargs = {'flags': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_affine3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_image_tps(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 3, 2),
    )
    trace_kwargs = {'align_corners': False}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 5, 2),
        torch.rand(5, 5, 2),
        torch.rand(5, 3, 2),
    )
    test_kwargs = {'align_corners': False}
    _test_function(
        kornia.geometry.transform.warp_image_tps,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_points_tps(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 3, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 2),
        torch.rand(5, 10, 2),
        torch.rand(5, 10, 2),
        torch.rand(5, 3, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.warp_points_tps,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_grid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 5, 2),
        torch.eye(3).unsqueeze(0),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 10, 10, 2),
        torch.eye(3).unsqueeze(0),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.warp_grid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_warp_grid3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 5, 5, 3),
        torch.eye(4).unsqueeze(0),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 10, 10, 10, 3),
        torch.eye(4).unsqueeze(0),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.warp_grid3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_remap(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 5, 5),
        torch.rand(1, 5, 5),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': None, 'normalized_coordinates': False}
    test_args = (
        torch.rand(1, 3, 10, 10),
        torch.rand(1, 10, 10),
        torch.rand(1, 10, 10),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': None, 'normalized_coordinates': False}
    _test_function(
        kornia.geometry.transform.remap,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2, 3, 5),
        torch.eye(2, 3).unsqueeze(0),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 2, 3, 5),
        torch.eye(2, 3).unsqueeze(0).repeat(5, 1, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rotate(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 4, 4),
        torch.tensor([90.]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([45., 45., 45., 45., 45.]),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.rotate,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_translate(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 4, 4),
        torch.tensor([[1., 0.]]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([[1., 0.]]).repeat(5, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.translate,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_scale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 4, 4),
        torch.rand(2),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.rand(5, 2),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.scale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_shear(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 4, 4),
        torch.tensor([[0.5, 0.0]]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([[0.1, 0.3]]).repeat(5, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.shear,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_hflip(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.hflip,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_vflip(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.vflip,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rot180(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.rot180,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_resize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        (6, 8),
    )
    trace_kwargs = {'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 4, 4),
        (12, 16),
    )
    test_kwargs = {'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.resize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_rescale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        (2, 3),
    )
    trace_kwargs = {'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 4, 4),
        (1.5, 2),
    )
    test_kwargs = {'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.rescale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_elastic_transform2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 2, 5, 5),
    )
    trace_kwargs = {'kernel_size': (63, 63), 'sigma': (32.0, 32.0), 'alpha': (1.0, 1.0)}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 2, 5, 5),
    )
    test_kwargs = {'kernel_size': (31, 31), 'sigma': (16.0, 16.0), 'alpha': (0.5, 0.5)}
    _test_function(
        kornia.geometry.transform.elastic_transform2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=5e-2,
        mode=mode,
    )


# TODO: failing due to dynamic control flow use in the compositional implementation of `ivy.interpolate`
# (in this case with `mode=bilinear`)
def test_pyrdown(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False, 'factor': 2.0}
    test_args = (
        torch.rand(5, 1, 8, 8),
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False, 'factor': 2.0}
    _test_function(
        kornia.geometry.transform.pyrdown,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_pyrup(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 1, 4, 4),
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.pyrup,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_build_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 8, 8),
        3,
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 16, 16),
        4,
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.build_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_build_laplacian_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 8, 8),
        3,
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 16, 16),
        4,
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.build_laplacian_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_upscale_double(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 8, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.upscale_double,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_perspective_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]),
        torch.tensor([[[1., 0.], [0., 0.], [0., 1.], [1., 1.]]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[[0., 0.], [2., 0.], [2., 2.], [0., 2.]]]),
        torch.tensor([[[2., 0.], [0., 0.], [0., 2.], [2., 2.]]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_perspective_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_perspective_transform3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]]),
        torch.tensor([[[1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [1., 1., 0.]]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]]]),
        torch.tensor([[[2., 0., 0.], [0., 0., 0.], [0., 2., 0.], [2., 2., 0.]]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_perspective_transform3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_projective_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[30., 45., 60.]]),
        torch.tensor([[1., 1., 1.]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([[45., 60., 75.]]),
        torch.tensor([[1.5, 1.5, 1.5]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_projective_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_rotation_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2),
        45. * torch.ones(1),
        torch.rand(1, 2)
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 2),
        90. * torch.ones(1),
        2.0 * torch.ones(1, 2)
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_rotation_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_shear_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_shear_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_shear_matrix3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([0.3]),
        torch.tensor([0.4]),
        torch.tensor([0.6])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75]),
        torch.tensor([0.4]),
        torch.tensor([0.5]),
        torch.tensor([0.6]),
        torch.tensor([0.8])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_shear_matrix3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_affine_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0.]]),
        torch.tensor([[0., 0.]]),
        torch.ones(1, 2),
        45. * torch.ones(1),
        torch.tensor([1.0]),
        torch.tensor([0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1.]]),
        torch.tensor([[1., 1.]]),
        2.0 * torch.ones(1, 2),
        90. * torch.ones(1),
        torch.tensor([1.5]),
        torch.tensor([0.75])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_affine_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_affine_matrix3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[0., 0., 0.]]),
        torch.ones(1, 3),
        torch.tensor([[45., 45., 45.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([0.3]),
        torch.tensor([0.4]),
        torch.tensor([0.6])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([[1., 1., 1.]]),
        2.0 * torch.ones(1, 3),
        torch.tensor([[90., 90., 90.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75]),
        torch.tensor([0.4]),
        torch.tensor([0.5]),
        torch.tensor([0.6]),
        torch.tensor([0.8])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_affine_matrix3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_invert_affine_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.invert_affine_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_projection_from_Rt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
        torch.rand(1, 3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3),
        torch.rand(5, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.projection_from_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_tps_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 5, 2),
        torch.rand(5, 5, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_tps_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=3e-1,
        mode=mode,
    )


def test_crop_by_indices(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
    )
    trace_kwargs = {'size': (40, 40), 'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
    )
    test_kwargs = {'size': (40, 40), 'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.crop_by_indices,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_crop_by_boxes(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
        torch.tensor([[[0, 0], [40, 0], [40, 40], [0, 40]]], dtype=torch.float32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
        torch.tensor([[[0, 0], [40, 0], [40, 40], [0, 40]]]*5, dtype=torch.float32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.crop_by_boxes,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_center_crop(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 64, 64),
        (32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        (32, 32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.center_crop,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_crop_and_resize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
        (32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
        (32, 32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.crop_and_resize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_Rotate(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Rotate")

    if backend_compile:
        pytest.skip()

    TranspiledRotate = ivy.transpile(kornia.geometry.transform.Rotate, source="torch", target=target_framework)

    x = torch.rand(2, 3, 4, 4)
    angle = torch.tensor([45.0, 90.0])
    torch_out = kornia.geometry.transform.Rotate(angle)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_angle = _nest_torch_tensor_to_new_framework(angle, target_framework)
    transpiled_out = TranspiledRotate(transpiled_angle)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Translate(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Translate")

    if backend_compile:
        pytest.skip()

    TranspiledTranslate = ivy.transpile(kornia.geometry.transform.Translate, source="torch", target=target_framework)

    x = torch.rand(2, 3, 4, 4)
    translation = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch_out = kornia.geometry.transform.Translate(translation)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_translation = _nest_torch_tensor_to_new_framework(translation, target_framework)
    transpiled_out = TranspiledTranslate(transpiled_translation)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Scale(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Scale")

    if backend_compile:
        pytest.skip()

    TranspiledScale = ivy.transpile(kornia.geometry.transform.Scale, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    scale_factor = torch.tensor([[2., 2.]])
    torch_out = kornia.geometry.transform.Scale(scale_factor)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_scale_factor = _nest_torch_tensor_to_new_framework(scale_factor, target_framework)
    transpiled_out = TranspiledScale(transpiled_scale_factor)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Shear(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Shear")

    if backend_compile:
        pytest.skip()

    TranspiledShear = ivy.transpile(kornia.geometry.transform.Shear, source="torch", target=target_framework)

    x = torch.rand(2, 3, 4, 4)
    shear = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    torch_out = kornia.geometry.transform.Shear(shear)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_shear = _nest_torch_tensor_to_new_framework(shear, target_framework)
    transpiled_out = TranspiledShear(transpiled_shear)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PyrDown(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.PyrDown")

    if backend_compile:
        pytest.skip()

    TranspiledPyrDown = ivy.transpile(kornia.geometry.transform.PyrDown, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.PyrDown()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledPyrDown()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PyrUp(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.PyrUp")

    if backend_compile:
        pytest.skip()

    TranspiledPyrUp = ivy.transpile(kornia.geometry.transform.PyrUp, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.PyrUp()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledPyrUp()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_ScalePyramid(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.ScalePyramid")

    if backend_compile:
        pytest.skip()

    TranspiledScalePyramid = ivy.transpile(kornia.geometry.transform.ScalePyramid, source="torch", target=target_framework)

    x = torch.rand(2, 4, 100, 100)
    torch_out, _, _ = kornia.geometry.transform.ScalePyramid(n_levels=3, min_size=15)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out, _, _ = TranspiledScalePyramid(n_levels=3, min_size=15)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Hflip(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Hflip")

    if backend_compile:
        pytest.skip()

    TranspiledHflip = ivy.transpile(kornia.geometry.transform.Hflip, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.Hflip()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledHflip()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Vflip(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Vflip")

    if backend_compile:
        pytest.skip()

    TranspiledVflip = ivy.transpile(kornia.geometry.transform.Vflip, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.Vflip()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledVflip()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Rot180(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Rot180")

    if backend_compile:
        pytest.skip()

    TranspiledRot180 = ivy.transpile(kornia.geometry.transform.Rot180, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.Rot180()(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledRot180()(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Resize(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Resize")

    if backend_compile:
        pytest.skip()

    TranspiledResize = ivy.transpile(kornia.geometry.transform.Resize, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.Resize((6, 8))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledResize((6, 8))(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Rescale(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Rescale")

    if backend_compile:
        pytest.skip()

    TranspiledRescale = ivy.transpile(kornia.geometry.transform.Rescale, source="torch", target=target_framework)

    x = torch.rand(1, 3, 4, 4)
    torch_out = kornia.geometry.transform.Rescale((2.0, 3.0))(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_out = TranspiledRescale((2.0, 3.0))(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Affine(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Affine")

    if backend_compile:
        pytest.skip()

    TranspiledAffine = ivy.transpile(kornia.geometry.transform.Affine, source="torch", target=target_framework)

    x = torch.rand(1, 2, 3, 5)
    angle = torch.tensor([45.0])
    torch_out = kornia.geometry.transform.Affine(angle)(x)

    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_angle = _nest_torch_tensor_to_new_framework(angle, target_framework)
    transpiled_out = TranspiledAffine(transpiled_angle)(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_HomographyWarper(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.HomographyWarper")

    if backend_compile:
        pytest.skip()

    TranspiledHomographyWarper = ivy.transpile(
        kornia.geometry.transform.HomographyWarper, source="torch", target=target_framework
    )

    height, width = 32, 32
    homography = torch.eye(3).unsqueeze(0)  # Identity homography
    x = torch.rand(1, 3, height, width)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    transpiled_homography = _nest_torch_tensor_to_new_framework(homography, target_framework)

    warper = kornia.geometry.transform.HomographyWarper(height, width)
    warper.precompute_warp_grid(homography)
    torch_out = warper(x)

    transpiled_warper = TranspiledHomographyWarper(height, width)
    transpiled_warper.precompute_warp_grid(transpiled_homography)
    transpiled_out = transpiled_warper(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Homography(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Homography")

    if backend_compile:
        pytest.skip()

    TranspiledHomography = ivy.transpile(
        kornia.geometry.transform.image_registrator.Homography, source="torch", target=target_framework
    )

    torch_out = kornia.geometry.transform.image_registrator.Homography()()
    transpiled_out = TranspiledHomography()()

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_ImageRegistrator(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.ImageRegistrator")

    if backend_compile:
        pytest.skip()

    TranspiledImageRegistrator = ivy.transpile(
        kornia.geometry.transform.ImageRegistrator, source="torch", target=target_framework
    )

    img_src = torch.rand(1, 1, 32, 32)
    img_dst = torch.rand(1, 1, 32, 32)
    torch_out = kornia.geometry.transform.ImageRegistrator('homography').register(img_src, img_dst)

    transpiled_img_src = _nest_torch_tensor_to_new_framework(img_src, target_framework)
    transpiled_img_dst = _nest_torch_tensor_to_new_framework(img_dst, target_framework)
    transpiled_out = TranspiledImageRegistrator('homography').register(transpiled_img_src, transpiled_img_dst)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_Similarity(target_framework, mode, backend_compile):
    print("kornia.geometry.transform.Similarity")

    if backend_compile:
        pytest.skip()

    TranspiledSimilarity = ivy.transpile(
        kornia.geometry.transform.image_registrator.Similarity, source="torch", target=target_framework
    )

    torch_out = kornia.geometry.transform.image_registrator.Similarity()()
    transpiled_out = TranspiledSimilarity()()

    _to_numpy_and_allclose(torch_out, transpiled_out)

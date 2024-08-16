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

def test_ssim_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 5, 5),
        torch.rand(1, 4, 5, 5),
        5,
    )
    trace_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    test_args = (
        torch.rand(5, 4, 5, 5),
        torch.rand(5, 4, 5, 5),
        7,
    )
    test_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    _test_function(
        kornia.losses.ssim_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_ssim3d_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 5, 5, 5),
        torch.rand(1, 4, 5, 5, 5),
        5,
    )
    trace_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    test_args = (
        torch.rand(5, 4, 5, 5, 5),
        torch.rand(5, 4, 5, 5, 5),
        7,
    )
    test_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    _test_function(
        kornia.losses.ssim3d_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_psnr_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
        1.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.psnr_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_total_variation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'sum'}
    test_args = (
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'sum'}
    _test_function(
        kornia.losses.total_variation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_inverse_depth_smoothness_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 5),
        torch.rand(1, 3, 4, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 5),
        torch.rand(5, 3, 4, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.inverse_depth_smoothness_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_charbonnier_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.charbonnier_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_welsch_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.welsch_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_cauchy_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.cauchy_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_geman_mcclure_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.geman_mcclure_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_binary_focal_loss_with_logits(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 3, 5)),
        torch.randint(2, (1, 3, 5)),
    )
    trace_kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
    test_args = (
        torch.randn((5, 3, 5)),
        torch.randint(2, (5, 3, 5)),
    )
    test_kwargs = {"alpha": 0.5, "gamma": 3.1, "reduction": 'mean'}
    _test_function(
        kornia.losses.binary_focal_loss_with_logits,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_focal_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.randint(5, (1, 3, 5)),
    )
    trace_kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.randint(5, (5, 3, 5)),
    )
    test_kwargs = {"alpha": 0.7, "gamma": 2.5, "reduction": 'mean'}
    _test_function(
        kornia.losses.focal_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_dice_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {"average": "micro", "eps": 1e-8}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {"average": "micro", "eps": 1e-8}
    _test_function(
        kornia.losses.dice_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_tversky_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {"alpha": 0.5, "beta": 0.5, "eps": 1e-8}
    test_args = (
        torch.randn(5, 5, 3, 5),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {"alpha": 0.5, "beta": 0.5, "eps": 1e-8}
    _test_function(
        kornia.losses.tversky_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_lovasz_hinge_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 1, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(1),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((5, 1, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(1),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.lovasz_hinge_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_lovasz_softmax_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.lovasz_softmax_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_js_div_loss_2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand((1, 1, 2, 4)),
        torch.rand((1, 1, 2, 4)),
    )
    trace_kwargs = {"reduction": "mean"}
    test_args = (
        torch.rand((5, 1, 2, 4)),
        torch.rand((5, 1, 2, 4)),
    )
    test_kwargs = {"reduction": "mean"}
    _test_function(
        kornia.losses.js_div_loss_2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_kl_div_loss_2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand((1, 1, 2, 4)),
        torch.rand((1, 1, 2, 4)),
    )
    trace_kwargs = {"reduction": "mean"}
    test_args = (
        torch.rand((5, 1, 2, 4)),
        torch.rand((5, 1, 2, 4)),
    )
    test_kwargs = {"reduction": "mean"}
    _test_function(
        kornia.losses.kl_div_loss_2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_HausdorffERLoss(target_framework, mode, backend_compile):
    print("kornia.losses.HausdorffERLoss")

    if backend_compile:
        pytest.skip()

    TranspiledHausdorffERLoss = ivy.transpiled(kornia.losses.HausdorffERLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.HausdorffERLoss()
    transpiled_loss_fn = TranspiledHausdorffERLoss()

    torch_args = (
        torch.randn(5, 3, 20, 20),
        (torch.rand(5, 1, 20, 20) * 2).long(),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_HausdorffERLoss3D(target_framework, mode, backend_compile):
    print("kornia.losses.HausdorffERLoss3D")

    if backend_compile:
        pytest.skip()

    TranspiledHausdorffERLoss3D = ivy.transpiled(kornia.losses.HausdorffERLoss3D, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.HausdorffERLoss3D()
    transpiled_loss_fn = TranspiledHausdorffERLoss3D()

    torch_args = (
        torch.randn(5, 3, 20, 20, 20),
        (torch.rand(5, 1, 20, 20, 20) * 2).long(),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_SSIMLoss(target_framework, mode, backend_compile):
    print("kornia.losses.SSIMLoss")

    if backend_compile:
        pytest.skip()

    TranspiledSSIMLoss = ivy.transpiled(kornia.losses.SSIMLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.SSIMLoss(5)
    transpiled_loss_fn = TranspiledSSIMLoss(5)

    torch_args = (
        torch.rand(1, 4, 5, 5),
        torch.rand(1, 4, 5, 5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_SSIM3DLoss(target_framework, mode, backend_compile):
    print("kornia.losses.SSIM3DLoss")

    if backend_compile:
        pytest.skip()

    TranspiledSSIM3DLoss = ivy.transpiled(kornia.losses.SSIM3DLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.SSIM3DLoss(5)
    transpiled_loss_fn = TranspiledSSIM3DLoss(5)

    torch_args = (
        torch.rand(1, 4, 5, 5, 5),
        torch.rand(1, 4, 5, 5, 5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_MS_SSIMLoss(target_framework, mode, backend_compile):
    print("kornia.losses.MS_SSIMLoss")

    if backend_compile:
        pytest.skip()

    TranspiledMS_SSIMLoss = ivy.transpiled(kornia.losses.MS_SSIMLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.MS_SSIMLoss()
    transpiled_loss_fn = TranspiledMS_SSIMLoss()

    torch_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 3, 5, 5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_TotalVariation(target_framework, mode, backend_compile):
    print("kornia.losses.TotalVariation")

    if backend_compile:
        pytest.skip()

    TranspiledTotalVariation = ivy.transpiled(kornia.losses.TotalVariation, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.TotalVariation()
    transpiled_loss_fn = TranspiledTotalVariation()

    torch_args = (
        torch.ones((2, 3, 4, 4)),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args).data
    transpiled_res = transpiled_loss_fn(*transpiled_args).data

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_PSNRLoss(target_framework, mode, backend_compile):
    print("kornia.losses.PSNRLoss")

    if backend_compile:
        pytest.skip()

    TranspiledPSNRLoss = ivy.transpiled(kornia.losses.PSNRLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.PSNRLoss(2.)
    transpiled_loss_fn = TranspiledPSNRLoss(2.)

    torch_args = (
        torch.ones(1),
        1.2 * torch.ones(1),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_InverseDepthSmoothnessLoss(target_framework, mode, backend_compile):
    print("kornia.losses.InverseDepthSmoothnessLoss")

    if backend_compile:
        pytest.skip()

    TranspiledInverseDepthSmoothnessLoss = ivy.transpiled(kornia.losses.InverseDepthSmoothnessLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.InverseDepthSmoothnessLoss()
    transpiled_loss_fn = TranspiledInverseDepthSmoothnessLoss()

    torch_args = (
        torch.rand(1, 1, 4, 5),
        torch.rand(1, 3, 4, 5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_CharbonnierLoss(target_framework, mode, backend_compile):
    print("kornia.losses.CharbonnierLoss")

    if backend_compile:
        pytest.skip()

    TranspiledCharbonnierLoss = ivy.transpiled(kornia.losses.CharbonnierLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.CharbonnierLoss(reduction="mean")
    transpiled_loss_fn = TranspiledCharbonnierLoss(reduction="mean")

    torch_args = (
        torch.randn(2, 3, 32, 2107),
        torch.randn(2, 3, 32, 2107),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_WelschLoss(target_framework, mode, backend_compile):
    print("kornia.losses.WelschLoss")

    if backend_compile:
        pytest.skip()

    TranspiledWelschLoss = ivy.transpiled(kornia.losses.WelschLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.WelschLoss(reduction="mean")
    transpiled_loss_fn = TranspiledWelschLoss(reduction="mean")

    torch_args = (
        torch.randn(2, 3, 32, 1904),
        torch.randn(2, 3, 32, 1904),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_CauchyLoss(target_framework, mode, backend_compile):
    print("kornia.losses.CauchyLoss")

    if backend_compile:
        pytest.skip()

    TranspiledCauchyLoss = ivy.transpiled(kornia.losses.CauchyLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.CauchyLoss(reduction="mean")
    transpiled_loss_fn = TranspiledCauchyLoss(reduction="mean")

    torch_args = (
        torch.randn(2, 3, 32, 2107),
        torch.randn(2, 3, 32, 2107),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_GemanMcclureLoss(target_framework, mode, backend_compile):
    print("kornia.losses.GemanMcclureLoss")

    if backend_compile:
        pytest.skip()

    TranspiledGemanMcclureLoss = ivy.transpiled(kornia.losses.GemanMcclureLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.GemanMcclureLoss(reduction="mean")
    transpiled_loss_fn = TranspiledGemanMcclureLoss(reduction="mean")

    torch_args = (
        torch.randn(2, 3, 32, 2107),
        torch.randn(2, 3, 32, 2107),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_BinaryFocalLossWithLogits(target_framework, mode, backend_compile):
    print("kornia.losses.BinaryFocalLossWithLogits")

    if backend_compile:
        pytest.skip()

    TranspiledBinaryFocalLossWithLogits = ivy.transpiled(kornia.losses.BinaryFocalLossWithLogits, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction="mean")
    transpiled_loss_fn = TranspiledBinaryFocalLossWithLogits(alpha=0.25, gamma=2.0, reduction="mean")

    torch_args = (
        torch.randn(1, 3, 5),
        torch.randint(2, (1, 3, 5)),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_DiceLoss(target_framework, mode, backend_compile):
    print("kornia.losses.DiceLoss")

    if backend_compile:
        pytest.skip()

    TranspiledDiceLoss = ivy.transpiled(kornia.losses.DiceLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.DiceLoss()
    transpiled_loss_fn = TranspiledDiceLoss()

    torch_args = (
        torch.randn(1, 5, 3, 5),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_TverskyLoss(target_framework, mode, backend_compile):
    print("kornia.losses.TverskyLoss")

    if backend_compile:
        pytest.skip()

    TranspiledTverskyLoss = ivy.transpiled(kornia.losses.TverskyLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.TverskyLoss(alpha=0.5, beta=0.5)
    transpiled_loss_fn = TranspiledTverskyLoss(alpha=0.5, beta=0.5)

    torch_args = (
        torch.randn(1, 5, 3, 5),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_FocalLoss(target_framework, mode, backend_compile):
    print("kornia.losses.FocalLoss")

    if backend_compile:
        pytest.skip()

    TranspiledFocalLoss = ivy.transpiled(kornia.losses.FocalLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    transpiled_loss_fn = TranspiledFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    torch_args = (
        torch.randn(1, 5, 3, 5),
        torch.randint(5, (1, 3, 5)),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_LovaszHingeLoss(target_framework, mode, backend_compile):
    print("kornia.losses.LovaszHingeLoss")

    if backend_compile:
        pytest.skip()

    TranspiledLovaszHingeLoss = ivy.transpiled(kornia.losses.LovaszHingeLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.LovaszHingeLoss()
    transpiled_loss_fn = TranspiledLovaszHingeLoss()

    torch_args = (
        torch.randn(1, 1, 3, 5),
        torch.empty(1, 3, 5, dtype=torch.long).random_(1),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)


def test_LovaszSoftmaxLoss(target_framework, mode, backend_compile):
    print("kornia.losses.LovaszSoftmaxLoss")

    if backend_compile:
        pytest.skip()

    TranspiledLovaszSoftmaxLoss = ivy.transpiled(kornia.losses.LovaszSoftmaxLoss, source="torch", target=target_framework)

    torch_loss_fn = kornia.losses.LovaszSoftmaxLoss()
    transpiled_loss_fn = TranspiledLovaszSoftmaxLoss()

    torch_args = (
        torch.randn(1, 5, 3, 5),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    torch_res = torch_loss_fn(*torch_args)
    transpiled_res = transpiled_loss_fn(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_res)
    transpiled_np = _nest_array_to_numpy(transpiled_res)
    _check_allclose(orig_np, transpiled_np)

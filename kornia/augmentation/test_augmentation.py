from helpers import (
    _check_allclose,
    _check_shape_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #

def _test_augmentation_class(
    augmentation_cls,
    target,
    init_args=(),
    init_kwargs={},
    call_args=(),
    call_kwargs={},
    deterministic_output=True,
    backend_compile=False,
    tolerance=1e-3,
):
    if backend_compile:
        pytest.skip()

    transpiled_cls = ivy.transpile(augmentation_cls, source="torch", target=target)

    torch_aug = augmentation_cls(*init_args, **init_kwargs)
    transpiled_init_args = _nest_torch_tensor_to_new_framework(init_args, target)
    transpiled_init_kwargs = _nest_torch_tensor_to_new_framework(init_kwargs, target)
    transpiled_aug = transpiled_cls(*transpiled_init_args, **transpiled_init_kwargs)

    assert dir(torch_aug) == dir(transpiled_aug), f"attributes/methods of transpiled object do not align with the original - orig: {dir(torch_aug)} != transpiled: {dir(transpiled_aug)}"

    torch_out = torch_aug(*call_args, **call_kwargs)
    transpiled_call_args = _nest_torch_tensor_to_new_framework(call_args, target)
    transpiled_call_kwargs = _nest_torch_tensor_to_new_framework(call_kwargs, target)
    transpiled_out = transpiled_aug(*transpiled_call_args, **transpiled_call_kwargs)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)

    _check_shape_allclose(orig_np, transpiled_np)

    if deterministic_output:
        orig_np = _nest_array_to_numpy(torch_out)
        transpiled_np = _nest_array_to_numpy(transpiled_out)
        _check_allclose(orig_np, transpiled_np, tolerance=tolerance)
    else:
        # TODO: add value test to ensure the output of 
        # `transpiled_aug(*transpiled_call_args, **transpiled_call_kwargs)`
        # changes most times it runs ??
        # or verify the mean and std are within a given range??
        pass


# Tests #
# ----- #

def test_ColorJiggle(target_framework, mode, backend_compile):
    print("kornia.augmentation.ColorJiggle")
    
    init_args = (0.1, 0.1, 0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}
    
    # TODO: test previous parameter state

    _test_augmentation_class(
        kornia.augmentation.ColorJiggle,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_ColorJitter(target_framework, mode, backend_compile):
    print("kornia.augmentation.ColorJitter")

    init_args = (0.1, 0.1, 0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.ColorJitter,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAutoContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAutoContrast")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.ones(5, 3, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAutoContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomBoxBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomBoxBlur")

    init_args = ((7, 7),)
    init_kwargs = {}
    call_args = (torch.ones(1, 1, 24, 24),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomBoxBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomBrightness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomBrightness")

    init_args = ()
    init_kwargs = {"brightness": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomBrightness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomChannelDropout(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomChannelDropout")

    init_args = ()
    init_kwargs = {"num_drop_channels": 1, "fill_value": 0.0, "p": 1.0}
    call_args = (torch.ones(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomChannelDropout,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomChannelShuffle(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomChannelShuffle")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.arange(1*2*2*2.).view(1,2,2,2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomChannelShuffle,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomClahe(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomClahe")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.rand(2, 3, 10, 20),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomClahe,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomContrast")

    init_args = ()
    init_kwargs = {"contrast": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomEqualize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomEqualize")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomEqualize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGamma(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGamma")

    init_args = ((0.5, 2.), (1.5, 1.5))
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGamma,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianBlur")

    init_args = ((3, 3), (0.1, 2.0))
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.5, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGaussianNoise(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGaussianNoise")

    init_args = ()
    init_kwargs = {"mean": 0., "std": 1., "p": 1.}
    call_args = (torch.ones(1, 1, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGaussianNoise,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomGrayscale(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomGrayscale")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn((1, 3, 3, 3)),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomGrayscale,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHue(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHue")

    init_args = ()
    init_kwargs = {"hue": (-0.5, 0.5), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHue,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomInvert(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomInvert")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomInvert,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomJPEG(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomJPEG")

    init_args = ()
    init_kwargs = {"jpeg_quality": (1.0, 50.0), "p": 1.}
    call_args = (0.1904 * torch.ones(2, 3, 32, 32), )
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomJPEG,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomLinearCornerIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomLinearCornerIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.25, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomLinearCornerIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomLinearIllumination(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomLinearIllumination")

    init_args = ()
    init_kwargs = {"gain": 0.25, "p": 1.}
    call_args = (torch.ones(1, 3, 3, 3) * 0.5,)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomLinearIllumination,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMedianBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMedianBlur")

    init_args = ((3, 3),)
    init_kwargs = {"p": 1}
    call_args = (torch.ones(1, 1, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMedianBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMotionBlur(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMotionBlur")

    init_args = (3, 35., 0.5)
    init_kwargs = {"p": 1.}
    call_args = (torch.ones(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMotionBlur,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlanckianJitter(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlanckianJitter")

    init_args = ()
    init_kwargs = {"mode": "blackbody", "select_from": [23, 24, 1, 2]}
    call_args = (torch.randn(2, 3, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlanckianJitter,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaBrightness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaBrightness")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaBrightness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaContrast(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaContrast")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaContrast,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPlasmaShadow(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPlasmaShadow")

    init_args = ()
    init_kwargs = {"roughness": (0.1, 0.7), "p": 1.}
    call_args = (torch.ones(1, 1, 3, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPlasmaShadow,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPosterize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPosterize")

    init_args = (3.,)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPosterize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRain(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRain")

    init_args = ()
    init_kwargs = {"p": 1, "drop_height": (1,2), "drop_width": (1,2), "number_of_drops": (1,1)}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRain,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRGBShift(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRGBShift")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 3, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRGBShift,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSaltAndPepperNoise(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSaltAndPepperNoise")

    init_args = ()
    init_kwargs = {"amount": 0.5, "salt_vs_pepper": 0.5, "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSaltAndPepperNoise,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSaturation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSaturation")

    init_args = ()
    init_kwargs = {"saturation": (0.5, 2.), "p": 1.}
    call_args = (torch.rand(1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSaturation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSharpness(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSharpness")

    init_args = (1.,)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSharpness,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSnow(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSnow")

    init_args = ()
    init_kwargs = {"p": 1.0, "snow_coefficient": (0.1, 0.6), "brightness": (1.0, 5.0)}
    call_args = (torch.rand(2, 3, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSnow,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomSolarize(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomSolarize")

    init_args = (0.1, 0.1)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomSolarize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_CenterCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.CenterCrop")

    init_args = (2,)
    init_kwargs = {"p": 1., "cropping_mode": "resample"}
    call_args = (torch.randn(1, 1, 4, 4),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.CenterCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )
    
    # TODO: test CenterCrop.inverse


def test_PadTo(target_framework, mode, backend_compile):
    print("kornia.augmentation.PadTo")

    init_args = ((4, 5),)
    init_kwargs = {"pad_value": 1.}
    call_args = (torch.tensor([[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.PadTo,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAffine(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAffine")

    init_args = ((-15., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAffine,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )

    # TODO: test RandomAffine.inverse, RandomAffine.transform_matrix


def test_RandomCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCrop")

    init_args = ((2, 2),)
    init_kwargs = {"p": 1., "cropping_mode": "resample"}
    call_args = (torch.arange(1*1*3*3.).view(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomElasticTransform(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomElasticTransform")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.ones(1, 1, 2, 2),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomElasticTransform,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomErasing(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomErasing")

    init_args = ((.4, .8), (.3, 1/.3))
    init_kwargs = {"p": 0.5}
    call_args = (torch.ones(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomErasing,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomFisheye(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomFisheye")

    init_args = (
        torch.tensor([-.3, .3]),
        torch.tensor([-.3, .3]),
        torch.tensor([.9, 1.]),
    )
    init_kwargs = {}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomFisheye,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHorizontalFlip(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHorizontalFlip")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.tensor([[[[0., 0., 0.],
                                 [0., 0., 0.],
                                 [0., 1., 1.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHorizontalFlip,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomPerspective(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomPerspective")

    init_args = (0.5,)
    init_kwargs = {"p": 0.5}
    call_args = (torch.tensor([[[[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomPerspective,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomResizedCrop(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomResizedCrop")

    init_args = ()
    init_kwargs = {"size": (3, 3), "scale": (3., 3.), "ratio": (2., 2.), "p": 1., "cropping_mode": "resample"}
    call_args = (torch.tensor([[[0., 1., 2.],
                                [3., 4., 5.],
                                [6., 7., 8.]]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomResizedCrop,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRotation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRotation")

    init_args = ()
    init_kwargs = {"degrees": 45.0, "p": 1.}
    call_args = (torch.tensor([[1., 0., 0., 2.],
                               [0., 0., 0., 0.],
                               [0., 1., 2., 0.],
                               [0., 0., 1., 2.]]),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRotation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomShear(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomShear")

    init_args = ((-5., 2., 5., 10.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomShear,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomThinPlateSpline(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomThinPlateSpline")

    init_args = ()
    init_kwargs = {}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomThinPlateSpline,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomVerticalFlip(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomVerticalFlip")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 3, 32, 32),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomVerticalFlip,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomCutMixV2(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCutMixV2")

    input = torch.rand(2, 1, 3, 3)
    input[0] = torch.ones((1, 3, 3))
    label = torch.tensor([0, 1])

    init_args = ()
    init_kwargs = {"data_keys": ["input", "class"]}
    call_args = (input, label)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCutMixV2,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomJigsaw(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomJigsaw")

    init_args = ((4, 4),)
    init_kwargs = {}
    call_args = (torch.randn(8, 3, 256, 256),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomJigsaw,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMixUpV2(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMixUpV2")

    init_args = ()
    init_kwargs = {"data_keys": ["input", "class"]}
    call_args = (torch.rand(2, 1, 3, 3), torch.tensor([0, 1]))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMixUpV2,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMosaic(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMosaic")

    init_args = ((300, 300),)
    init_kwargs = {"data_keys": ["input", "bbox_xyxy"]}
    call_args = (
        torch.randn(8, 3, 224, 224),
        torch.tensor([[
            [70, 5, 150, 100],
            [60, 180, 175, 220],
        ]]).repeat(8, 1, 1),
    )
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMosaic,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomTransplantation(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomTransplantation")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(2, 3, 5, 5), torch.randint(0, 3, (2, 5, 5)))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomTransplantation,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_CenterCrop3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.CenterCrop3D")

    init_args = (2,)
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 1, 2, 4, 6),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.CenterCrop3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomAffine3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomAffine3D")

    init_args = ((15., 20., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomAffine3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomCrop3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomCrop3D")

    init_args = ((2, 2, 2),)
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomCrop3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomDepthicalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomDepthicalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomDepthicalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomHorizontalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomHorizontalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomHorizontalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomRotation3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomRotation3D")

    init_args = ((15., 20., 20.),)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomRotation3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomVerticalFlip3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomVerticalFlip3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.eye(3).repeat(3, 1, 1),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomVerticalFlip3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomEqualize3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomEqualize3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomEqualize3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomMotionBlur3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomMotionBlur3D")

    init_args = (3, 35., 0.5)
    init_kwargs = {"p": 1.}
    call_args = (torch.rand(1, 1, 3, 5, 5),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomMotionBlur3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_RandomTransplantation3D(target_framework, mode, backend_compile):
    print("kornia.augmentation.RandomTransplantation3D")

    init_args = ()
    init_kwargs = {"p": 1.}
    call_args = (torch.randn(2, 3, 5, 5), torch.randint(0, 3, (2, 5, 5)))
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.RandomTransplantation3D,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_Denormalize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Denormalize")

    init_args = ()
    init_kwargs = {"mean": torch.zeros(1, 4), "std": torch.ones(1, 4)}
    call_args = (torch.rand(1, 4, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Denormalize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_Normalize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Normalize")

    init_args = ()
    init_kwargs = {"mean": torch.zeros(4), "std": torch.ones(4)}
    call_args = (torch.rand(1, 4, 3, 3),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Normalize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=False,
        backend_compile=backend_compile,
    )


def test_LongestMaxSize(target_framework, mode, backend_compile):
    print("kornia.augmentation.LongestMaxSize")

    init_args = (100,)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 200, 200),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.LongestMaxSize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )


def test_Resize(target_framework, mode, backend_compile):
    print("kornia.augmentation.Resize")

    init_args = ((100, 100),)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 50, 50),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.Resize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )


def test_SmallestMaxSize(target_framework, mode, backend_compile):
    print("kornia.augmentation.SmallestMaxSize")

    init_args = (100,)
    init_kwargs = {}
    call_args = (torch.rand(10, 3, 50, 50),)
    call_kwargs = {}

    _test_augmentation_class(
        kornia.augmentation.SmallestMaxSize,
        target_framework,
        init_args,
        init_kwargs,
        call_args,
        call_kwargs,
        deterministic_output=True,
        backend_compile=backend_compile,
    )

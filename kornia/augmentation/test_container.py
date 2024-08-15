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


# Tests #
# ----- #

def test_AugmentationSequential(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.AugmentationSequential")

    if backend_compile:
        pytest.skip()

    TranspiledAugmentationSequential = ivy.transpile(
        kornia.augmentation.container.AugmentationSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.AugmentationSequential(
        kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        kornia.augmentation.RandomAffine(360, p=1.0),
        data_keys=["input", "mask", "bbox", "keypoints"],
        same_on_batch=False,
        random_apply=10,
    )
    transpiled_aug_list = TranspiledAugmentationSequential(
        TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        TranspiledRandomAffine(360, p=1.0),
        data_keys=["input", "mask", "bbox", "keypoints"],
        same_on_batch=False,
        random_apply=10,
    )

    torch_args = (
        torch.randn(2, 3, 5, 6),
        torch.ones(2, 3, 5, 6),
        torch.tensor([[
            [1., 1.],
            [2., 1.],
            [2., 2.],
            [1., 2.],
        ]]).expand(2, 1, -1, -1),
        torch.tensor([[[1., 1.]]]).expand(2, -1, -1),
    )
    torch_out = torch_aug_list(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_ManyToManyAugmentationDispather(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.ManyToManyAugmentationDispather")

    if backend_compile:
        pytest.skip()

    TranspiledManyToManyAugmentationDispather = ivy.transpile(
        kornia.augmentation.container.ManyToManyAugmentationDispather,
        source="torch",
        target=target_framework,
    )
    TranspiledAugmentationSequential = ivy.transpile(
        kornia.augmentation.container.AugmentationSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.ManyToManyAugmentationDispather(
        kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.augmentation.RandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        ),
        kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.augmentation.RandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        )
    )
    transpiled_aug_list = TranspiledManyToManyAugmentationDispather(
        TranspiledAugmentationSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            TranspiledRandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        ),
        TranspiledAugmentationSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            TranspiledRandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        )
    )

    torch_args = (
        (torch.randn(2, 3, 5, 6), torch.ones(2, 3, 5, 6)),
        (torch.randn(2, 3, 5, 6), torch.ones(2, 3, 5, 6)),
    )
    torch_out = torch_aug_list(*torch_args)
    
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_ManyToOneAugmentationDispather(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.ManyToOneAugmentationDispather")

    if backend_compile:
        pytest.skip()

    TranspiledManyToOneAugmentationDispather = ivy.transpile(
        kornia.augmentation.container.ManyToOneAugmentationDispather,
        source="torch",
        target=target_framework,
    )
    TranspiledAugmentationSequential = ivy.transpile(
        kornia.augmentation.container.AugmentationSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.ManyToOneAugmentationDispather(
        kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.augmentation.RandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        ),
        kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            kornia.augmentation.RandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        )
    )
    transpiled_aug_list = TranspiledManyToOneAugmentationDispather(
        TranspiledAugmentationSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            TranspiledRandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        ),
        TranspiledAugmentationSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
            TranspiledRandomAffine(360, p=1.0),
            data_keys=["input", "mask",],
        )
    )

    torch_args = (
        torch.randn(2, 3, 5, 6),
        torch.ones(2, 3, 5, 6),
    )
    torch_out = torch_aug_list(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_ImageSequential(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.ImageSequential")

    if backend_compile:
        pytest.skip()

    TranspiledImageSequential = ivy.transpile(
        kornia.augmentation.container.ImageSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )
    TranspiledBgrToRgb = ivy.transpile(
        kornia.color.BgrToRgb,
        source="torch",
        target=target_framework,
    )
    TranspiledMedianBlur = ivy.transpile(
        kornia.filters.MedianBlur,
        source="torch",
        target=target_framework,
    )
    TranspiledInvert = ivy.transpile(
        kornia.enhance.Invert,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomMixUpV2 = ivy.transpile(
        kornia.augmentation.RandomMixUpV2,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.ImageSequential(
        kornia.color.BgrToRgb(),
        kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        kornia.filters.MedianBlur((3, 3)),
        kornia.augmentation.RandomAffine(360, p=1.0),
        kornia.enhance.Invert(),
        kornia.augmentation.RandomMixUpV2(p=1.0),
        same_on_batch=True,
        random_apply=10,
    )
    transpiled_aug_list = TranspiledImageSequential(
        TranspiledBgrToRgb(),
        TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        TranspiledMedianBlur((3, 3)),
        TranspiledRandomAffine(360, p=1.0),
        TranspiledInvert(),
        TranspiledRandomMixUpV2(p=1.0),
        same_on_batch=True,
        random_apply=10,
    )

    torch_args = (
        torch.randn(2, 3, 5, 6),
    )
    torch_out = torch_aug_list(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_PatchSequential(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.PatchSequential")

    if backend_compile:
        pytest.skip()

    TranspiledPatchSequential = ivy.transpile(
        kornia.augmentation.container.PatchSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledImageSequential = ivy.transpile(
        kornia.augmentation.container.ImageSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomPerspective = ivy.transpile(
        kornia.augmentation.RandomPerspective,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomSolarize = ivy.transpile(
        kornia.augmentation.RandomSolarize,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.PatchSequential(
        kornia.augmentation.container.ImageSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            kornia.augmentation.RandomPerspective(0.2, p=0.5),
            kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.5),
        ),
        kornia.augmentation.RandomAffine(360, p=1.0),
        kornia.augmentation.container.ImageSequential(
            kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            kornia.augmentation.RandomPerspective(0.2, p=0.5),
            kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.5),
        ),
        kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.1),
        grid_size=(2,2),
        patchwise_apply=True,
        same_on_batch=True,
        random_apply=False,
    )
    transpiled_aug_list = TranspiledPatchSequential(
        TranspiledImageSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            TranspiledRandomPerspective(0.2, p=0.5),
            TranspiledRandomSolarize(0.1, 0.1, p=0.5),
        ),
        TranspiledRandomAffine(360, p=1.0),
        TranspiledImageSequential(
            TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
            TranspiledRandomPerspective(0.2, p=0.5),
            TranspiledRandomSolarize(0.1, 0.1, p=0.5),
        ),
        TranspiledRandomSolarize(0.1, 0.1, p=0.1),
        grid_size=(2,2),
        patchwise_apply=True,
        same_on_batch=True,
        random_apply=False,
    )

    torch_args = (
        torch.randn(2, 3, 224, 224),
    )
    torch_out = torch_aug_list(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)


def test_VideoSequential(target_framework, mode, backend_compile):
    print("kornia.augmentation.container.VideoSequential")

    if backend_compile:
        pytest.skip()

    TranspiledVideoSequential = ivy.transpile(
        kornia.augmentation.container.VideoSequential,
        source="torch",
        target=target_framework,
    )
    TranspiledColorJiggle = ivy.transpile(
        kornia.augmentation.ColorJiggle,
        source="torch",
        target=target_framework,
    )
    TranspiledRandomAffine = ivy.transpile(
        kornia.augmentation.RandomAffine,
        source="torch",
        target=target_framework,
    )
    TranspiledBgrToRgb = ivy.transpile(
        kornia.color.BgrToRgb,
        source="torch",
        target=target_framework,
    )

    torch_aug_list = kornia.augmentation.container.VideoSequential(
        kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        kornia.color.BgrToRgb(),
        kornia.augmentation.RandomAffine(360, p=1.0),
        random_apply=10,
        data_format="BCTHW",
        same_on_frame=True
    )
    transpiled_aug_list =  TranspiledVideoSequential(
        TranspiledColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0),
        TranspiledBgrToRgb(),
        TranspiledRandomAffine(360, p=1.0),
        random_apply=10,
        data_format="BCTHW",
        same_on_frame=True
    )

    torch_args = (
        torch.randn(2, 3, 1, 5, 6).repeat(1, 1, 4, 1, 1),
    )
    torch_out = torch_aug_list(*torch_args)

    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    transpiled_out = transpiled_aug_list(*transpiled_args)

    orig_np = _nest_array_to_numpy(torch_out)
    transpiled_np = _nest_array_to_numpy(transpiled_out)
    _check_shape_allclose(orig_np, transpiled_np)

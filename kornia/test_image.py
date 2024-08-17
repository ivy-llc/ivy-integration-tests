from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _nest_array_to_numpy,
)

from color_operations.colorspace import ColorSpace
import ivy
import kornia
import numpy as np
import pytest
import torch


# Tests #
# ----- #

def test_ImageSize(target_framework, mode, backend_compile):
    print("kornia.image.ImageSize")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpile(kornia.image.ImageSize, source="torch", target=target_framework)

    size = TranspiledImageSize(3, 4)
    assert size.height == 3
    assert size.width == 4


def test_PixelFormat(target_framework, mode, backend_compile):
    print("kornia.image.PixelFormat")

    if backend_compile:
        pytest.skip()

    TranspiledPixelFormat = ivy.transpile(kornia.image.PixelFormat, source="torch", target=target_framework)

    pixel_format = TranspiledPixelFormat(ColorSpace.rgb, 8)
    assert pixel_format.color_space.name == "rgb"
    assert pixel_format.bit_depth == 8


def test_ChannelsOrder(target_framework, mode, backend_compile):
    print("kornia.image.ChannelsOrder")

    if backend_compile:
        pytest.skip()

    TranspiledChannelsOrder = ivy.transpile(kornia.image.ChannelsOrder, source="torch", target=target_framework)

    assert TranspiledChannelsOrder.CHANNELS_FIRST.value == 0
    assert TranspiledChannelsOrder.CHANNELS_LAST.value == 1


def test_ImageLayout(target_framework, mode, backend_compile):
    print("kornia.image.ImageLayout")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpile(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledChannelsOrder = ivy.transpile(kornia.image.ChannelsOrder, source="torch", target=target_framework)
    TranspiledImageLayout = ivy.transpile(kornia.image.ImageLayout, source="torch", target=target_framework)

    layout = TranspiledImageLayout(TranspiledImageSize(3, 4), 3, TranspiledChannelsOrder.CHANNELS_LAST)

    assert layout.image_size.height == 3
    assert layout.image_size.width == 4
    assert layout.channels == 3
    assert layout.channels_order.value == 1


def test_Image(target_framework, mode, backend_compile):
    print("kornia.image.Image")

    if backend_compile:
        pytest.skip()

    TranspiledImageSize = ivy.transpile(kornia.image.ImageSize, source="torch", target=target_framework)
    TranspiledPixelFormat = ivy.transpile(kornia.image.PixelFormat, source="torch", target=target_framework)
    TranspiledChannelsOrder = ivy.transpile(kornia.image.ChannelsOrder, source="torch", target=target_framework)
    TranspiledImageLayout = ivy.transpile(kornia.image.ImageLayout, source="torch", target=target_framework)
    TranspiledImage = ivy.transpile(kornia.image.Image, source="torch", target=target_framework)

    torch_data = torch.randint(0, 255, (3, 4, 5), dtype=torch.uint8)
    transpiled_data = _array_to_new_backend(torch_data, target_framework)

    # torch
    pixel_format = kornia.image.PixelFormat(
        color_space=ColorSpace.rgb,
        bit_depth=8,
    )
    layout = kornia.image.ImageLayout(
        image_size=kornia.image.ImageSize(4, 5),
        channels=3,
        channels_order=kornia.image.ChannelsOrder.CHANNELS_FIRST,
    )
    torch_img = kornia.image.Image(torch_data, pixel_format, layout)

    # transpiled
    pixel_format = TranspiledPixelFormat(
        color_space=ColorSpace.rgb,
        bit_depth=8,
    )
    layout = TranspiledImageLayout(
        image_size=TranspiledImageSize(4, 5),
        channels=3,
        channels_order=TranspiledChannelsOrder.CHANNELS_FIRST,
    )
    transpiled_img = TranspiledImage(transpiled_data, pixel_format, layout)
    assert transpiled_img.channels == 3

    assert dir(torch_img) == dir(transpiled_img), f"attributes/methods of transpiled object do not align with the original - orig: {dir(torch_img)} != transpiled: {dir(transpiled_img)}"

    orig_np = _nest_array_to_numpy(torch_img.data)
    transpiled_np = _nest_array_to_numpy(transpiled_img.data)
    _check_allclose(orig_np, transpiled_np)


def test_Image_from_numpy(target_framework, mode, backend_compile):
    print("kornia.image.Image.from_numpy")

    if backend_compile:
        pytest.skip()

    TranspiledImage = ivy.transpile(kornia.image.Image, source="torch", target=target_framework)

    data = np.ones((4, 5, 3), dtype=np.uint8)
    img = TranspiledImage.from_numpy(data, color_space=ColorSpace.rgb)

    assert img.channels == 3
    assert img.width == 5
    assert img.height == 4
    _check_allclose(data, _nest_array_to_numpy(img.data))


def test_Image_from_dlpack(target_framework, mode, backend_compile):
    print("kornia.image.Image.from_dlpack")

    if backend_compile:
        pytest.skip()

    TranspiledImage = ivy.transpile(kornia.image.Image, source="torch", target=target_framework)

    x = np.ones((4, 5, 3))
    img = TranspiledImage.from_dlpack(x.__dlpack__())
    _check_allclose(x, _nest_array_to_numpy(img.data))


def test_Image_to_numpy(target_framework, mode, backend_compile):
    print("kornia.image.Image.to_numpy")

    if backend_compile:
        pytest.skip()

    TranspiledImage = ivy.transpile(kornia.image.Image, source="torch", target=target_framework)

    data = np.ones((4, 5, 3), dtype=np.uint8)
    img = TranspiledImage.from_numpy(data, color_space=ColorSpace.rgb)
    img_data = img.to_numpy()
    _check_allclose(data, img_data)

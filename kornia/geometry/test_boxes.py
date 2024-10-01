from helpers import (
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #


def _check_boxes_same(torch_boxes, transpiled_boxes):
    dir_torch_boxes = set(dir(torch_boxes))
    dir_transpiled_boxes = set([d for d in dir(transpiled_boxes) if d not in ("__already_s2s",)])
    assert dir_torch_boxes == dir_transpiled_boxes, f"attributes/methods of transpiled class do not align with the original - orig: {dir(torch_boxes)} != transpiled: {dir(transpiled_boxes)}"
    _to_numpy_and_allclose(torch_boxes.data, transpiled_boxes.data)


# Tests #
# ----- #

def test_Boxes(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledBoxes = ivy.transpile(kornia.geometry.boxes.Boxes, source="torch", target=target_framework)

    torch_args = (
        torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]]),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    # test .from_tensor
    torch_boxes = kornia.geometry.boxes.Boxes.from_tensor(*torch_args, mode="xyxy")
    transpiled_boxes = TranspiledBoxes.from_tensor(*transpiled_args, mode="xyxy")
    _check_boxes_same(torch_boxes, transpiled_boxes)

    # test .compute_area
    torch_area = torch_boxes.compute_area()
    transpiled_area = transpiled_boxes.compute_area()
    _to_numpy_and_allclose(torch_area, transpiled_area)

    # test .get_boxes_shape
    torch_heights, torch_widths = torch_boxes.get_boxes_shape()
    transpiled_heights, transpiled_widths = transpiled_boxes.get_boxes_shape()
    _to_numpy_and_allclose(torch_heights, transpiled_heights)
    _to_numpy_and_allclose(torch_widths, transpiled_widths)

    # test .merge
    torch_x = torch.as_tensor([[6, 6, 10, 10], [6, 6, 10, 10]])
    transpiled_x = _nest_torch_tensor_to_new_framework(torch_x, target_framework)
    merge_boxes = kornia.geometry.boxes.Boxes.from_tensor(torch_x, mode="xyxy")
    transpiled_merge_boxes = TranspiledBoxes.from_tensor(transpiled_x, mode="xyxy")
    torch_merged_boxes = torch_boxes.merge(merge_boxes)
    transpiled_merged_boxes = transpiled_boxes.merge(transpiled_merge_boxes)
    _check_boxes_same(torch_merged_boxes, transpiled_merged_boxes)

    # test .to_mask
    height, width = 10, 10
    torch_mask = torch_boxes.to_mask(height, width)
    transpiled_mask = transpiled_boxes.to_mask(height, width)
    _to_numpy_and_allclose(torch_mask, transpiled_mask)

    # test .to_tensor
    torch_tensor = torch_boxes.to_tensor(mode="xyxy")
    transpiled_tensor = transpiled_boxes.to_tensor(mode="xyxy")
    _to_numpy_and_allclose(torch_tensor, transpiled_tensor)

    # test .transform_boxes
    transform_matrix = torch.eye(3)
    transpiled_transform_matrix = _nest_torch_tensor_to_new_framework(transform_matrix, target_framework)
    torch_transformed_boxes = torch_boxes.transform_boxes(transform_matrix)
    transpiled_transformed_boxes = transpiled_boxes.transform_boxes(transpiled_transform_matrix)
    _check_boxes_same(torch_transformed_boxes, transpiled_transformed_boxes)

    # test .translate
    translate_size = torch.as_tensor([[2, 3]])
    transpiled_translate_size = _nest_torch_tensor_to_new_framework(translate_size, target_framework)
    torch_translated_boxes = torch_boxes.translate(translate_size)
    transpiled_translated_boxes = transpiled_boxes.translate(transpiled_translate_size)
    _check_boxes_same(torch_translated_boxes, transpiled_translated_boxes)


def test_Boxes3D(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes3D")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledBoxes3D = ivy.transpile(kornia.geometry.boxes.Boxes3D, source="torch", target=target_framework)

    torch_args = (
        torch.as_tensor([[0, 3, 6, 1, 4, 8], [5, 1, 3, 8, 4, 9]]),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    # test .from_tensor
    torch_boxes3d = kornia.geometry.boxes.Boxes3D.from_tensor(*torch_args, mode="xyzxyz")
    transpiled_boxes3d = TranspiledBoxes3D.from_tensor(*transpiled_args, mode="xyzxyz")
    _check_boxes_same(torch_boxes3d, transpiled_boxes3d)

    # test .get_boxes_shape
    torch_depths, torch_heights, torch_widths = torch_boxes3d.get_boxes_shape()
    transpiled_depths, transpiled_heights, transpiled_widths = transpiled_boxes3d.get_boxes_shape()
    _to_numpy_and_allclose(torch_depths, transpiled_depths)
    _to_numpy_and_allclose(torch_heights, transpiled_heights)
    _to_numpy_and_allclose(torch_widths, transpiled_widths)

    # test .to_mask
    depth, height, width = 10, 10, 10
    torch_mask = torch_boxes3d.to_mask(depth, height, width)
    transpiled_mask = transpiled_boxes3d.to_mask(depth, height, width)
    _to_numpy_and_allclose(torch_mask, transpiled_mask)

    # test .to_tensor
    torch_tensor = torch_boxes3d.to_tensor(mode="xyzxyz")
    transpiled_tensor = transpiled_boxes3d.to_tensor(mode="xyzxyz")
    _to_numpy_and_allclose(torch_tensor, transpiled_tensor)

    # test .transform_boxes
    transform_matrix = torch.eye(4)
    transpiled_transform_matrix = _nest_torch_tensor_to_new_framework(transform_matrix, target_framework)
    torch_transformed_boxes3d = torch_boxes3d.transform_boxes(transform_matrix)
    transpiled_transformed_boxes3d = transpiled_boxes3d.transform_boxes(transpiled_transform_matrix)
    _check_boxes_same(torch_transformed_boxes3d, transpiled_transformed_boxes3d)

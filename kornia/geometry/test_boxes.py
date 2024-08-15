from helpers import (
    _array_to_new_backend,
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
)

import ivy
import kornia
import pytest
import torch


# Globals #
# ------- #

try:
    TFBoxes = ivy.transpile(kornia.geometry.boxes.Boxes, source="torch", target="tensorflow")
except:
    TFBoxes = None
try:
    JAXBoxes = ivy.transpile(kornia.geometry.boxes.Boxes, source="torch", target="jax")
except:
    JAXBoxes = None
try:
    NPBoxes = ivy.transpile(kornia.geometry.boxes.Boxes, source="torch", target="numpy")
except:
    NPBoxes = None

transpiled_boxes = {
    "jax": JAXBoxes,
    "numpy": NPBoxes,
    "tensorflow": TFBoxes,
}


# Helpers #
# ------- #

def _check_boxes_same(torch_boxes, transpiled_boxes):
    assert dir(torch_boxes) == dir(transpiled_boxes), f"attributes/methods of transpiled class do not align with the original - orig: {dir(torch_boxes)} != transpiled: {dir(transpiled_boxes)}"

    orig_data = _nest_array_to_numpy(torch_boxes.data)
    transpiled_data = _nest_array_to_numpy(transpiled_boxes.data)

    _check_allclose(orig_data, transpiled_data, tolerance=1e-3) 


def _test_boxes_method(fn_name, target, args=(), kwargs={}, backend_compile=False):
    if backend_compile:
        pytest.skip()
    if TFBoxes is None:
        raise Exception("Failure during kornia.geometry.boxes.Boxes transpilation")

    x = torch.ones((5, 3, 4, 2))
    torch_boxes = kornia.geometry.boxes.Boxes(
        x,
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )
    transpiled_boxes = TFBoxes(
        _array_to_new_backend(x, target),
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )
    _check_boxes_same(torch_boxes, transpiled_boxes)

    torch_method = getattr(torch_boxes, fn_name)
    transpiled_method = getattr(transpiled_boxes, fn_name)

    orig_out = torch_method(*args, **kwargs)
    graph_args = _nest_torch_tensor_to_new_framework(args, target)
    graph_kwargs = _nest_torch_tensor_to_new_framework(kwargs, target)
    graph_out = transpiled_method(*graph_args, **graph_kwargs)

    if isinstance(orig_out, kornia.geometry.boxes.Boxes):
        _check_boxes_same(orig_out, graph_out)
    else:
        orig_np = _nest_array_to_numpy(orig_out)
        graph_np = _nest_array_to_numpy(graph_out)

        _check_allclose(orig_np, graph_np, tolerance=1e-3)
    _check_boxes_same(torch_boxes, transpiled_boxes)


# Tests #
# ----- #


def test_Boxes_compute_area(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes.compute_area")
    _test_boxes_method("compute_area", target_framework, backend_compile=backend_compile)


def test_Boxes_from_tensor(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes.from_tensor")
    _test_boxes_method(
        "from_tensor",
        target_framework,
        args=(torch.rand((10, 3, 4)),),
        kwargs={"validate_boxes": False},
        backend_compile=backend_compile,
    )


def test_Boxes_get_boxes_shape(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes.get_boxes_shape")
    _test_boxes_method(
        "get_boxes_shape",
        target_framework,
        backend_compile=backend_compile,
    )


def test_Boxes_merge(target_framework, mode, backend_compile):
    print("kornia.geometry.boxes.Boxes.merge")

    x1 = torch.ones((5, 3, 4, 2))
    x2 = torch.ones((5, 6, 4, 2))

    torch_boxes1 = kornia.geometry.boxes.Boxes(
        x1,
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )
    torch_boxes2 = kornia.geometry.boxes.Boxes(
        x1,
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )

    transpiled_boxes1 = transpiled_boxes[target_framework](
        _array_to_new_backend(x1, target_framework),
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )
    transpiled_boxes2 = transpiled_boxes[target_framework](
        _array_to_new_backend(x2, target_framework),
        raise_if_not_floating_point=True,
        mode="vertices_plus",
    )

    torch_boxes3 = torch_boxes1.merge(torch_boxes2)
    transpiled_boxes3 = transpiled_boxes1.merge(transpiled_boxes2)
    _check_boxes_same(torch_boxes3, transpiled_boxes3)

    torch_boxes1.merge(torch_boxes2, inplace=True)
    transpiled_boxes1.merge(transpiled_boxes2, inplace=True)
    _check_boxes_same(torch_boxes1, transpiled_boxes1)

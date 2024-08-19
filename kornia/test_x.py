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

def test_ImageClassifierTrainer(target_framework, mode, backend_compile):
    print("kornia.x.ImageClassifierTrainer")

    if backend_compile:
        pytest.skip()

    TranspiledImageClassifierTrainer = ivy.transpile(kornia.x.ImageClassifierTrainer, source="torch", target=target_framework)
    pytest.skip()


def test_SemanticSegmentationTrainer(target_framework, mode, backend_compile):
    print("kornia.x.SemanticSegmentationTrainer")

    if backend_compile:
        pytest.skip()

    TranspiledSemanticSegmentationTrainer = ivy.transpile(kornia.x.SemanticSegmentationTrainer, source="torch", target=target_framework)
    pytest.skip()


def test_ObjectDetectionTrainer(target_framework, mode, backend_compile):
    print("kornia.x.ObjectDetectionTrainer")

    if backend_compile:
        pytest.skip()

    TranspiledObjectDetectionTrainer = ivy.transpile(kornia.x.ObjectDetectionTrainer, source="torch", target=target_framework)
    pytest.skip()


def test_ModelCheckpoint(target_framework, mode, backend_compile):
    print("kornia.x.ModelCheckpoint")

    if backend_compile:
        pytest.skip()

    TranspiledModelCheckpoint = ivy.transpile(kornia.x.ModelCheckpoint, source="torch", target=target_framework)
    pytest.skip()


def test_EarlyStopping(target_framework, mode, backend_compile):
    print("kornia.x.EarlyStopping")

    if backend_compile:
        pytest.skip()

    TranspiledEarlyStopping = ivy.transpile(kornia.x.EarlyStopping, source="torch", target=target_framework)
    pytest.skip()

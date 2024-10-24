import ivy
import kornia
import pytest
import torch


# Tests #
# ----- #

def test_load_image(target_framework, mode, backend_compile):
    print("kornia.io.load_image")

    if backend_compile:
        pytest.skip()

    transpiled_load_image = ivy.transpile(kornia.io.load_image, source="torch", target=target_framework)

    # TODO: add some test of using the transpiled function


def test_write_image(target_framework, mode, backend_compile):
    print("kornia.io.write_image")

    if backend_compile:
        pytest.skip()

    transpiled_load_image = ivy.transpile(kornia.io.write_image, source="torch", target=target_framework)

    # TODO: add some test of using the transpiled function

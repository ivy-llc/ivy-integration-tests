import ivy
import pytest

TARGET_FRAMEWORKS = ["numpy", "jax", "tensorflow", "torch"]
S2S_TARGET_FRAMEWORKS = ["tensorflow"]
BACKEND_COMPILE = False
TARGET = "all"
S2S = False


@pytest.fixture(autouse=True)
def run_around_tests():
    ivy.unset_backend()


def pytest_addoption(parser):
    parser.addoption(
        "--backend-compile",
        action="store_true",
        help="Whether to use backend compilation (such as jax.jit) during testing",
    )
    parser.addoption(
        "--target",
        action="store",
        default="all",
        help="Target for the transpilation tests",
    )
    parser.addoption(
        "--source-to-source",
        action="store_true",
        help="Whether to run the tests on the source-to-source translator or functional transpiler",
    )


def pytest_configure(config):
    getopt = config.getoption

    global BACKEND_COMPILE
    BACKEND_COMPILE = getopt("--backend-compile")

    global TARGET
    TARGET = getopt("--target")

    global S2S
    S2S = getopt("--source-to-source")


def pytest_generate_tests(metafunc):
    configs = list()
    if S2S:
        if TARGET != "all":
            configs.append((TARGET, "s2s", BACKEND_COMPILE))
        else:
            for target in S2S_TARGET_FRAMEWORKS:
                configs.append((target, "s2s", BACKEND_COMPILE))
    elif TARGET not in ["jax", "numpy", "tensorflow", "torch"]:
        for target in TARGET_FRAMEWORKS:
            configs.append((target, "transpile", BACKEND_COMPILE))
        configs.append(("torch", "trace", BACKEND_COMPILE))
    else:
        configs.append((TARGET, "transpile", BACKEND_COMPILE))
    metafunc.parametrize("target_framework,mode,backend_compile", configs)

import copy
import ivy
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import kornia
import numpy as np
import pytest
import tensorflow as tf
import torch

jax.config.update('jax_enable_x64', True)


# Helpers #
# ------- #

def _check_allclose(x, y, tolerance=1e-3):
    """
    Checks that all values are close. Any arrays must already be in numpy format, rather than native framework.
    """

    if isinstance(x, np.ndarray):
        assert np.allclose(x, y, atol=tolerance), "numpy array values are not all close"
        return

    if isinstance(x, (list, set, tuple)):
        all([
            _check_allclose(element_x, element_y, tolerance=tolerance) for element_x, element_y in zip(x, y)
        ])
        return

    if isinstance(x, dict):
        all([key_x == key_y for key_x, key_y in zip(x.keys(), y.keys())])
        all([
            _check_allclose(element_x, element_y, tolerance=tolerance)
            for element_x, element_y in zip(x.values(), y.values())
        ])
        return

    if isinstance(x, float):
        assert x - y < tolerance, f"float values differ: {x} != {y}"
        return

    assert x == y, f"values differ: {x} != {y}"


def _check_shape_allclose(x, y, tolerance=1e-3):
    """
    Checks that all array shapes are close. Any arrays must already be in numpy format, rather than native framework.
    """

    if isinstance(x, np.ndarray):
        assert np.allclose(x.shape, y.shape, atol=tolerance), "numpy array shapes are not all close"
        return

    if isinstance(x, (list, set, tuple)):
        all([
            _check_allclose(element_x, element_y, tolerance=tolerance) for element_x, element_y in zip(x, y)
        ])
        return

    if isinstance(x, dict):
        all([key_x == key_y for key_x, key_y in zip(x.keys(), y.keys())])
        all([
            _check_allclose(element_x, element_y, tolerance=tolerance)
            for element_x, element_y in zip(x.values(), y.values())
        ])
        return

    if isinstance(x, float):
        assert x - y < tolerance, f"float values differ: {x} != {y}"
        return

    assert x == y, f"values differ: {x} != {y}"


def _native_array_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    if isinstance(x, tf.Tensor):
        return x.numpy()
    if isinstance(x, jnp.ndarray):
        return np.asarray(x)
    if isinstance(x, nnx.Variable):
        return np.asarray(x.value)
    return x


def _nest_array_to_numpy(
    nest, shallow=True
):
    return ivy.nested_map(
        lambda x: _native_array_to_numpy(x),
        nest,
        include_derived=True,
        shallow=shallow,
    )


def _to_numpy_and_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_allclose(orig_data, transpiled_data, tolerance=tolerance) 


def _to_numpy_and_shape_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_shape_allclose(orig_data, transpiled_data, tolerance=tolerance) 


def _array_to_new_backend(
    x,
    target,
):
    """
    Converts a torch tensor to an array/tensor in a different framework.
    If the input is not a torch tensor, the input if returned without modification.
    """

    if isinstance(x, torch.Tensor):
        if target == "torch": return x
        y = x.detach().numpy()
        if target == "jax":
            y = jnp.array(y)
        elif target == "tensorflow":
            y = tf.convert_to_tensor(y)
        return y
    elif isinstance(x, torch.dtype):
        if target == "numpy":
            return np.dtype(str(x).split(".")[-1])
        elif target == "jax":
            return jnp.dtype(np.dtype(str(x).split(".")[-1]))
        elif target == "tensorflow":
            return tf.dtypes.as_dtype(np.dtype(str(x).split(".")[-1]))
        return x
    else:
        return x


def _nest_torch_tensor_to_new_framework(
    nest, target, shallow=True
):
    return ivy.nested_map(
        lambda x: _array_to_new_backend(x, target),
        nest,
        include_derived=True,
        shallow=shallow,
    )


def _backend_compile(obj, target):
    if target == "tensorflow":
        return tf.function(obj)
    elif target == "jax":
        return jax.jit(obj)
    return obj


def _target_to_simplified(target: str):
    """
    Convert the name of a target framework to its simplified form,
    such as 'tensorflow' -> 'tf'.
    """
    if target == "numpy":
        return "np"
    if target == "tensorflow":
        return "tf"
    if target == "jax":
        return "jax"
    if target == "torch":
        return "pt"
    return target


def _test_trace_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    backend_compile,
    tolerance=1e-3,
):
    graph = ivy.trace_graph(
        fn,
        to="torch",
        args=trace_args,
        kwargs=trace_kwargs,
        backend_compile=backend_compile,
        graph_caching=True,
    )

    graph_args = copy.deepcopy(test_args)
    graph_kwargs = copy.deepcopy(test_kwargs)

    orig_out = fn(*test_args, **test_kwargs)
    graph_out = graph(*graph_args, **graph_kwargs)

    orig_np = _nest_array_to_numpy(orig_out)
    graph_np = _nest_array_to_numpy(graph_out)

    _check_allclose(orig_np, graph_np, tolerance=tolerance)


def _test_transpile_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
):
    graph = ivy.transpile(
        fn,
        source="torch",
        to=target,
        args=trace_args,
        kwargs=trace_kwargs,
        backend_compile=backend_compile,
        graph_caching=True,
    )

    orig_out = fn(*test_args, **test_kwargs)
    graph_args = _nest_torch_tensor_to_new_framework(test_args, target)
    graph_kwargs = _nest_torch_tensor_to_new_framework(test_kwargs, target)
    graph_out = graph(*graph_args, **graph_kwargs)

    orig_np = _nest_array_to_numpy(orig_out)
    graph_np = _nest_array_to_numpy(graph_out)

    _check_allclose(orig_np, graph_np, tolerance=tolerance)


def _test_source_to_source_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
    deterministic=True,
):
    if backend_compile and target == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target)
    translated_fn = eval("transpiled_" + f"{fn.__module__}.{fn.__name__}")

    if backend_compile:
        try:
            fn = torch.compile(fn)
            fn(*trace_args, **trace_kwargs)
            orig_compilable = True
        except:
            orig_compilable = False

        # only test with backend compilation if the original function was compilable in torch
        if orig_compilable:
            translated_fn = _backend_compile(translated_fn, target)

    # test it works with the trace_args as input
    orig_out = fn(*trace_args, **trace_kwargs)
    graph_args = _nest_torch_tensor_to_new_framework(trace_args, target)
    graph_kwargs = _nest_torch_tensor_to_new_framework(trace_kwargs, target)
    graph_out = translated_fn(*graph_args, **graph_kwargs)

    if deterministic:
        _to_numpy_and_allclose(orig_out, graph_out, tolerance=tolerance)
    else:
        _to_numpy_and_shape_allclose(orig_out, graph_out, tolerance=tolerance)

    # test it works with the test_args as input
    orig_out = fn(*test_args, **test_kwargs)
    graph_args = _nest_torch_tensor_to_new_framework(test_args, target)
    graph_kwargs = _nest_torch_tensor_to_new_framework(test_kwargs, target)
    graph_out = translated_fn(*graph_args, **graph_kwargs)

    if deterministic:
        _to_numpy_and_allclose(orig_out, graph_out, tolerance=tolerance)
    else:
        _to_numpy_and_shape_allclose(orig_out, graph_out, tolerance=tolerance)


def _test_function(
    fn,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
    mode="transpile",
    skip=False,
    deterministic=True,
):
    # print out the full function module/name, so it will appear in the test_report.json
    print(f"{fn.__module__}.{fn.__name__}")

    if skip and mode != "s2s":
        # any skipped due to DCF issues should still work with ivy.source_to_source
        pytest.skip()

    if mode == "s2s":
        _test_source_to_source_function(
            fn,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            target,
            backend_compile,
            tolerance=tolerance,
            deterministic=deterministic,
        )
    elif mode == "trace":
        if target != "torch":
            pytest.skip()

        _test_trace_function(
            fn,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            backend_compile,
            tolerance=tolerance,
        )
    else:
        _test_transpile_function(
            fn,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            target,
            backend_compile,
            tolerance=tolerance,
        )

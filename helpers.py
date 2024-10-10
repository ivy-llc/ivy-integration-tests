import copy
import inspect
import gast
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
            _check_shape_allclose(element_x, element_y, tolerance=tolerance) for element_x, element_y in zip(x, y)
        ])
        return

    if isinstance(x, dict):
        all([key_x == key_y for key_x, key_y in zip(x.keys(), y.keys())])
        all([
            _check_shape_allclose(element_x, element_y, tolerance=tolerance)
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

def _get_fn_name_from_stack():
    # Get the previous calling stack frame
    stack = inspect.stack()
    caller_frame = stack[2]  # The function two levels above (eg: _test_function -> test_rgb_to_grayscale)
    
    source_code = inspect.getsource(caller_frame.frame)
    parsed_ast = gast.parse(source_code)
    
    # Traverse the AST to find the call to _test_function and extract the first argument
    for node in gast.walk(parsed_ast):
        if isinstance(node, gast.Call) and hasattr(node.func, 'id') and node.func.id == '_test_function':
            # Found the call to _test_function, extract the first argument (the fn)
            first_arg = node.args[0]  # This is the first argument passed to _test_function
            if isinstance(first_arg, gast.Attribute):
                # Get the full name of the function (e.g., "kornia.color.rgb_to_grayscale")
                return gast.unparse(first_arg).strip()
    return None

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
    fn_name,
    trace_args,
    trace_kwargs,
    test_args,
    test_kwargs,
    target,
    backend_compile,
    tolerance=1e-3,
    deterministic=True,
    class_info=None,
):
    if backend_compile and target == "numpy":
        pytest.skip()

    transpiled_kornia = ivy.transpile(kornia, source="torch", target=target)
    def transpile_and_instantiate(arg, arg_class_info=None):
        if arg_class_info:
            # If we have class info, transpile the class and instantiate it
            transpiled_class = ivy.transpile(arg_class_info['object'], source="torch", target=target)
            args = arg_class_info.get('args', ())
            kwargs = arg_class_info.get('kwargs', {})
            transpiled_args = _nest_torch_tensor_to_new_framework(args, target)
            transpiled_kwargs = _nest_torch_tensor_to_new_framework(kwargs, target)
            return transpiled_class(*transpiled_args, **transpiled_kwargs)
        else:
            # For other arguments, convert to the target framework
            return _nest_torch_tensor_to_new_framework(arg, target)

    if fn_name:
        translated_fn = eval("transpiled_" + f"{fn_name}")
    else:
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

    # Transpile and prepare trace arguments
    transpiled_trace_args = [
        transpile_and_instantiate(arg, class_info.get('trace_args', {}).get(i) if class_info else None)
        for i, arg in enumerate(trace_args)
    ]
    transpiled_trace_kwargs = {
        k: transpile_and_instantiate(v, class_info.get('trace_kwargs', {}).get(k) if class_info else None)
        for k, v in trace_kwargs.items()
    }

    # Transpile and prepare test arguments
    transpiled_test_args = [
        transpile_and_instantiate(arg, class_info.get('test_args', {}).get(i) if class_info else None)
        for i, arg in enumerate(test_args)
    ]
    transpiled_test_kwargs = {
        k: transpile_and_instantiate(v, class_info.get('test_kwargs', {}).get(k) if class_info else None)
        for k, v in test_kwargs.items()
    }

    if target == 'tensorflow':
        # build the model
        graph_out = translated_fn(*transpiled_trace_args, **transpiled_trace_kwargs)
    
    # sync models if needed
    [ivy.sync_models(m1, m2) for m1, m2 in zip(trace_args, transpiled_trace_args) if isinstance(m1, torch.nn.Module)]
    [ivy.sync_models(m1, m2) for m1, m2 in zip(trace_kwargs.values(), transpiled_trace_kwargs.values()) if isinstance(m1, torch.nn.Module)]
    [ivy.sync_models(m1, m2) for m1, m2 in zip(test_args, transpiled_test_args) if isinstance(m1, torch.nn.Module)]
    [ivy.sync_models(m1, m2) for m1, m2 in zip(test_kwargs.values(), transpiled_test_kwargs.values()) if isinstance(m1, torch.nn.Module)]

    # Test with trace_args
    orig_out = fn(*trace_args, **trace_kwargs)
    graph_out = translated_fn(*transpiled_trace_args, **transpiled_trace_kwargs)

     
    if deterministic:
        _to_numpy_and_allclose(orig_out, graph_out, tolerance=tolerance)
    else:
        _to_numpy_and_shape_allclose(orig_out, graph_out, tolerance=tolerance)

    # test it works with the test_args as input
    orig_out = fn(*test_args, **test_kwargs)
    graph_out = translated_fn(*transpiled_test_args, **transpiled_test_kwargs)

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
    class_info=None,
):
    # print out the full function module/name, so it will appear in the test_report.json
    print(f"{fn.__module__}.{fn.__name__}")
    fn_name = _get_fn_name_from_stack()
    if skip and mode != "s2s":
        # any skipped due to DCF issues should still work with ivy.source_to_source
        pytest.skip()

    if mode == "s2s":
        _test_source_to_source_function(
            fn,
            fn_name,
            trace_args,
            trace_kwargs,
            test_args,
            test_kwargs,
            target,
            backend_compile,
            tolerance=tolerance,
            deterministic=deterministic,
            class_info=class_info,

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

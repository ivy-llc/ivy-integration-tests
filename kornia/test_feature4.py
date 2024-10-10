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

def test_SIFTFeature(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTFeature")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledSIFTFeature = ivy.transpile(kornia.feature.SIFTFeature, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.SIFTFeature(num_features=10)
    torch_out = model(x)

    transpiled_model = TranspiledSIFTFeature(num_features=10)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_SIFTFeatureScaleSpace(target_framework, mode, backend_compile):
    print("kornia.feature.SIFTFeatureScaleSpace")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledSIFTFeatureScaleSpace = ivy.transpile(kornia.feature.SIFTFeatureScaleSpace, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.SIFTFeatureScaleSpace(num_features=10)
    torch_out = model(x)

    transpiled_model = TranspiledSIFTFeatureScaleSpace(num_features=10)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_GFTTAffNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.GFTTAffNetHardNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledGFTTAffNetHardNet = ivy.transpile(kornia.feature.GFTTAffNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)

    model = kornia.feature.GFTTAffNetHardNet(num_features=10)
    torch_out = model(x)

    transpiled_model = TranspiledGFTTAffNetHardNet(num_features=10)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetAffNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetAffNetHardNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledKeyNetAffNetHardNet = ivy.transpile(kornia.feature.KeyNetAffNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    
    model = kornia.feature.KeyNetAffNetHardNet(num_features=10)
    torch_out = model(x)

    transpiled_model = TranspiledKeyNetAffNetHardNet(num_features=10)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_KeyNetHardNet(target_framework, mode, backend_compile):
    print("kornia.feature.KeyNetHardNet")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledKeyNetHardNet = ivy.transpile(kornia.feature.KeyNetHardNet, source="torch", target=target_framework)

    x = torch.rand(1, 1, 256, 256)
    transpiled_x = _nest_torch_tensor_to_new_framework(x, target_framework)
    
    model = kornia.feature.KeyNetHardNet(num_features=10)
    torch_out = model(x)

    transpiled_model = TranspiledKeyNetHardNet(num_features=10)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_x)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_DescriptorMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.DescriptorMatcher")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledDescriptorMatcher = ivy.transpile(kornia.feature.DescriptorMatcher, source="torch", target=target_framework)

    x1 = torch.rand(2, 256)
    x2 = torch.rand(2, 256)
    transpiled_x1 = _nest_torch_tensor_to_new_framework(x1, target_framework)
    transpiled_x2 = _nest_torch_tensor_to_new_framework(x2, target_framework)

    model = kornia.feature.DescriptorMatcher('snn', 0.8)
    torch_out = model(x1, x2)

    transpiled_model = TranspiledDescriptorMatcher('snn', 0.8)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_x1, transpiled_x2)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_x1, transpiled_x2)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_GeometryAwareDescriptorMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.GeometryAwareDescriptorMatcher")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledGeometryAwareDescriptorMatcher = ivy.transpile(kornia.feature.GeometryAwareDescriptorMatcher, source="torch", target=target_framework)

    torch_args = (
        torch.rand(2, 256),
        torch.rand(2, 256),
        torch.rand(2, 2, 2, 3),
        torch.rand(2, 2, 2, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)

    model = kornia.feature.GeometryAwareDescriptorMatcher('fginn')
    torch_out = model(*torch_args)

    transpiled_model = TranspiledGeometryAwareDescriptorMatcher('fginn')
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(*transpiled_args)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(*transpiled_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LocalFeatureMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.LocalFeatureMatcher")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledGFTTAffNetHardNet = ivy.transpile(kornia.feature.GFTTAffNetHardNet, source="torch", target=target_framework)
    TranspiledDescriptorMatcher = ivy.transpile(kornia.feature.DescriptorMatcher, source="torch", target=target_framework)
    TranspiledLocalFeatureMatcher = ivy.transpile(kornia.feature.LocalFeatureMatcher, source="torch", target=target_framework)

    data = {
        "image0": torch.rand(1, 1, 320, 200),
        "image1": torch.rand(1, 1, 128, 128),
    }
    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)

    torch_local_feature = kornia.feature.GFTTAffNetHardNet(10)
    torch_matcher = kornia.feature.DescriptorMatcher('snn', 0.8)
    model = kornia.feature.LocalFeatureMatcher(torch_local_feature, torch_matcher)
    torch_out = model(data)

    transpiled_local_feature = TranspiledGFTTAffNetHardNet(10)
    transpiled_matcher = TranspiledDescriptorMatcher('snn', 0.8)
    transpiled_model = TranspiledLocalFeatureMatcher(transpiled_local_feature, transpiled_matcher)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_data)
    
    ivy.sync_models(model, transpiled_model)

    transpiled_out = transpiled_model(transpiled_data)

    _to_numpy_and_shape_allclose(torch_out, transpiled_out)


def test_LightGlueMatcher(target_framework, mode, backend_compile):
    print("kornia.feature.LightGlueMatcher")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledLightGlueMatcher = ivy.transpile(kornia.feature.LightGlueMatcher, source="torch", target=target_framework)

    torch_args = (
        torch.rand(2, 128),
        torch.rand(5, 128),
        torch.rand(1, 2, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target_framework)
    
    model = kornia.feature.LightGlueMatcher('disk')
    torch_out = model(*torch_args)

    transpiled_model = TranspiledLightGlueMatcher('disk')
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(*transpiled_args)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(*transpiled_args)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LightGlue(target_framework, mode, backend_compile):
    print("kornia.feature.LightGlue")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledLightGlue = ivy.transpile(kornia.feature.LightGlue, source="torch", target=target_framework)

    data = {
        "image0": {
            "keypoints": torch.rand(1, 100, 2),
            "descriptors": torch.rand(1, 100, 256),
            "image_size": torch.tensor([[640, 480]]),
        },
        "image1": {
            "keypoints": torch.rand(1, 120, 2),
            "descriptors": torch.rand(1, 120, 256),
            "image_size": torch.tensor([[640, 480]]),
        }
    }
    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)
    
    model = kornia.feature.LightGlue(features='superpoint')
    torch_out = model(data)

    transpiled_model = TranspiledLightGlue(features='superpoint')
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_data)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_data)

    _to_numpy_and_allclose(torch_out, transpiled_out)



def test_LoFTR(target_framework, mode, backend_compile):
    print("kornia.feature.LoFTR")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledLoFTR = ivy.transpile(kornia.feature.LoFTR, source="torch", target=target_framework)

    data = {"image0": torch.rand(1, 1, 320, 200), "image1": torch.rand(1, 1, 128, 128)}
    transpiled_data = _nest_torch_tensor_to_new_framework(data, target_framework)
    
    model = kornia.feature.LoFTR(None)
    torch_out = model(data)

    transpiled_model = TranspiledLoFTR(None)
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_data)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_data)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PassLAF(target_framework, mode, backend_compile):
    print("kornia.feature.PassLAF")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledPassLAF = ivy.transpile(kornia.feature.PassLAF, source="torch", target=target_framework)

    laf = torch.rand(1, 2, 3, 3)
    img = torch.rand(1, 3, 32, 32)
    torch_out = kornia.feature.PassLAF()(laf, img)

    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    transpiled_out = TranspiledPassLAF()(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_PatchAffineShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.PatchAffineShapeEstimator")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledPatchAffineShapeEstimator = ivy.transpile(kornia.feature.PatchAffineShapeEstimator, source="torch", target=target_framework)

    patch = torch.rand(1, 1, 19, 19)
    transpiled_patch = _nest_torch_tensor_to_new_framework(patch, target_framework)
    
    model = kornia.feature.PatchAffineShapeEstimator()
    torch_out = model(patch)

    transpiled_model = TranspiledPatchAffineShapeEstimator()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_patch)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_patch)

    _to_numpy_and_allclose(torch_out, transpiled_out)


def test_LAFAffineShapeEstimator(target_framework, mode, backend_compile):
    print("kornia.feature.LAFAffineShapeEstimator")

    if backend_compile or target_framework == "numpy":
        pytest.skip()

    TranspiledLAFAffineShapeEstimator = ivy.transpile(kornia.feature.LAFAffineShapeEstimator, source="torch", target=target_framework)

    laf = torch.rand(1, 2, 2, 3)
    img = torch.rand(1, 1, 32, 32)
    transpiled_laf = _nest_torch_tensor_to_new_framework(laf, target_framework)
    transpiled_img = _nest_torch_tensor_to_new_framework(img, target_framework)
    
    model = kornia.feature.LAFAffineShapeEstimator()
    torch_out = model(laf, img)

    transpiled_model = TranspiledLAFAffineShapeEstimator()
    if target_framework == "tensorflow":
        # build the layers 
        transpiled_model(transpiled_laf, transpiled_img)
    
    ivy.sync_models(model, transpiled_model)
    
    transpiled_out = transpiled_model(transpiled_laf, transpiled_img)

    _to_numpy_and_allclose(torch_out, transpiled_out)

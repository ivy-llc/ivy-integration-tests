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


# Helpers #
# ------- #

def _to_numpy_and_shape_allclose(torch_x, transpiled_x, tolerance=1e-3):
    orig_data = _nest_array_to_numpy(torch_x)
    transpiled_data = _nest_array_to_numpy(transpiled_x)
    _check_shape_allclose(orig_data, transpiled_data, tolerance=tolerance) 


# Tests #
# ----- #

def test_RANSAC(target_framework, mode, backend_compile):
    print("kornia.geometry.ransac.RANSAC")

    if backend_compile:
        pytest.skip()

    # Initialize RANSAC with default parameters
    ransac = kornia.geometry.ransac.RANSAC(model_type='homography')

    # Transpile the RANSAC class to the target framework
    TranspiledRANSAC = ivy.transpile(kornia.geometry.ransac.RANSAC, source="torch", target=target_framework)

    # Prepare synthetic keypoints data for source and destination images
    kp1 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    kp2 = torch.tensor([[0.0, 0.0], [1.1, 1.1], [2.0, 2.1], [3.0, 3.1]], requires_grad=True)

    transpiled_kp1 = _nest_torch_tensor_to_new_framework(kp1, target_framework)
    transpiled_kp2 = _nest_torch_tensor_to_new_framework(kp2, target_framework)

    # Run RANSAC on the original data
    torch_model, torch_inliers = ransac(kp1, kp2)

    # Run transpiled RANSAC on the transpiled data
    transpiled_ransac = TranspiledRANSAC(model_type='homography')
    transpiled_model, transpiled_inliers = transpiled_ransac(transpiled_kp1, transpiled_kp2)

    # Ensure that the estimated models are close to each other
    _to_numpy_and_shape_allclose(torch_model, transpiled_model)

    # Ensure that the inlier masks are consistent
    _to_numpy_and_shape_allclose(torch_inliers, transpiled_inliers)

    # Test RANSAC with custom parameters
    ransac_custom = kornia.geometry.ransac.RANSAC(model_type='homography', inl_th=1.5, max_iter=20, confidence=0.95)
    TranspiledRANSACCustom = ivy.transpile(kornia.geometry.ransac.RANSAC, source="torch", target=target_framework)
    transpiled_ransac_custom = TranspiledRANSACCustom(model_type='homography', inl_th=1.5, max_iter=20, confidence=0.95)

    torch_model_custom, torch_inliers_custom = ransac_custom(kp1, kp2)
    transpiled_model_custom, transpiled_inliers_custom = transpiled_ransac_custom(transpiled_kp1, transpiled_kp2)

    _to_numpy_and_shape_allclose(torch_model_custom, transpiled_model_custom)
    _to_numpy_and_shape_allclose(torch_inliers_custom, transpiled_inliers_custom)

    # Test RANSAC on a different model type (e.g., fundamental matrix)
    ransac_fundamental = kornia.geometry.ransac.RANSAC(model_type='fundamental')
    TranspiledRANSACFundamental = ivy.transpile(kornia.geometry.ransac.RANSAC, source="torch", target=target_framework)
    transpiled_ransac_fundamental = TranspiledRANSACFundamental(model_type='fundamental')

    kp1 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    kp2 = torch.tensor([[0.0, 0.0], [1.1, 1.1], [2.0, 2.1], [3.0, 3.1], [0.0, 0.0], [1.1, 1.1], [2.0, 2.1], [3.0, 3.1]], requires_grad=True)
    transpiled_kp1 = _nest_torch_tensor_to_new_framework(kp1, target_framework)
    transpiled_kp2 = _nest_torch_tensor_to_new_framework(kp2, target_framework)

    torch_model_fundamental, torch_inliers_fundamental = ransac_fundamental(kp1, kp2)
    transpiled_model_fundamental, transpiled_inliers_fundamental = transpiled_ransac_fundamental(transpiled_kp1, transpiled_kp2)

    _to_numpy_and_shape_allclose(torch_model_fundamental, transpiled_model_fundamental)
    _to_numpy_and_shape_allclose(torch_inliers_fundamental, transpiled_inliers_fundamental)

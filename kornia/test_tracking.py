from helpers import (
    _nest_torch_tensor_to_new_framework,
    _to_numpy_and_allclose,
)

import ivy
import kornia
import pytest
import torch


# Helpers #
# ------- #


def _create_synthetic_homography_image(image, H, size):
    """Create a synthetically warped image based on a homography."""
    return kornia.geometry.transform.warp_perspective(image, H, size)


# Tests #
# ----- #

def test_HomographyTracker(target_framework, mode, backend_compile):
    print("kornia.tracking.HomographyTracker")

    if backend_compile:
        pytest.skip()

    TranspiledHomographyTracker = ivy.transpile(kornia.tracking.HomographyTracker, source="torch", target=target_framework)

    tracker = kornia.tracking.HomographyTracker()
    transpiled_tracker = TranspiledHomographyTracker()

    target_image = torch.rand(1, 1, 240, 320, requires_grad=True)

    # Define a known homography transformation (slight rotation + translation)
    H = torch.tensor([[[0.98, -0.17, 20.0],
                       [0.17,  0.98, 10.0],
                       [0.0,   0.0,  1.0]]])
    next_frame = _create_synthetic_homography_image(target_image, H, (240, 320))

    transpiled_target_image = _nest_torch_tensor_to_new_framework(target_image, target_framework)
    transpiled_next_frame = _nest_torch_tensor_to_new_framework(next_frame, target_framework)

    tracker.set_target(target_image)
    transpiled_tracker.set_target(transpiled_target_image)

    # Test match_initial method
    torch_initial_homography, torch_initial_match = tracker.match_initial(next_frame)
    transpiled_initial_homography, transpiled_initial_match = transpiled_tracker.match_initial(transpiled_next_frame)

    _to_numpy_and_allclose(torch_initial_homography, transpiled_initial_homography)
    assert torch_initial_match == transpiled_initial_match

    # Test forward method - when no previous homography is set
    torch_forward_homography, torch_forward_match = tracker.forward(next_frame)
    transpiled_forward_homography, transpiled_forward_match = transpiled_tracker.forward(transpiled_next_frame)

    _to_numpy_and_allclose(torch_forward_homography, transpiled_forward_homography)
    assert torch_forward_match == transpiled_forward_match

    # Set a previous homography to the tracker
    tracker.previous_homography = torch.eye(3)
    transpiled_tracker.previous_homography = _nest_torch_tensor_to_new_framework(torch.eye(3), target_framework)

    # Test track_next_frame method
    torch_next_frame_homography, torch_next_frame_match = tracker.track_next_frame(next_frame)
    transpiled_next_frame_homography, transpiled_next_frame_match = transpiled_tracker.track_next_frame(transpiled_next_frame)

    _to_numpy_and_allclose(torch_next_frame_homography, transpiled_next_frame_homography)
    assert torch_next_frame_match == transpiled_next_frame_match

    # Test forward method - when a previous homography is set
    torch_forward_homography_with_previous, torch_forward_match_with_previous = tracker.forward(next_frame)
    transpiled_forward_homography_with_previous, transpiled_forward_match_with_previous = transpiled_tracker.forward(transpiled_next_frame)

    _to_numpy_and_allclose(torch_forward_homography_with_previous, transpiled_forward_homography_with_previous)
    assert torch_forward_match_with_previous == transpiled_forward_match_with_previous

    # Test reset_tracking method
    tracker.reset_tracking()
    transpiled_tracker.reset_tracking()

    assert tracker.previous_homography is None
    assert transpiled_tracker.previous_homography is None

    # Test no_match method
    torch_no_match_homography, torch_no_match_flag = tracker.no_match()
    transpiled_no_match_homography, transpiled_no_match_flag = transpiled_tracker.no_match()

    _to_numpy_and_allclose(torch_no_match_homography, transpiled_no_match_homography)
    assert torch_no_match_flag == transpiled_no_match_flag

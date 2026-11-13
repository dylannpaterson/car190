import numpy as np

from joint_car190_led_model import JointCAR190LED


def test_response_jacobians_match_finite_differences():
    model = JointCAR190LED.__new__(JointCAR190LED)
    model.mode_count = 2
    wave = np.array([410.0, 530.0, 680.0])
    ice = 1.46 + 0.012j
    y0 = 1.0 + 0.0j
    clean_y = np.array([0.72 + 0.12j, 0.66 + 0.18j, 0.58 + 0.22j])
    reflection = (y0 - clean_y) / (y0 + clean_y)
    item = {
        "wave": wave,
        "ice": ice,
        "y0": y0,
        "first": np.array([1.0 + 0.02j, 0.98 - 0.03j, 1.01 + 0.01j]),
        "clean_y": clean_y,
        "reflection": reflection,
        "basis": np.array([[1.0, 0.3, -0.5], [0.1, 0.8, 0.4]]),
        "factor": np.array([1.0, 1.2, 0.8]),
    }
    thickness = np.array([20.0, 28.0])
    theta = np.array([0.2, -0.15, 0.3, 0.25])
    log_rate, theta_jac, thickness_jac = model._response(theta, item, thickness)
    h = 1e-5

    numerical_theta = np.column_stack(
        [
            (
                model._response(theta + np.eye(len(theta))[j] * h, item, thickness)[0]
                - model._response(theta - np.eye(len(theta))[j] * h, item, thickness)[0]
            )
            / (2 * h)
            for j in range(len(theta))
        ]
    )
    numerical_thickness = np.array(
        [
            (
                model._response(theta, item, thickness + np.eye(len(thickness))[j] * h)[0]
                - model._response(theta, item, thickness - np.eye(len(thickness))[j] * h)[0]
            )[j]
            / (2 * h)
            for j in range(len(thickness))
        ]
    )

    assert np.isfinite(log_rate).all()
    np.testing.assert_allclose(theta_jac, numerical_theta, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(thickness_jac, numerical_thickness, rtol=2e-6, atol=2e-8)


def test_joint_likelihood_gradient_includes_reference_normalization():
    model = JointCAR190LED.__new__(JointCAR190LED)
    model.mode_count = 2
    model.times = np.array([0.0, 1.0, 2.0])
    wave = np.array([410.0, 530.0, 680.0])
    ice = 1.46 + 0.012j
    y0 = 1.0 + 0.0j
    clean_y = np.array([0.72 + 0.12j, 0.66 + 0.18j, 0.58 + 0.22j])
    item = {
        "tile": 0,
        "time_index": np.array([0, 1, 2]),
        "wave": wave,
        "ice": ice,
        "y0": y0,
        "first": np.array([1.0 + 0.02j, 0.98 - 0.03j, 1.01 + 0.01j]),
        "clean_y": clean_y,
        "reflection": (y0 - clean_y) / (y0 + clean_y),
        "basis": np.array([[1.0, 0.3, -0.5], [0.1, 0.8, 0.4]]),
        "factor": np.array([1.0, 1.2, 0.8]),
        "thickness": np.array([20.0, 28.0, 34.0]),
        "saved": np.array([0.01, -0.02, 0.03]),
        "observed": np.array([-0.04, 0.02, 0.01]),
        "sigma": np.array([0.3, 0.4, 0.5]),
    }
    item["baseline_log_rate"] = model._response(
        np.zeros(4), item, item["thickness"]
    )[0]
    model.bands = [[item], [], [], [], [], []]
    theta = np.array([0.2, -0.15, 0.3, 0.25])
    thickness = np.full((256, 3), 25.0)
    thickness[0] = item["thickness"] + np.array([-1.0, 0.5, 2.0])
    residual, theta_gradient, depth_gradient = model.evaluate(theta, thickness)
    h = 1e-5

    def loss(parameters, depths):
        values = model.evaluate(parameters, depths, with_gradient=False)[0]
        return 0.5 * values @ values

    numerical_theta = np.array(
        [
            (
                loss(theta + np.eye(len(theta))[j] * h, thickness)
                - loss(theta - np.eye(len(theta))[j] * h, thickness)
            )
            / (2 * h)
            for j in range(len(theta))
        ]
    )
    numerical_depth = np.array(
        [
            (
                loss(theta, thickness + np.eye(3)[j][None, :] * np.eye(256)[0][:, None] * h)
                - loss(theta, thickness - np.eye(3)[j][None, :] * np.eye(256)[0][:, None] * h)
            )
            / (2 * h)
            for j in range(3)
        ]
    )

    assert residual.shape == (3,)
    np.testing.assert_allclose(theta_gradient, numerical_theta, rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(depth_gradient[0], numerical_depth, rtol=3e-6, atol=3e-8)
    np.testing.assert_allclose(depth_gradient[1:], 0.0, atol=1e-12)

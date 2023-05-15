__doc__ = """ Test common functions used in contact in Elastica.joint implementation"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from elastica.joint import (
    _dot_product,
    _norm,
    _clip,
    _out_of_bounds,
    _find_min_dist,
    _aabbs_not_intersecting,
)


class TestDotProduct:
    "class to test the dot product function"

    @pytest.mark.parametrize("ndim", [3])
    def test_dot_product_using_numpy(self, ndim):
        """
        This method was generated by "copilot" in VS code;
        This method uses numpy dot product to compare with the output of our function,
        numpy dot product uses an optimized implementation that takes advantage of
        hardware-specific optimizations such as SIMD.
        """
        vector1 = np.random.randn(ndim)
        vector2 = np.random.randn(ndim)
        dot_product = _dot_product(vector1, vector2)
        assert_allclose(dot_product, np.dot(vector1, vector2))

    def test_dot_product_with_verified_values(self):
        "Testing function with analytically verified values"

        "test for parallel vectors"
        vec1 = [1, 0, 0]
        vec2 = [2, 0, 0]
        dot_product = _dot_product(vec1, vec2)
        assert_allclose(dot_product, 2)

        "test for perpendicular vectors"
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        dot_product = _dot_product(vec1, vec2)
        assert_allclose(dot_product, 0)

        "test for opposite vectors"
        vec1 = [1, 0, 0]
        vec2 = [-1, 0, 0]
        dot_product = _dot_product(vec1, vec2)
        assert_allclose(dot_product, -1)

        "test for complex vectors"
        vec1 = [1, -2, 3]
        vec2 = [-2, 1, 3]
        dot_product = _dot_product(vec1, vec2)
        assert_allclose(dot_product, 5)


class TestNorm:
    "class to test the _norm function"

    def test_norm_with_verified_values(self):
        "Testing function with analytically verified values"

        "test for null vector"
        vec1 = [0, 0, 0]
        norm = _norm(vec1)
        assert_allclose(norm, 0)

        "test for unit vector"
        vec1 = [1, 1, 1]
        norm = _norm(vec1)
        assert_allclose(norm, 1.7320508075688772)

        "test for arbitrary natural number vector"
        vec1 = [1, 2, 3]
        norm = _norm(vec1)
        assert_allclose(norm, 3.7416573867739413)

        "test for decimal values vector"
        vec1 = [0.001, 0.002, 0.003]
        norm = _norm(vec1)
        assert_allclose(norm, 0.0037416573867739412)


class TestClip:
    "class to test the _clip function"

    @pytest.mark.parametrize(
        "x, result",
        [(0.5, 1), (1.5, 1.5), (2.5, 2)],
    )
    def test_norm_with_verified_values(self, x, result):
        "Testing function with analytically verified values"

        low = 1.0
        high = 2.0
        assert _clip(x, low, high) == result


class TestOutOfBounds:
    "class to test the _out_of_bounds function"

    @pytest.mark.parametrize(
        "x, result",
        [(0.5, 1), (1.5, 0), (2.5, 1)],
    )
    def test_out_of_bounds_with_verified_values(self, x, result):
        "testing function with analytically verified values"

        low = 1.0
        high = 2.0
        assert _out_of_bounds(x, low, high) == result


class TestFindMinDist:
    "class to test the _find_min_dist function"

    def test_find_min_dist(self):
        "testing function with analytically verified values"

        "intersecting lines"
        x1 = np.array([0, 0, 0])
        e1 = np.array([1, 1, 1])
        x2 = np.array([0, 1, 0])
        e2 = np.array([1, 0, 1])
        min_dist_vec, closestpointofline1, closestpointofline2 = _find_min_dist(
            x1, e1, x2, e2
        )
        assert_allclose(min_dist_vec, [0, 0, 0])
        assert_allclose(closestpointofline1, [1, 1, 1])
        assert_allclose(closestpointofline2, [-1, -1, -1])

        "arbitrary close lines"
        tol = 1e-5
        x1 = np.array([0, 0, 0])
        e1 = np.array([1, 1, 1])
        x2 = np.array([0, 0, 1])
        e2 = np.array([1.5, 2, 0])
        min_dist_vec, closestpointofline1, closestpointofline2 = _find_min_dist(
            x1, e1, x2, e2
        )
        assert_allclose(
            min_dist_vec, [-0.153846, 0.115385, 0.038462], rtol=tol, atol=tol
        )
        assert_allclose(
            closestpointofline1, [0.807692, 1.076923, 1], rtol=tol, atol=tol
        )
        assert_allclose(
            closestpointofline2, [-0.961538, -0.961538, -0.961538], rtol=tol, atol=tol
        )

        "arbitrary away lines"
        x1 = np.array([7, 0, 3.5])
        e1 = np.array([1, 0, 0])
        x2 = np.array([0, 70, 0])
        e2 = np.array([0, 0, 11.5])
        min_dist_vec, closestpointofline1, closestpointofline2 = _find_min_dist(
            x1, e1, x2, e2
        )
        assert_allclose(min_dist_vec, [-7, 70, 0])
        assert_allclose(closestpointofline1, [0, 70, 3.5])
        assert_allclose(closestpointofline2, [7, 0, 3.5])


class TestAABBSNotIntersecting:
    "class to test the _aabb_intersecting function"

    def test_aabbs_not_intersectin(self):
        "testing function with analytically verified values"

        " intersecting boxes"
        aabb_one = np.array([[0, 0], [0, 0], [0, 0]])
        aabb_two = np.array([[0, 0], [0, 0], [0, 0]])
        assert _aabbs_not_intersecting(aabb_one, aabb_two) == 0

        " non Intersecting boxes"
        aabb_one = np.array([[0, 1], [0, 1], [0, 1]])
        aabb_two = np.array([[2, 3], [2, 3], [2, 3]])
        assert _aabbs_not_intersecting(aabb_one, aabb_two) == 1
import os
import time
import platform
import itertools
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable
from matplotlib.patches import Circle

from verify_student_output import OutputCheck
from detect_circles_student import detectCircles

im = plt.imread("data/im1.jpg")

if platform.system() == "Windows":
    NIX = False
    print("Running on Windows system")
else:
    NIX = True
    print("Running on Linux/OS X system")


def print_success_message(test_case):
    print("UnitTest {0} passed successfully!".format(test_case))


class PS04Test(OutputCheck):
    def testHoughTransform(
        self, detectCircles: Callable, center: np.ndarray, radius: int, identifier: str
    ) -> bool:
        """
        Test the hough transform implementation to detect circles
        by setting the useGradient flag to False
        """
        print("Testing circle detector on {0} circle".format(identifier))
        useGrad = False
        thresh = 0.95
        circles, _ = detectCircles(im, radius, thresh, useGrad)

        r, c = circles.shape
        self.assertEqual(c, 3, msg="Returned numpy array should have dimensions K x 3")
        self.assertLessEqual(r, 10, msg="More than 10 centers returned")

        circles = circles[:, :-1]
        check_flag = False
        check_out = np.empty((0, 2))
        for i0, i1 in itertools.product(
            np.arange(center.shape[0]), np.arange(circles.shape[0])
        ):
            if np.all(np.isclose(center[i0], circles[i1], atol=8)):
                check_out = np.concatenate((check_out, [circles[i1]]), axis=0)

        if check_out.shape[0] > 0:
            check_flag = True
            print_success_message("testHoughTransform")

        return check_flag

    def testHoughTransformGradient(
        self, detectCircles: Callable, center: np.ndarray, radius: int, identifier: str
    ) -> bool:
        """
        Test the hough transform implementation to detect circles
        by setting the useGradient flag to True
        """
        print("Testing circle detector on {0} circle".format(identifier))
        useGrad = True
        thresh = 0.95
        circles, _ = detectCircles(im, radius, thresh, useGrad)

        r, c = circles.shape
        self.assertEqual(c, 3, msg="Returned numpy array should have dimensions K x 3")
        self.assertLessEqual(r, 10, msg="More than 10 centers returned")

        circles = circles[:, :-1]
        check_flag = False
        check_out = np.empty((0, 2))
        for i0, i1 in itertools.product(
            np.arange(center.shape[0]), np.arange(circles.shape[0])
        ):
            if np.all(np.isclose(center[i0], circles[i1], atol=8)):
                check_out = np.concatenate((check_out, [circles[i1]]), axis=0)

        if check_out.shape[0] > 0:
            check_flag = True
            print_success_message("testHoughTransformGradient")
        return check_flag

    def testDetectCircles(self, detectCircles: Callable, useGrad: bool = False) -> None:
        centers = np.array([[210, 655], [660, 137], [538, 517], [234, 288],])
        radii = [75, 90, 100, 143]
        identifiers = ["Red", "Pink", "Green", "Blue"]

        n_circles = 4

        # Check implementation with and without gradient
        check_flags = []
        for i in range(len(radii)):
            center = centers[i, :]
            center = np.expand_dims(center, axis=0)
            rad = radii[i]
            id_circ = identifiers[i]
            if useGrad:
                res = self.testHoughTransformGradient(
                    detectCircles, center, rad, id_circ
                )
            else:
                res = self.testHoughTransform(detectCircles, center, rad, id_circ)

            check_flags.append(res)

        n_passed = sum(check_flags)
        if n_passed == n_circles:
            print_success_message(
                "testDetectCircles (useGradient set to {})".format(str(useGrad))
            )

        self.assertTrue(
            n_passed == n_circles,
            "Only {0} out of {1} circles were detected successfully (useGradient set to False)".format(
                n_passed, n_circles
            ),
        )

        print("*" * 50)

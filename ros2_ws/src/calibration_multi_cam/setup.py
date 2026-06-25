import os
from glob import glob

from setuptools import find_packages, setup

package_name = "calibration_multi_cam"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="Multi-camera intrinsic + extrinsic calibration for ROS 2 (AprilGrid, kalibr-style).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "intrinsic_calibrator_node = calibration_multi_cam.intrinsic_calibrator_node:main",
            "extrinsic_calibrator_node = calibration_multi_cam.extrinsic_calibrator_node:main",
            "publisher_node = calibration_multi_cam.publisher_node:main",
        ],
    },
)

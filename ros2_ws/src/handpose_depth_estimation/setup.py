import os
from glob import glob

from setuptools import find_packages, setup

package_name = "handpose_depth_estimation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="Per-joint 3D hand-pose estimation: triangulate all 21 MediaPipe "
    "hand landmarks from two selected cameras and reproject for QA.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "handpose_depth_node = "
            "handpose_depth_estimation.handpose_depth_node:main",
        ],
    },
)

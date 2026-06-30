import os
from glob import glob

from setuptools import find_packages, setup

package_name = "single_cam_pose_estimation"

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
    description="Monocular 6-DoF hand-pose estimation per camera: PnP on "
    "MediaPipe's hand-local model -> 21 world joints, 6-DoF pose, and "
    "reprojection QA.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "single_cam_pose_node = "
            "single_cam_pose_estimation.single_cam_pose_node:main",
        ],
    },
)

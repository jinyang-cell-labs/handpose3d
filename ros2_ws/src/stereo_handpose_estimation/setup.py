import os
from glob import glob

from setuptools import find_packages, setup

package_name = "stereo_handpose_estimation"

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
    description="Stereo hand-pose estimation from MediaPipe landmark messages: "
    "triangulate the hand centroid and place hand_world_landmarks in 3D.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stereo_handpose_node = "
            "stereo_handpose_estimation.stereo_handpose_node:main",
        ],
    },
)

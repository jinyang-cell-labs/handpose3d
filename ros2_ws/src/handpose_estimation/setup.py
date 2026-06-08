import os
from glob import glob

from setuptools import find_packages, setup

package_name = "handpose_estimation"

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
        (os.path.join("share", package_name, "models"), glob("models/*.task")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="3D hand pose estimation from two calibrated cameras via MediaPipe + DLT.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "handpose_node = handpose_estimation.handpose_node:main",
        ],
    },
)

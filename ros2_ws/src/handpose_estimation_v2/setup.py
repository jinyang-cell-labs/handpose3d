import os
from glob import glob

from setuptools import find_packages, setup

package_name = "handpose_estimation_v2"

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
    description=(
        "Model-based 3D hand pose estimation: direct multi-view rigid "
        "template fit on 2D reprojection error (replaces triangulation + "
        "Procrustes of handpose_estimation)."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "handpose_node_v2 = handpose_estimation_v2.handpose_node:main",
        ],
    },
)

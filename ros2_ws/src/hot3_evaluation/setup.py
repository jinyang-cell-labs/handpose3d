import os
from glob import glob

from setuptools import find_packages, setup

package_name = "hot3_evaluation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="Fisheye stereo depth evaluation for the HOT3D SLAM camera pair.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stereo_depth_eval_node = "
            "hot3_evaluation.stereo_depth_eval_node:main",
        ],
    },
)

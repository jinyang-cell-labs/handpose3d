import os
from glob import glob

from setuptools import find_packages, setup

package_name = "multi_cam_stream"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"),
         glob("config/*.yaml") + glob("config/*.rviz")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="Minimal multi-USB-camera frontend publishing image_raw (bgr8) per camera.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_stream_node = multi_cam_stream.camera_stream_node:main",
        ],
    },
)

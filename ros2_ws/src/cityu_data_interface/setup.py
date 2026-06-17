import os
from glob import glob

from setuptools import find_packages, setup

package_name = "cityu_data_interface"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "config/camera_info"),
            glob("config/camera_info/*.yaml"),
        ),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jinyang",
    maintainer_email="jinyang@cell-labs.ai",
    description="Replays the CityU stereo hand pose dataset as ROS 2 camera topics.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cityu_data_publisher_node = "
            "cityu_data_interface.cityu_data_publisher_node:main",
        ],
    },
)

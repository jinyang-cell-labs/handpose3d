import os
from glob import glob

from setuptools import find_packages, setup

package_name = "gui_service_call"

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
    description="Tkinter GUI to trigger ROS 2 services as buttons and show "
    "their responses; config-driven via services.yaml.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "service_caller_node = gui_service_call.service_caller_node:main",
        ],
    },
)

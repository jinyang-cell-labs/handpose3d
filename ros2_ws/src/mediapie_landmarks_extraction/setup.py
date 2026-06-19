import os
from glob import glob

from setuptools import find_packages, setup

package_name = "mediapie_landmarks_extraction"

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
    description="Basic MediaPipe hand-landmark extraction: subscribe to image "
    "topics, annotate with 21 2D landmarks and republish.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "landmarks_node = mediapie_landmarks_extraction.landmarks_node:main",
        ],
    },
)

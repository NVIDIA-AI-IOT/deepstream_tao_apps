"""
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//build:ds_gxf.bzl", "ds_gxf_sample_app")

ds_gxf_sample_app(
    name = "faciallandmark-translator",
    graph = "App-FacialLandmark.yaml",
    parameters = [
        "App-FacialLandmark.parameters.yaml",
    ],
)

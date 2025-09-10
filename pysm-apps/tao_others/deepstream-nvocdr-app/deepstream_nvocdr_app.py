# Copyright and license information
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe, osd
from pyservicemaker.logging import get_logger
from pyservicemaker.flow import RenderMode
from multiprocessing import Process
import os, sys

sys.path.append("../../")
from common.utils import flow_cls_hook, dump_config

logger = get_logger("deepstream_nvocdr_app")
logger.setLevel("DEBUG")

# alias flow_cls_hook as overlay
make_link_element = flow_cls_hook

def deepstream_nvocdr_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)

    pipeline = Pipeline("deepstream_nvocdr_app")
    # add Monkey patch for osd
    Flow.make_link_element = make_link_element  # type: ignore
    flow = Flow(pipeline)
    flow = flow.batch_capture(
        conf.stream_list, width=conf.streammux_width, height=conf.streammux_height
    )

    # Add nvdsvideotemplate to the pipeline
    list_props = conf.video_template_customlib_props
    vt_properties = {
        "customlib_name": conf.video_template_customlib_name
    }
    type_names = []
    properties = []
    type_names.append("nvdsvideotemplate")
    properties.append(vt_properties)
    flow = flow.make_link_element(type_names, properties)
    for prop in list_props:
        my_dict = {"customlib-props": prop}
        flow.pipeline[flow._streams[0]].set(my_dict)
  
    # Add OSD overlay to the pipeline for encoder can output bbox and mask
    osd_properties = {}
    flow = flow.make_link_element(["nvdsosd"], [osd_properties])

    if conf.output_type == 1:
        filename=conf.output_filename+".mp4"
        flow = flow.encode(dest=filename, use_sw_codec=conf.output_enc_type, bitrate=conf.output_bitrate)
    elif conf.output_type == 2:
        flow = flow.render(mode=RenderMode.DISCARD, enable_osd=False, sync=True)
    elif conf.output_type == 3:
        flow = flow.render(enable_osd=False)
    else:
        flow = flow.render(mode=RenderMode.DISCARD, enable_osd=False, sync=True)
    # Execute the pipeline
    flow()


if __name__ == "__main__":
    # Check input arguments
    if len(sys.argv) != 2:
        logger.error(f"usage: {sys.argv[0]} <yaml> ")
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_nvocdr_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

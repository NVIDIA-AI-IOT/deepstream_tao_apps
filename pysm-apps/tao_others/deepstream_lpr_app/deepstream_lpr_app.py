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

logger = get_logger("deepstream_lpr_app")
logger.setLevel("DEBUG")

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            # TODO: missing binding of obj_meta.mask_params
            text = f"Frame Number={frame_meta.frame_number}, Object Count={len(list(frame_meta.object_items))}"
            logger.debug(f"Object Counter: Pad Idx={frame_meta.pad_index}, {text}")

            display_meta = batch_meta.acquire_display_meta()
            label = osd.Text()
            label.display_text = text.encode("ascii")
            label.x_offset = 10
            label.y_offset = 12
            label.font.name = osd.FontFamily.Serif
            label.font.size = 12
            label.font.color = osd.Color(1.0, 1.0, 1.0, 1.0)
            label.set_bg_color = True
            label.bg_color = osd.Color(0.0, 0.0, 0.0, 1.0)
            display_meta.add_text(label)
            frame_meta.append(display_meta)


# alias flow_cls_hook as overlay
make_link_element = flow_cls_hook


def deepstream_lpr_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)

    pipeline = Pipeline("deepstream_lpr_app")
    # add Monkey patch for osd
    Flow.make_link_element = make_link_element
    flow = Flow(pipeline)
    flow = flow.batch_capture(
        conf.stream_list, width=1280, height=720
    )
    flow = flow.infer(conf.pgie_infer_config, with_triton=conf.pgie_infer_type)
    flow = flow.attach(what=Probe("counter", ObjectCounterMarker()))
    flow = flow.attach(what="measure_fps_probe", name="fps_probe")
    flow = flow.track(ll_config_file=conf.tracker_ll_cfg_file, ll_lib_file=conf.tracker_ll_lib_file)
    flow = flow.infer(conf.sgie_0_infer_config, with_triton=conf.sgie_0_infer_type)
    flow = flow.infer(conf.sgie_1_infer_config, with_triton=conf.sgie_1_infer_type)
    # Add OSD overlay to the pipeline for encoder can output bbox and mask
    type_names = []
    properties = []
    if conf.analytics:
        analytics_properties = {
            "config-file": conf.analytics_config_file
        }
        type_names.append("nvdsanalytics")
        properties.append(analytics_properties)
    type_names.append("nvdsosd")
    osd_properties = {}
    properties.append(osd_properties)
    flow = flow.make_link_element(type_names, properties)

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
    process = Process(target=deepstream_lpr_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

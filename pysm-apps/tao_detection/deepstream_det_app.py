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
import os, sys, math
import numpy as np
import cv2

sys.path.append("../")
from common.utils import flow_cls_hook, dump_config

logger = get_logger("tao_det_app")
logger.setLevel("DEBUG")
show_mask = os.environ.get("SHOW_MASK", 0)


class ObjectCounterMarker(BatchMetadataOperator):
    def _save_instance_mask(self, img_path, ins_mask, target_width, target_height):
        print(f"instance mask : {ins_mask.shape} {ins_mask.dtype}")
        mask_uint8 = (ins_mask * 255).astype(np.uint8)

        mask_image = cv2.resize(
            mask_uint8,
            (target_width, target_height),
            dst=mask_uint8,
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.imwrite(img_path, mask_image)  # Save mask to image

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            text = f"Frame Number={frame_meta.frame_number}, Object Count={len(list(frame_meta.object_items))}"
            logger.debug(f"Object Counter: Pad Idx={frame_meta.pad_index}, {text}")
            is_first_object = True
            if show_mask and frame_meta.frame_number % 100 == 0:
                for object_meta in frame_meta.object_items:
                    if is_first_object:
                        print(
                            f"Object Rect {object_meta.rect_params} Mask {object_meta.mask_params}"
                        )
                        is_first_object = False
                        ins_mask = object_meta.mask_params.mask_array
                        img_path = f"frame-{frame_meta.frame_number}-obj-mask.jpg"
                        self._save_instance_mask(
                            img_path,
                            ins_mask,
                            math.floor(object_meta.rect_params.width),
                            math.floor(object_meta.rect_params.height),
                        )

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
overlay = flow_cls_hook


def deepstream_det_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)
    osd_properties = {}
    if show_mask:
        osd_properties = {
            "display-mask": True,
            "display-bbox": False,
            "display-text": False,
            "process-mode": 0,
        }
    pipeline = Pipeline("deepstream_det_app")
    # add Monkey patch for osd
    Flow.overlay = overlay  # type: ignore
    flow = Flow(pipeline)
    flow = flow.batch_capture(
        conf.stream_list, width=conf.streammux_width, height=conf.streammux_height
    )
    flow = flow.infer(conf.pgie_infer_config, with_triton=conf.pgie_infer_type)
    flow = flow.attach(what=Probe("counter", ObjectCounterMarker()))
    flow = flow.attach(what="measure_fps_probe", name="fps_probe")
    # Add OSD overlay to the pipeline for encoder can output bbox and mask
    type_names = []
    properties = []
    type_names.append("nvdsosd")
    properties.append(osd_properties)
    flow = flow.overlay(type_names, properties)

    # Fork the pipeline for parallel processing
    # This allows simultaneous video rendering and encode to files
    flow = flow.fork()

    if conf.eglsink:
        flow = flow.render(enable_osd=False)
    if conf.filesink:
        flow = flow.encode(dest="pysm_det_output.mp4", use_sw_codec=conf.enc_type)
    if conf.fakesink:
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
    process = Process(target=deepstream_det_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

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

# Import required libraries
from pyservicemaker import Pipeline, Flow, BatchMetadataOperator, Probe
from pyservicemaker.logging import get_logger
from pyservicemaker.flow import RenderMode
from multiprocessing import Process
import numpy as np
import cv2

import sys

sys.path.append("../")
from common.utils import flow_cls_hook, dump_config

logger = get_logger("tao_seg_app")
logger.setLevel("DEBUG")

COLORS = np.array(
    [
        [128, 128, 64],
        [0, 0, 128],
        [0, 128, 128],
        [128, 0, 0],
        [128, 0, 128],
        [128, 128, 0],
        [0, 128, 0],
        [0, 0, 64],
        [0, 0, 192],
        [0, 128, 64],
        [0, 128, 192],
        [128, 0, 64],
        [128, 0, 192],
        [128, 128, 128],
        [128, 64, 128],
        [128, 64, 0],
        [0, 64, 128],
        [192, 128, 0],
        [192, 128, 64],
    ],
    dtype=np.uint8,
)


class SegmentationMaskReceiver(BatchMetadataOperator):
    def _map_mask_as_display_bgr(self, mask):
        """
        Assigning multiple colors as image output using the information
        contained in mask. (BGR is OpenCV standard.)
        """
        # get unique class indices from the mask
        unique_classes = np.unique(mask)
        # create BGR image with zeros
        bgr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # use numpy advanced indexing to assign colors
        for idx in unique_classes:
            bgr[mask == idx] = COLORS[idx]

        return bgr

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            logger.debug(f"Frame Number={frame_meta.frame_number}")
            for seg_meta_it in frame_meta.segmentation_items:
                if frame_meta.frame_number % 100 == 0:
                    seg_meta = seg_meta_it.as_segmentation()
                    masks = seg_meta.class_map
                    masks = np.array(masks, copy=True, order='C')
                    masks=masks.reshape(seg_meta.height, seg_meta.width)
                    print(f"type(masks)={type(masks)}, shape={masks.shape}")
                    frame_image = self._map_mask_as_display_bgr(masks)
                    cv2.imwrite(f"frame-{frame_meta.frame_number}.jpg", frame_image)


# alias flow_cls_hook as segvisual
segvisual = flow_cls_hook


def deepstream_seg_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)
    pipeline = Pipeline("deepstream_seg_app")
    # add Monkey patch for segvisual
    Flow.segvisual = segvisual  # type: ignore
    flow = Flow(pipeline)
    flow = flow.batch_capture(
        conf.stream_list, width=conf.streammux_width, height=conf.streammux_height
    )
    flow = flow.infer(conf.pgie_infer_config, with_triton=conf.pgie_infer_type)

    #add segvisual element
    type_names = []
    properties = []
    type_names.append("nvsegvisual")
    properties.append(conf.segvisual)
    flow = flow.segvisual(type_names, properties)

    flow = flow.attach(what=Probe("segmentation_mask", SegmentationMaskReceiver()))
    flow = flow.attach(what="measure_fps_probe", name="fps_probe")
    # Fork the pipeline for parallel processing
    # This allows simultaneous video rendering and encode to files
    flow = flow.fork()

    if conf.eglsink:
        flow = flow.render(enable_osd=False)
    if conf.filesink:
        flow = flow.encode(dest="pysm_seg_output.mp4", use_sw_codec=conf.enc_type)
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
    process = Process(target=deepstream_seg_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

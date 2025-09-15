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
from enum import Enum
import math

sys.path.append("../../")
from common.utils import flow_cls_hook, dump_config

logger = get_logger("deepstream_pose_classification_app")
logger.setLevel("DEBUG")

class Sink_Type(Enum):
    fakesink = 0
    filesink = 1
    rtspsink = 2
    displaysink = 3
#Padding due to AR SDK model requires bigger bboxes
PAD_DIM=128
_pad_dim=PAD_DIM
MUXER_OUTPUT_WIDTH=1280
MUXER_OUTPUT_HEIGHT=720
_image_width=MUXER_OUTPUT_WIDTH
_image_height=MUXER_OUTPUT_HEIGHT

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        # Padding due to AR SDK model requires bigger bounding boxes
        muxer_output_width_pad = _pad_dim * 2 + _image_width
        muxer_output_height_pad = _pad_dim * 2 + _image_height

        for frame_meta in batch_meta.frame_items:
            for obj_meta in frame_meta.object_items:
                sizex = obj_meta.rect_params.width * 0.5
                sizey = obj_meta.rect_params.height * 0.5
                centrx = obj_meta.rect_params.left + sizex
                centry = obj_meta.rect_params.top + sizey
                sizex *= 1.25
                sizey *= 1.25
                if sizex < sizey:
                    sizex = sizey
                else:
                    sizey = sizex
                obj_meta.rect_params.width = round(2.0 * sizex)
                obj_meta.rect_params.height = round(2.0 * sizey)
                obj_meta.rect_params.left = round(centrx - obj_meta.rect_params.width / 2.0)
                obj_meta.rect_params.top = round(centry - obj_meta.rect_params.height / 2.0)

                sizex = obj_meta.rect_params.width * 0.5
                sizey = obj_meta.rect_params.height * 0.5
                centrx = obj_meta.rect_params.left + sizex
                centry = obj_meta.rect_params.top + sizey

                x_scale = 192.0 / sizex
                y_scale = 256.0 / sizey

                if x_scale < y_scale:
                    sizey = 256.0 / x_scale  # Expand height
                else:
                    sizex = 192.0 / y_scale  # Expand width

                obj_meta.rect_params.width = round(2.0 * sizex)
                obj_meta.rect_params.height = round(2.0 * sizey)
                obj_meta.rect_params.left = round(centrx - obj_meta.rect_params.width / 2.0)
                obj_meta.rect_params.top = round(centry - obj_meta.rect_params.height / 2.0)

                if obj_meta.rect_params.left < 0.0:
                    obj_meta.rect_params.left = 0.0
                if obj_meta.rect_params.top < 0.0:
                    obj_meta.rect_params.top = 0.0
                if obj_meta.rect_params.left + obj_meta.rect_params.width > muxer_output_width_pad - 1:
                    obj_meta.rect_params.width = muxer_output_width_pad - 1 - obj_meta.rect_params.left
                if obj_meta.rect_params.top + obj_meta.rect_params.height > muxer_output_height_pad - 1:
                    obj_meta.rect_params.height = muxer_output_height_pad - 1 - obj_meta.rect_params.top


# alias flow_cls_hook as overlay
make_link_element = flow_cls_hook

def sink_out(flow, sink_type, enc_name="test.mp4", enc_type=1, enc_bitrate=2000000):
    print("sink_out", sink_type, enc_name, enc_type, enc_bitrate)
    if sink_type == Sink_Type.fakesink.value:
        flow = flow.render(mode=RenderMode.DISCARD, enable_osd=False, sync=True)
    elif sink_type == Sink_Type.filesink.value:
        flow = flow.encode(dest=enc_name, use_sw_codec=enc_type, bitrate=enc_bitrate)
    elif sink_type == Sink_Type.rtspsink.value:
        flow = flow.make_link_element(["nvrtspoutsinkbin"], [{"enc-type": enc_type}])
    elif sink_type == Sink_Type.displaysink.value:
        flow = flow.render(enable_osd=False)

def deepstream_pose_classification_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)

    pipeline = Pipeline("deepstream_pose_classification_app")
    # add Monkey patch for osd
    Flow.make_link_element = make_link_element
    flow = Flow(pipeline)

    #the pipeline is ......->nvvideoconvert->capsfilter->pgie->tracker->sgie0->nvdspostprocess->
    #->nvdspreprocess->
    flow = flow.batch_capture(
        conf.stream_list, width=1280, height=720
    )

    _pad_dim = PAD_DIM * _image_width / MUXER_OUTPUT_WIDTH
    muxer_output_width_pad = int(_pad_dim * 2 + _image_width)
    muxer_output_height_pad = int(_pad_dim * 2 + _image_height)
    flow = flow.make_link_element(["nvvideoconvert", "capsfilter"],
        [{
        "dest-crop":  f"{_pad_dim}:{_pad_dim}:{_image_width}:{_image_height}",
        "interpolation-method":  1},
         {"caps": f"video/x-raw(memory:NVMM),width={muxer_output_width_pad},height={muxer_output_height_pad}"}
        ]
    )

    flow = flow.infer(conf.pgie_infer_config, with_triton=conf.pgie_infer_type)
    flow = flow.track(ll_config_file=conf.tracker_ll_cfg_file, ll_lib_file=conf.tracker_ll_lib_file)
    flow = flow.attach(what=Probe("counter", ObjectCounterMarker()))
    flow = flow.infer(conf.sgie_0_infer_config, with_triton=conf.sgie_0_infer_type)
    print(conf.postprocess_0_config_file_path, conf.postprocess_0_lib_name)
    flow = flow.make_link_element(["nvdspreprocess", "nvdspostprocess"],
        [{"config-file":  conf.preprocess_1_config_file_path},
         {"postprocesslib-config-file": conf.postprocess_0_config_file_path,
            "postprocesslib-name": conf.postprocess_0_lib_name}]
    )
    flow = flow.infer(conf.sgie_1_infer_config, with_triton=conf.sgie_1_infer_type)

    source_num = len(conf.stream_list)
    tiler_rows = int(math.sqrt(source_num))
    tiler_columns = int(math.ceil(source_num / tiler_rows))
    flow = flow.make_link_element(
        ["nvdslogger", "nvvideoconvert", "capsfilter", "nvmultistreamtiler", "nvvideoconvert", "nvdsosd"],
        [{"fps-measurement-interval-sec":  1},
         { "src-crop":  f"video/x-raw(memory:NVMM),width={muxer_output_width_pad},height={muxer_output_height_pad}"},
         { "caps":  f"video/x-raw(memory:NVMM),width={_image_width},height={_image_height}"},
         { "rows": tiler_rows, "columns": tiler_columns, "width": MUXER_OUTPUT_WIDTH, "height": MUXER_OUTPUT_HEIGHT},
         {},
         {"display-mask": False, "display-bbox": True, "display-text": True, "process-mode": 0},
        ]
    )

    #sink
    sink_out(flow, conf.sinkType, "out.mp4", conf.sinkEncType)

    # Execute the pipeline
    flow()


if __name__ == "__main__":
    # Check input arguments
    if len(sys.argv) != 2:
        logger.error(f"usage: {sys.argv[0]} <yaml> ")
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_pose_classification_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

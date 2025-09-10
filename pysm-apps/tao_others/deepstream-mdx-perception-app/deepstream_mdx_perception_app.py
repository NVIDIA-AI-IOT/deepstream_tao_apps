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
import torch
from enum import Enum

sys.path.append("../../")
from common.utils import flow_cls_hook, dump_config

logger = get_logger("deepstream_mdx_perception_app")
logger.setLevel("DEBUG")

class Sink_Type(Enum):
    fakesink = 1
    displaysink = 2
    filesink = 3
    msgbrokersink = 6

class ObjectCounterMarker(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            for object_meta in frame_meta.object_items:
                for user_meta in object_meta.tensor_items:
                    output_layers=user_meta.as_tensor_output().get_layers()
                    layer_name="embeddings"
                    vector_data = output_layers.pop(layer_name, None)
                    embedding_vector = torch.utils.dlpack.from_dlpack(vector_data).to('cpu')
                    print(embedding_vector)


# alias flow_cls_hook as overlay
make_link_element = flow_cls_hook

def sink_out(flow,  properties: list):
    print("properties", properties)
    sink_type=properties["type"]
    if sink_type == Sink_Type.fakesink.value:
        flow = flow.render(mode=RenderMode.DISCARD, enable_osd=False, sync=True)
    elif sink_type == Sink_Type.displaysink.value:
        flow = flow.render(enable_osd=False)
    elif sink_type == Sink_Type.filesink.value:
        flow = flow.encode(dest=properties["name"], use_sw_codec=properties["encType"], bitrate=properties["bitrate"])
    elif sink_type == Sink_Type.msgbrokersink.value:
        # Configure message publishing to Kafka
        # Sends processed data to external systems
        flow.publish(
                msg_broker_proto_lib=properties["msg_broker_proto_lib"],
                msg_broker_conn_str=properties["msg_broker_conn_str"],
                topic=properties["msg_topic"],
                msg_conv_config=properties["msg_conv_config"],
                sync=False,  # Asynchronous publishing for better performance
            )

def deepstream_mdx_perception_app(yaml_config_path):
    conf = dump_config(yaml_config_path)
    if conf is None:
        logger.error(f"Failed to load configuration from {yaml_config_path}")
        sys.exit(1)

    pipeline = Pipeline("deepstream_mdx_perception_app")
    # add Monkey patch for osd
    Flow.make_link_element = make_link_element
    flow = Flow(pipeline)

    flow = flow.batch_capture(
        conf.source_cvs, width=1280, height=720
    )
    flow = flow.infer(conf.pgie_config_file, with_triton=conf.pgie_infer_type)
    flow = flow.track(ll_config_file=conf.tracker_ll_cfg_file, ll_lib_file=conf.tracker_ll_lib_file)
    flow = flow.infer(conf.sgie_0_config_file, with_triton=conf.sgie_0_infer_type)
    flow = flow.attach(what=Probe("counter", ObjectCounterMarker()))
    # Message metadata generation probe - prepares data for Kafka publishing
    flow = flow.attach(
            what="add_message_meta_probe",
            name="message_generator"
        )
    # Add OSD overlay to the pipeline for encoder can output bbox and mask
    osd_properties = {
        "display-mask": False,
        "display-bbox": True,
        "display-text": True,
        "process-mode": 0,
    }
    flow = flow.make_link_element(["nvdsosd"], [osd_properties])

    # Fork the pipeline for parallel processing
    # This allows simultaneous video rendering and encode to files
    flow = flow.fork()

    #sink0
    if conf.sink_0_enable == 1:
        sink_properties = {
            "type": conf.sink_0_type,
        }
        sink_out(flow, sink_properties)
    #sink1
    if conf.sink_1_enable == 1:
        sink_properties = {
            "type": conf.sink_1_type,
            "name": conf.sink_1_filename,
            "encType": conf.sink_1_encType,
            "bitrate": conf.sink_1_bitrate,
        }
        sink_out(flow, sink_properties)
    #sink2
    if conf.sink_2_enable == 1:
        sink_properties = {
            "type": conf.sink_2_type,
            "msg_conv_config": conf.sink_2_msg_conv_config,
            "msg_broker_proto_lib": conf.sink_2_msg_broker_proto_lib,
            "msg_broker_conn_str": conf.sink_2_msg_broker_conn_str,
            "msg_topic": conf.sink_2_msg_topic,
        }
        sink_out(flow, sink_properties)

    # Execute the pipeline
    flow()


if __name__ == "__main__":
    # Check input arguments
    if len(sys.argv) != 2:
        logger.error(f"usage: {sys.argv[0]} <yaml> ")
        sys.exit(1)

    # Flow()() is a blocking call due to which the KeyboardInterrupt may not be processed immediately.
    # we use Process from multiprocessing which runs the main function in a different process and processes KeyboardInterrupt immediately.
    process = Process(target=deepstream_mdx_perception_app, args=(sys.argv[1],))
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        logger.debug("\nCtrl+C detected. Terminating process...")
        process.terminate()

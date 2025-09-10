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

import os, yaml
from pyservicemaker import Flow
from pyservicemaker.logging import get_logger
from typing import Dict, Optional, Any
from cuda import cuda
import subprocess

logger = get_logger("tao_seg_app")
logger.setLevel("DEBUG")


class Config:
    def __init__(self, config_path, config):
        self._config_path = config_path
        self._config = config

    @property
    def config(self):
        return self._config

    def __getattr__(self, name: str) -> Any:
        match name:
            case "stream_list":
                return self._config["source-list"]["list"].split(";")
            case "source_cvs":
                cvs_config = self._config["source"]["csv-file-path"]
                yaml_directory = os.path.dirname(os.path.abspath(self._config_path))
                cvs_abs_path = os.path.join(yaml_directory, cvs_config)
                print("cvs_abs_path;", cvs_abs_path)
                urls = []
                with open(cvs_abs_path, "r") as file:
                    for line in file:
                        items = line.split(",")
                        if items[2] == "uri":
                            continue
                        urls.append(items[2])
                print(urls)
                return urls
            case "streammux_width":
                return self._config["streammux"]["width"]
            case "streammux_height":
                return self._config["streammux"]["height"]
            case "pgie_infer_type":
                return self._config["primary-gie"]["plugin-type"]
            case "pgie_infer_config":
                infer_config = self._config["primary-gie"]["config-file-path"]
                yaml_directory = os.path.dirname(os.path.abspath(self._config_path))
                return os.path.join(yaml_directory, infer_config)
            case "pgie_config_file":
                infer_config = self._config["primary-gie"]["config-file"]
                yaml_directory = os.path.dirname(os.path.abspath(self._config_path))
                return os.path.join(yaml_directory, infer_config)
            case str() if name.startswith("sgie") and "_" in name:
                print(name.split("_"))
                index = int(name.split("_")[1]) # get sgie index
                sgie_key = f"secondary-gie{index}"
                print(f"sgie_key: {sgie_key}")
                match name:
                    case _ if "infer_type" in name:
                        return self._config[sgie_key]["plugin-type"]
                    case _ if "infer_config" in name:
                        infer_config = self._config[sgie_key]["config-file-path"]
                        yaml_directory = os.path.dirname(
                            os.path.abspath(self._config_path)
                        )
                        return os.path.join(yaml_directory, infer_config)
                    case _ if "config_file" in name:
                        infer_config = self._config[sgie_key]["config-file"]
                        yaml_directory = os.path.dirname(
                            os.path.abspath(self._config_path)
                        )
                        return os.path.join(yaml_directory, infer_config)
            case "eglsink":
                return self._config["eglsink"].get("enable") or False
            case "fakesink":
                return self._config["fakesink"].get("enable") or False
            case "filesink":
                return self._config["filesink"].get("enable") or False
            case "enc_type":
                # 0 for hardware encoding, 1 for software encoding
                # if the hardware doesn't supports encoding, force return 1
                if is_enc_hw_support() is False:
                    return 1
                return self._config["filesink"].get("enc-type") or 0
                #use filesink by default

            #preprocess
            case str() if name.startswith("preprocess") and "_" in name:
                print(name.split("_"))
                index = int(name.split("_")[1]) # get sgie index
                sgie_key = f"secondary-preprocess{index}"
                print(f"preprocess_key: {sgie_key}")
                match name:
                    case _ if "config_file_path" in name:
                        infer_config = self._config[sgie_key]["config-file-path"]
                        yaml_directory = os.path.dirname(
                            os.path.abspath(self._config_path)
                        )
                        return os.path.join(yaml_directory, infer_config)

            #postprocess
            case str() if name.startswith("postprocess") and "_" in name:
                print(name.split("_"))
                index = int(name.split("_")[1]) # get sgie index
                sgie_key = f"secondary-postprocess{index}"
                print(f"postprocess_key: {sgie_key}")
                match name:
                    case _ if "config_file_path" in name:
                        tmp = self._config[sgie_key]["config-file-path"]
                        yaml_directory = os.path.dirname(
                            os.path.abspath(self._config_path)
                        )
                        return os.path.join(yaml_directory, tmp)
                    case _ if "lib_name" in name:
                        tmp = self._config[sgie_key]["lib-name"]
                        yaml_directory = os.path.dirname(
                            os.path.abspath(self._config_path)
                        )
                        return os.path.join(yaml_directory, tmp)

            #sink
            case str() if name.startswith("sink") and "_" in name:
                index = int(name.split("_")[1]) # get sgie index
                print("index:", index)
                sgie_key = f"sink{index}"
                print(f"sgie_key: {sgie_key}")
                match name:
                    case _ if "enable" in name:
                        return self._config[sgie_key].get("enable")  or 0
                    case _ if "type" in name:
                        return self._config[sgie_key].get("type")  or 1
                    case _ if "filename" in name:
                        return self._config[sgie_key].get("output-file")  or "test"
                    case _ if "encType" in name:
                        return self._config[sgie_key].get("enc-type")  or 0
                    case _ if "bitrate" in name:
                        return self._config[sgie_key].get("bitrate")  or 2000000
                    case _ if "msg_conv_config" in name:
                        return self._config[sgie_key].get("msg-conv-config")  or \
                        "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test5/configs/dstest5_msgconv_sample_config.txt"
                    case _ if "msg_broker_proto_lib" in name:
                        return self._config[sgie_key].get("msg-broker-proto-lib")  or \
                        "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
                    case _ if "msg_broker_conn_str" in name:
                        return self._config[sgie_key].get("msg-broker-conn-str")  or \
                        "127.0.0.1;9092"
                    case _ if "msg_topic" in name:
                        return self._config[sgie_key].get("topic")  or \
                        "test"

            case "sinkType":
                return self._config["sink"].get("sink-type") or 1
            case "sinkEncType":
                return self._config["sink"].get("enc-type") or 0
            #output
            case "output_type":
                return self._config["output"].get("type") or 1
            case "output_enc_type":
                return self._config["output"].get("enc-type") or 0
            case "output_filename":
                return self._config["output"].get("filename") or "test"
            case "output_bitrate":
                return self._config["output"].get("bitrate") or 2000000
            case "output_codec":
                return self._config["output"].get("codec") or 0

            #tracker
            case "tracker_ll_cfg_file":
                return self._config["tracker"].get("ll-config-file")
            case "tracker_ll_lib_file":
                return self._config["tracker"].get("ll-lib-file")

            #analytics
            case "analytics":
                return self._config["analytics"].get("enable") or False
            case "analytics_config_file":
                config = self._config["analytics"]["config-file"]
                yaml_directory = os.path.dirname(os.path.abspath(self._config_path))
                return os.path.join(yaml_directory, config)

            #segvisual
            case "segvisual":
                data = self._config["segvisual"]
                segvisual_width = data["width"]
                segvisual_height = data["height"]
                segvisual_orig_background = data.get("orig_background") or False
                segvisual_alpha = data.get("alpha") or 0.5
                return {
                    "width": segvisual_width,
                    "height": segvisual_height,
                    "original-background": segvisual_orig_background,
                    "alpha": float(segvisual_alpha),
                }

            #video_template
            case "video_template_customlib_name":
                return self._config["video-template"].get("customlib-name")
            case "video_template_customlib_props":
                return self._config["video-template"].get("customlib-props")
            case _:
                if name in self._config:
                    return self._config[name]
                else:
                    raise AttributeError(f"Config has no attribute '{name}'")


def dump_config(yaml_config_path: str) -> Optional[Config]:
    """
    Load configuration from a YAML file.
    Args:
        config_file (str): Path to the YAML configuration file.
    Returns:
        Config: An instance of Config containing the loaded configuration, or None if loading fails.
    """
    try:
        with open(yaml_config_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return Config(yaml_config_path, yaml_data)
    except FileNotFoundError:
        logger.error(f"file not found: {yaml_config_path}")
        return None
    except PermissionError:
        logger.error(f"no permission: {yaml_config_path}")
        return None
    except Exception as e:
        logger.error(f"unknown error : {e}")
        raise e

def get_node_name(func, name) -> str:
    if not hasattr(func, "name_counter"):
        func.name_counter = 0
    else:
        func.name_counter += 1
    base_name = func.__name__.split('.')[-1]
    return f"{base_name}-{name}-{func.name_counter}"

def flow_cls_hook(self, type_names: list, properties: list) -> "Flow":
    """
    Dynamic add method to Flow class. For adding a new element to the pipeline
    Args:
        type_names(list(str)): list of element name
        properties(list(dict)): list of property dict
    Return: A derived flow
    Raises: Upstream Exception
    """
    last_name=""
    for index, name in enumerate(type_names):
        element_name = get_node_name(flow_cls_hook, name)
        self._pipeline.add(name, element_name, properties[index])
        self._pipeline.link(self._streams[0], element_name)
        self._streams=[element_name]
        last_name=element_name
    return Flow(self._pipeline, streams=[last_name], parent=self)


def is_enc_hw_support():
    enc_hw_support = True

    process = subprocess.Popen(['gst-inspect-1.0', 'nvv4l2h264enc'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("enc_hw is not supported")
        enc_hw_support = False

    return enc_hw_support

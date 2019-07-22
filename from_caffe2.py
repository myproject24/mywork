# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Caffe2 Models
=====================
**Author**: `Hiroyuki Makino <https://makihiro.github.io/>`_

This article is an introductory tutorial to deploy Caffe2 models with Relay.

For us to begin with, Caffe2 should be installed.

A quick solution is to install via conda

.. code-block:: bash

    # for cpu
    conda install pytorch-nightly-cpu -c pytorch
    # for gpu with CUDA 8
    conda install pytorch-nightly cuda80 -c pytorch

or please refer to official site
https://caffe2.ai/docs/getting-started.html
"""

######################################################################
# Load pretrained Caffe2 model
# ----------------------------
# We load a pretrained resnet50 classification model provided by Caffe2.
import os
import re
import sys
from tvm import rpc
from tvm.contrib import util, xcode
from caffe2.python.models.download import ModelDownloader
mf = ModelDownloader()

class Model:
    def __init__(self, model_name):
        self.init_net, self.predict_net, self.value_info = mf.get_c2_model(model_name)

resnet50 = Model('resnet50')

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from tvm.contrib.download import download_testdata
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()
# input preprocess
def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :].astype('float32')
    return image

data = transform_image(img)

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

if not re.match(r"^platform=.*,id=.*$", destination):
    print("Bad format: {}".format(destination))
    print("Example of expected string: platform=iOS,id=1234567890abcabcabcabc1234567890abcabcab")
    sys.exit(1)

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target_host = "llvm -target=%s-apple-darwin" % arch

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src):
    return xcode.compile_metal(src, sdk=sdk)

######################################################################
# Compile the model on Relay
# --------------------------

# Caffe2 input tensor name, shape and type
input_name = resnet50.predict_net.op[0].input[0]
shape_dict = {input_name: data.shape}
dtype_dict = {input_name: data.dtype}

# parse Caffe2 model and convert into Relay computation graph
from tvm import relay
mod, params = relay.frontend.from_caffe2(resnet50.init_net, resnet50.predict_net, shape_dict, dtype_dict)

# compile the model
target = 'metal'
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, target_host=target_host, params=params)

#Save the library 
temp = util.tempdir()
path_dso1 = temp.relpath("dev_lib.dylib")
lib.export_library(path_dso1, xcode.create_dylib,
				 arch=arch, sdk=sdk)
xcode.codesign(path_dso1)

# Start RPC test server that contains the compiled library.
server = xcode.popen_test_rpc(proxy_host, proxy_port, key,
							  destination=destination,
							  libs=[path_dso1])

# connect to the proxy
remote = rpc.connect(proxy_host, proxy_port, key=key)
ctx = remote.metal(0)
load_lib = remote.load_module("dev_lib.dylib")
module = graph_runtime.create(graph, loaded_lib, ctx) # This line is thrwoing error.
module.load_params(loaded_params)
caffe2_out = module.run(data=input_data)

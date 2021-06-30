import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
config = tf.compat.v1.ConfigProto()
if tf.test.is_gpu_available():
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.visible_device_list = "1"
tf.compat.v1.Session(config=config)


print("python {}".format(sys.version))
print("tensorflow version {}".format(tf.__version__))
import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import uff

UFF_OUTPUT_FILENAME = 'model_tensorrt.uff'

#OUTPUT_NAMES = ["output_names"]

with tf.Session() as persisted_sess:
  print("load graph")
  with gfile.FastGFile("../rpg_public_dronet/model/model_tensorflow.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter("./tf_summary", graph=persisted_sess.graph)
    # Print all operation names
    #for op in persisted_sess.graph.get_operations():
    #  print(op)



import tensorrt as trt
from tensorrt.parsers import uffparser

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)


# Load your newly created Tensorflow frozen model and convert it to UFF
uff_model = uff.from_tensorflow_frozen_model("../rpg_public_dronet/model/model_tensorflow.pb", ["activation_8/Sigmoid", "dense_1/BiasAdd"], output_filename=UFF_OUTPUT_FILENAME)
uff.from_tensorflow(graphdef=frozen_graph,
                        output_filename=UFF_OUTPUT_FILENAME,
                        output_nodes=OUTPUT_NAMES,
                        text=True)



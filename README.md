# Dronet implementation on TX2

See full project description in the project report .doc file

## Software Installation:

### Keras Only

•	Jetson TX2 installed using Jetpack 3.3 (already includes CUDA, CUDNN drivers)  
•	Install python 2 with Numpy (optional - matplotlib)  
•	Install Tensorflow + Keras with CUDA support  
•	Install ROS Kinetic (desktop version)  
&nbsp;&nbsp;o	Install ROS and setup the workspace according to the dronet installation notes  
&nbsp;&nbsp;o	Get this project's source and overwrite the dronet_perception component  
&nbsp;&nbsp;o	Compile ROS packages: dronet_perception, dronet_control and bebop_autonomy from dronet github  

### TensorRT

On host PC:  
•	Install Python and Keras with Tensorflow backend. CUDA is required, but a GPU is not.  
•	Jetpack 3.3  
•	Get the PC-component project source code.  
•	Using the scripts included, translate the Keras HDF5 model to a Tensorflow PB model, using the included application:
`python keras_to_tensorflow.py --input_model="path/to/keras/model.h5" --input_model_json="path/to/keras/model.json" --output_model="path/to/save/model.pb"`  
•	Translate the Tensorflow PB model to UFF model: `python pb_to_uff.py`  
If a different model than the default is used, the contents of the script should be altered to match it.  

On Jetson TX2:  
•	Jetson TX2 installed using Jetpack 3.3 (already includes CUDA, CUDNN drivers and the TensorRT software)  
•	Install Python 2 with numpy (option- matploblib)  
•	Install ROS Kinetic (desktop version)  
&nbsp;&nbsp;o	Install ROS and setup the workspace according to the dronet installation notes  
&nbsp;&nbsp;o	Get this project's source and copy over the dronet_perception_trt package  
&nbsp;&nbsp;o	Compile ROS packages: dronet_perception_trt, dronet_control and bebop_autonomy  
•	Get the project TensorRT inference library and:  
&nbsp;&nbsp;o	Compile the TensorRT inference library (trtinference) by running `make solibs` in the base source directory.  
&nbsp;&nbsp;o	Setup the paths for the trtinference.h and the newly created libtrtinference.so in the CMakeLists file.  

Alternatively, flash a cloned image with the entire project/prerequisites already installed on the TX2 EMMC. This requires Jetpack to be installed on the host.

## Running

### Keras Only

•	Launch dronet_bebop.launch, dronet_launch.launch (from dronet_perception) and deep_navigation.launch (in this order). This will connect to the drone and stream the video feed into the neural network, displaying its predictions. The prediction will then be picked up by the control block that can send commands to the drone (OFF by default)  
•	To issue the drone to start to fly, send the command  
`rostopic pub --once /bebop/takeoff std_msgs/Empty}`  
WARNING: THE DRONE WILL NOW START TO HOVER  
•	To issue the autonomous navigation, send the command  
`rostopic pub --once /bebop/state_change std_msgs/Bool "data: true"`  
WARNING: THE DRONE WILL NOW SELF NAVIGATE  
•	To stop the autonomous navigation, send the command `rostopic pub --once /bebop/state_change std_msgs/Bool "data: false"`  
•	To order the drone to land, send the command `rostopic pub --once /bebop/land std_msgs/Empty`

### TensorRT

NOTE: There is an issue with the TensorRT inference model, which outputs different values than the regular (Keras) model. This is under investigation with Nvidia (see [discussion here](https://devtalk.nvidia.com/default/topic/1055217/tensorrt/model-accuracy-penalty-with-tensorrt-on-jetson-tx2/)).


On the Jetson TX2:  
•	Launch dronet_bebop.launch, dronet_launch.launch (from dronet_perception_trt) and deep_navigation.launch (in this order). This will connect to the drone and stream the video feed into the neural network, displaying its predictions. The prediction will then be picked up by the control block that can send commands to the drone (OFF by default)  
•	To issue the drone to start to fly, send the command  
`rostopic pub --once /bebop/takeoff std_msgs/Empty}`  
WARNING: THE DRONE WILL NOW START TO HOVER  
•	To issue the autonomous navigation, send the command  
`rostopic pub --once /bebop/state_change std_msgs/Bool "data: true"`  
WARNING: THE DRONE WILL NOW SELF NAVIGATE  
•	To stop the autonomous navigation, send the command  
`rostopic pub --once /bebop/state_change std_msgs/Bool "data: false"`  
•	To order the drone to land, send the command  
`rostopic pub --once /bebop/land std_msgs/Empty`  

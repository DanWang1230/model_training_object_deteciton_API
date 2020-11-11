# model_training_object_deteciton_API

This repo is part of my [capstone project](https://github.com/DanWang1230/Capstone_Programming_A_Self_Driving_Car) in Udacity's nano degree of self-driving car. The goal is to use a classifier to detect the traffic light colors. I used a pretrained model in the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Especially, I fine-tuned a `faster_rcnn_resnet101_coco` model using a [customized dataset](https://github.com/DanWang1230/creating_dataset_TFRecord).

I found [Repo 1](https://github.com/josehoras/Self-Driving-Car-Nanodegree-Capstone), [Repo 2](https://github.com/vatsl/TrafficLight_Detection-TensorFlowAPI) and [this](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d) instruction very helpful.

---
## Tensorflow Object Detection API

### 1. Models

```
git clone https://github.com/tensorflow/models.git
```

We will be mainly working on the folder `models/research/object_detection`.

### 2. Install Dependencies

```
sudo apt-get install protobuf-compiler
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```
A tutorial in `models/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb` can be helpful. 

### 3. Compile Protobuf and Add libraries to PYTHONPATH

```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## Pipeline

### 1. Model Config File

You can download the config file for a specifc pretrained model from [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

### 2. Modifying Config File

Then the config file is modified to work on four classes (red, green, yellow, off) rather than the 90 classes of the COCO dataset. Also, we need to create a `label_map_path`. An example is shown in `/data/label_map_path.pbtxt`. For very detaile instructions, please refer to [this file](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-4-training-the-model-68a9e5d5a333).

### 3. Training
```
python train.py --logtostderr --train_dir=models/train --pipeline_config_path=faster_rcnn_resnet101_coco.config
```



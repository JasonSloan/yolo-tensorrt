## 1. Environment Dependencies

### Installing TensorRT

```bash
gcc: 
sudo apt-get install gcc

driver(565.57.01):
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg
sudo mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
sudo apt-get install linux-headers-$(uname -r)
sudo apt-get install nvidia-open
sudo apt-get install cuda-drivers
sudo reboot

verify driver installation: nvidia-smi

cuda(12.6):
sudo apt-get install cuda-toolkit

cudnn(9):
sudo apt-get install zlib1g
sudo apt-get -y install cudnn9-cuda-12

tensorrt(10.6.0.26):
sudo apt-get install tensorrt

verify installation of tensorrt:
dpkg-query -W tensorrt
tensorrt        10.6.0.26-1+cuda12.6
```

### Installing spdlog

```
apt-get install libspdlog-dev
```

### Adding Pre-Compiled OpenCV

Copy the pre-compiled opencv folder to your machine.

If you are using ubuntu, download the pre-built opencv4.2 library directly from [here](https://github.com/JasonSloan/DeepFusion/releases/download/v111/opencv4.2.tar), and put them into the  'yolo-tensorrt' directory, then set

```bash
export LD_LIBRARY_PATH=/path/to/opencv4.2/lib:$LD_LIBRARY_PATH
```

## 2. Code Implementation

This repository contains **object detection** as well as **keypoints detection** inference code for YOLOv5, YOLOv8, and YOLOv11, based on the Linux version of the TensorRT inference framework.

### Features

- **Multi-threaded Asynchronous Inference**
  Inference is performed asynchronously using multiple threads.
- **Callback Functions for Results**
  Results are returned through callback functions.
- **Batch Inference Support**
  Supports both single-batch and multi-batch inference.

### Notes

- The current `CMakeLists.txt` file is configured to build a dynamic library.

- Test code for invoking inference can be found in the <test-model-infer> repository.

- To maintain consistency with YOLOv5, during inference with YOLOv8 or YOLOv11, the model's output dimensions should be [bs, n_grids, n_classes] instead of the official format [bs, nclasses, ngrids]. Therefore, you need to add a line of code to do transpose in the `forward` method of the `Detect` class in the `head.py` file of the official training code.

  ```bash
  class Detect(nn.Module):
  	......
      def forward(self, x):
          """Concatenates and returns predicted bounding boxes and class probabilities."""
          if self.end2end:
              return self.forward_end2end(x)

          for i in range(self.nl):
              x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
          if self.training:  # Training path
              return x
          y = self._inference(x)
          if self.export and str(self.__class__)[8:-2].split('.')[-1] == 'Detect':
              y = y.transpose(-1, -2)  # Add this line of code
          return y if self.export else (y, x)
  ```

------

Reference: https://github.com/shouxieai/tensorRT_Pro


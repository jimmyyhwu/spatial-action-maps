# spatial-action-maps

This code release accompanies the following paper:

### Spatial Action Maps for Mobile Manipulation

Jimmy Wu, Xingyuan Sun, Andy Zeng, Shuran Song, Johnny Lee, Szymon Rusinkiewicz, Thomas Funkhouser

*Robotics: Science and Systems (RSS), 2020*

[Project Page](https://spatial-action-maps.cs.princeton.edu) | [PDF](https://spatial-action-maps.cs.princeton.edu/paper.pdf) | [arXiv](https://arxiv.org/abs/2004.09141) | [Video](https://youtu.be/FcbIcU_VnzU)

**Abstract:** Typical end-to-end formulations for learning robotic navigation involve predicting a small set of steering command actions (e.g., step forward, turn left, turn right, etc.) from images of the current state (e.g., a bird's-eye view of a SLAM reconstruction). Instead, we show that it can be advantageous to learn with dense action representations defined in the same domain as the state. In this work, we present "spatial action maps," in which the set of possible actions is represented by a pixel map (aligned with the input image of the current state), where each pixel represents a local navigational endpoint at the corresponding scene location. Using ConvNets to infer spatial action maps from state images, action predictions are thereby spatially anchored on local visual features in the scene, enabling significantly faster learning of complex behaviors for mobile manipulation tasks with reinforcement learning. In our experiments, we task a robot with pushing objects to a goal location, and find that policies learned with spatial action maps achieve much better performance than traditional alternatives.

![](https://user-images.githubusercontent.com/6546428/91783724-e44bc400-ebb5-11ea-9b98-bfd90e52a82a.gif) | ![](https://user-images.githubusercontent.com/6546428/91783722-e44bc400-ebb5-11ea-902e-0b0468129231.gif) | ![](https://user-images.githubusercontent.com/6546428/91783719-e31a9700-ebb5-11ea-955f-2fa18a0508c1.gif) | ![](https://user-images.githubusercontent.com/6546428/91783721-e3b32d80-ebb5-11ea-82f5-6711e2e79fa6.gif)
:---: | :---: | :---: | :---:

![](https://user-images.githubusercontent.com/6546428/91783715-e1e96a00-ebb5-11ea-85fb-cb9e75f7f5a6.gif) | ![](https://user-images.githubusercontent.com/6546428/91783711-df871000-ebb5-11ea-9e6e-4a09d545b508.gif) | ![](https://user-images.githubusercontent.com/6546428/91783716-e2820080-ebb5-11ea-944c-83e49d16d9bc.gif)
:---: | :---: | :---:

## Installation

We recommend using a [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment for this codebase. The following commands will set up a new conda environment with the correct requirements (tested on Ubuntu 18.04.3 LTS):

```bash
# Create and activate new conda env
conda create -y -n my-conda-env python=3.7
conda activate my-conda-env

# Install pytorch and numpy
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
conda install -y numpy=1.17.3

# Install pip requirements
pip install -r requirements.txt

# Install shortest path module (used in simulation environment)
cd spfa
python setup.py install
```

## Quickstart

We provide four pretrained policies, one for each test environment. Use `download-pretrained.sh` to download them:

```bash
./download-pretrained.sh
```

You can then use `enjoy.py` to run a trained policy in the simulation environment.

For example, to load the pretrained policy for `SmallEmpty`, you can run:

```bash
python enjoy.py --config-path logs/20200125T213536-small_empty/config.yml
```

You can also run `enjoy.py` without specifying a config path, and it will find all policies in the `logs` directory and allow you to pick one to run:

```bash
python enjoy.py
```

## Training in the Simulation Environment

The [`config/experiments`](config/experiments) directory contains template config files for all experiments in the paper. To start a training run, you can give one of the template config files to the `train.py` script. For example, the following will train a policy on the `SmallEmpty` environment:

```
python train.py config/experiments/base/small_empty.yml
```

The training script will create a log directory and checkpoint directory for the new training run inside `logs/` and `checkpoints/`, respectively. Inside the log directory, it will also create a new config file called `config.yml`, which stores training run config variables and can be used to resume training or to load a trained policy for evaluation.

### Simulation Environment

To explore the simulation environment using our proposed dense action space (spatial action maps), you can use the `tools_click_agent.py` script, which will allow you to click on the local overhead map to select actions and move around in the environment.

```bash
python tools_click_agent.py
```

### Evaluation

Trained policies can be evaluated using the `evaluate.py` script, which takes in the config path for the training run. For example, to evaluate the `SmallEmpty` pretrained policy, you can run:

```
python evaluate.py --config-path logs/20200125T213536-small_empty/config.yml
```

This will load the trained policy from the specified training run, and run evaluation on it. The results are saved to an `.npy` file in the `eval` directory. You can then run `jupyter notebook` and navigate to [`eval_summary.ipynb`](eval_summary.ipynb) to load the `.npy` files and generate tables and plots of the results.

## Running in the Real Environment

We train policies in simulation and run them directly on the real robot by mirroring the real environment inside the simulation. To do this, we first use [ArUco](https://docs.opencv.org/4.1.2/d5/dae/tutorial_aruco_detection.html) markers to estimate 2D poses of robots and cubes in the real environment, and then use the estimated poses to update the simulation. Note that setting up the real environment, particularly the marker pose estimation, can take a fair amount of time and effort.

### Vector SDK Setup

If you previously ran `pip install -r requirements.txt` following the installation instructions above, the `anki_vector` library should already be installed. Run the following command to set up each robot you plan to use:

```bash
python -m anki_vector.configure
```

After the setup is complete, you can open the Vector config file located at `~/.anki_vector/sdk_config.ini` to verify that all of your robots are present.

You can also run some of the [official examples](https://developer.anki.com/vector/docs/downloads.html#sdk-examples) to verify that the setup procedure worked. For further reference, please see the [Vector SDK documentation](https://developer.anki.com/vector/docs/index.html).

### Connecting to the Vector

The following command will try to connect to all the robots in your Vector config file and keep them still. It will print out a message for each robot it successfully connects to, and can be used to verify that the Vector SDK can connect to all of your robots.

```bash
python vector_keep_still.py
```

**Note:** If you get the following error, you will need to make a small fix to the `anki_vector` library.

```
AttributeError: module 'anki_vector.connection' has no attribute 'CONTROL_PRIORITY_LEVEL'
```

Locate the `anki_vector/behavior.py` file inside your installed conda libraries. The full path should be in the error message. At the bottom of `anki_vector/behavior.py`, change `connection.CONTROL_PRIORITY_LEVEL.RESERVE_CONTROL` to `connection.ControlPriorityLevel.RESERVE_CONTROL`.

---

Sometimes the IP addresses of your robots will change. To update the Vector config file with the new IP addresses, you can run the following command:

```bash
python vector_run_mdns.py
```

The script uses mDNS to find all Vector robots on the local network, and will automatically update their IP addresses in the Vector config file. It will also print out the hostname, IP address, and MAC address of every robot found. Make sure `zeroconf` is installed (`pip install zeroconf`) or mDNS may not work well. Alternatively, you can just open the Vector config file at `~/.anki_vector/sdk_config.ini` in a text editor and manually update the IP addresses.

### Controlling the Vector

The `vector_keyboard_controller.py` script is adapted from the [remote control example](https://github.com/anki/vector-python-sdk/blob/master/examples/apps/remote_control/remote_control.py) in the official SDK, and can be used to verify that you are able to control the robot using the Vector SDK. Use it as follows:

```bash
python vector_keyboard_controller.py --robot-index ROBOT_INDEX
```

The `--robot-index` argument specifies the robot you wish to control and refers to the index of the robot in the Vector config file (`~/.anki_vector/sdk_config.ini`). If no robot index is specified, the script will check all robots in the Vector config file and select the first robot it is able to connect to.

### 3D Printed Parts

The real environment setup contains some 3D printed parts. We used the [Sindoh 3DWOX 1](https://www.amazon.com/Sindoh-3DWOX-Printer-New-Model/dp/B07C79C9RB) 3D printer to print them, but other printers should work too. All 3D model files are in the [`stl`](stl) directory:
* `cube.stl`: 3D model for the cubes (objects)
* `blade.stl`: 3D model for the bulldozer blade attached to the front of the robot
* `board-corner.stl`: 3D model for the board corners, which are used for pose estimation with ArUco markers

### Running Trained Policies on the Real Robot

First see the [`aruco`](aruco) directory for instructions on setting up pose estimation with ArUco markers.

Once the setup is completed, make sure the pose estimation server is started before proceeding:

```bash
cd aruco
python server.py
```

---

The `vector_click_agent.py` script is analogous to `tools_click_agent.py`, and allows you to click on the local overhead map to control the real robot. The script is also useful for verifying that all components of the real environment setup are working correctly, including pose estimation and robot control. The simulation environment should mirror the real setup with millimeter-level precision. You can start it using the following command:

```bash
python vector_click_agent.py --robot-index ROBOT_INDEX
```

If the poses in the simulation do not look correct, you can restart the pose estimation server with the `--debug` flag to enable debug visualizations:

```bash
cd aruco
python server.py --debug
```

---

Once you have verified that manual control with `vector_click_agent.py` works, you can then run a trained policy using the `vector_enjoy.py` script. For example, to load the `SmallEmpty` pretrained policy, you can run:

```bash
python vector_enjoy.py --robot-index ROBOT_INDEX --config-path logs/20200125T213536-small_empty/config.yml
```

## Citation

If you find this work useful for your research, please consider citing:

```
@inproceedings{wu2020spatial,
  title = {Spatial Action Maps for Mobile Manipulation},
  author = {Wu, Jimmy and Sun, Xingyuan and Zeng, Andy and Song, Shuran and Lee, Johnny and Rusinkiewicz, Szymon and Funkhouser, Thomas},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year = {2020}
}
```

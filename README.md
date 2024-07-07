<p align="center">
  <img src="https://github.com/Pedro-Roque/epic/blob/dev-intersection_selection/demo/output/experiment_trajectory.gif" />
</p>

# EpiC - Epipolar Coordination
A distributed formation control scheme based on epipolar geometry.

## Prerequisites:
The instructions here provided were tested in Ubuntu 20.04 and 22.04 operating systems, in an x86 architecture. To run the demo, please follow the steps below:
1. Make sure you have [Python Poetry](https://python-poetry.org/docs/#installation) and [Python 3.8](https://www.python.org/downloads/release/python-3810/) installed
2. Clone this repository to your local home folder
```bash
cd ~
git clone git@github.com:KTH-DHSG/epic.git
```
3. Run `poetry install` inside the cloned folder
```bash
cd ~/epic
poetry install
```
4. Run the demo!

## Running a demo:
To run the provided example, run: 
```bash
cd ~/epic
poetry run python demo/multi_agent_dynamic.py
```
This will run a simulation of a group of 6 agents in a simulated environment with a randomized initial and target pose in the vicinity of the poses in the file `demo/6_agent_geometry.json`.

## Examples:
In the first example, we show the simulated formation control of a group of 6 agents in a simulated environment. We consider camera types with different distortion models.

![Simulated Formation Control](https://github.com/Pedro-Roque/epic/blob/dev-intersection_selection/demo/output/simulated_animation.gif) 

In the second example, we show the real formation control of a group of 3 agents. The leader agent, to the left, is manually teleoperated, while the followers are controlled by the EpiC framework. The agents are equipped with monocular Flir Blackfly USB 3.0 Cameras.

![Real Formation Control](https://github.com/Pedro-Roque/epic/blob/dev-intersection_selection/demo/output/experiment_trajectory.gif)

## Experimental Setting
Here follow some details of the experimental conditions observed in the videos. For any other question, reach out to the repository maintainer:

- Features: SIFT
- Feature Matching: SIFT descriptors with RANSAC geometric filtering
- Cameras:
  - Leader: FLIR Blackfly USB 3
  - Follower 1: FLIR Blackfly USB 3
  - Follower 2: IMX323 USB Camera
  - Resolution used: 640 x 480 px
- Target Formation (position with respect to leader)
  - Leader: [0.0, 0.0, 0.0, −0.5235, −0.4628, 0.4753, 0.5345]
  - Follower 1: [−0.1136, 1.5824, 0.0370, −0.4499, −0.5410, 0.5492, 0.4506]
  - Follower 2: [−1.0820, 0.7141, 0.0190, 0.5003, 0.5035, −0.4985, −0.4976]
  - Format: [px py pz qx qy qz qw] (position x,y,z,  quaternion x,y,z,scalar)
- Onboard Computers: Intel NUC Core i3 16GB RAM (2020 model)

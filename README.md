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

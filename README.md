# Robot Navigation Using AI Techniques

A comprehensive robotics project implementing three distinct AI approaches for autonomous navigation in a simulated Gazebo environment. The robot navigates from a start position to a goal while intelligently avoiding obstacles using Fuzzy Logic, Behavior Trees, and Q-Learning (Reinforcement Learning).

## Project Overview

This project demonstrates autonomous robot navigation from position (-4, -4) to goal position (-3.5, 3.5) in a 10m × 10m arena with various obstacles. Three different AI techniques are implemented and compared:

- **Fuzzy Logic Controller** - Rule-based navigation with adaptive obstacle avoidance
- **Behavior Tree Controller** - Hierarchical decision-making with priority-based behaviors  
- **Q-Learning Agent** - Reinforcement learning with training and testing capabilities

## Prerequisites

Install the following software before starting:

1. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
2. **MobaXterm** (Windows only) - [Download here](https://mobaxterm.mobatek.net/download.html)
3. **Git** - [Download here](https://git-scm.com/downloads)

## Installation & Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd 218_PROJECT
```

### Step 2: Start Docker Desktop
- Launch Docker Desktop and wait for "Engine running" status
- Keep it running in the background

### Step 3: Start MobaXterm (Windows Users)
- Open MobaXterm for X11 display forwarding
- Leave it running in the background

### Step 4: Build the Docker Container
Open PowerShell or Terminal in the project directory:
```bash
docker-compose up -d
```
**Note:** Initial setup takes 10-15 minutes to download dependencies.

### Step 5: Connect to Container
```bash
docker exec -it gazebo_sim bash
```

### Step 6: Configure ROS Environment
Inside the container, run:
```bash
source /opt/ros/noetic/setup.bash
export DISPLAY=host.docker.internal:0
export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH
```

### Step 7: Launch Gazebo Simulation
```bash
LIBGL_ALWAYS_SOFTWARE=1 roslaunch gazebo_ros empty_world.launch \
  world_name:=/root/robot_project/worlds/simple_world.world \
  paused:=false gui:=true
```

A Gazebo window will open showing the robot in the arena with obstacles and a green target cylinder.

## Running the Controllers

Open a **second terminal** and connect:
```bash
docker exec -it gazebo_sim bash
source /opt/ros/noetic/setup.bash
cd /root/robot_project/scripts
```

### 1. Fuzzy Logic Controller
Uses rule-based fuzzy inference for navigation with adaptive obstacle avoidance.

```bash
python3 controller.py
```

### 2. Behavior Tree Controller
Employs hierarchical behavior trees with priority-based decision making.

```bash
python3 behavior_tree_complete.py
```

### 3. Q-Learning Controller

**Training Mode** (30 episodes - default):
```bash
python3 rl_navigation_final.py
```

**Training Mode** (custom episodes):
```bash
python3 rl_navigation_final.py 50
```

**Testing Mode** (uses trained Q-table):
```bash
python3 rl_navigation_final.py test
```

**Testing Mode** (custom number of tests):
```bash
python3 rl_navigation_final.py test 5
```

## Stopping the Simulation

1. Press `Ctrl+C` in the terminal running the AI controller
2. Press `Ctrl+C` in the terminal running Gazebo
3. Exit the container: `exit`
4. Stop Docker: `docker-compose down`

## Project Structure

```
218_PROJECT/
├── models/
│   └── simple_robot/
│       ├── model.config          # Robot metadata
│       └── model.sdf             # Robot definition with sensors
├── worlds/
│   └── simple_world.world        # Simulation environment with obstacles
├── scripts/
│   ├── controller.py             # Fuzzy logic main controller
│   ├── fuzzy_brain.py            # Fuzzy inference system
│   ├── behavior_tree_complete.py # Behavior tree implementation
│   └── rl_navigation_final.py    # Q-learning reinforcement learning
├── docker-compose.yml            # Docker configuration
├── REPORT_.pdf                   # Project Report
└── README.md                     # Project documentation
```

## Technical Specifications

**Environment:**
- Arena: 10m × 10m bounded area
- Start Position: (-4, -4)
- Goal Position: (-3.5, 3.5) marked by green cylinder
- Goal Tolerance: 0.8m

**Robot Sensors:**
- 360° LIDAR (10m range, 360 samples)
- Odometry
- IMU
- Contact/Bumper sensor

**Obstacles:**
- Red box, yellow cylinder
- Narrow passage walls
- Purple obstacle
- Checkpoint sphere

## Approach Comparison

| Approach | Strengths | Best For |
|----------|-----------|----------|
| **Fuzzy Logic** | Simple rules, fast execution, no training | Predictable environments |
| **Behavior Trees** | Modular, hierarchical priorities | Complex decision sequences |
| **Q-Learning** | Learns optimal policy, adapts to environment | Dynamic environments |

## Troubleshooting

**Gazebo window doesn't appear:**
- Ensure Docker Desktop is running
- On Windows, verify MobaXterm is running
- Check DISPLAY variable: `echo $DISPLAY`

**Robot doesn't move:**
- Verify ROS environment is sourced
- Check topics: `rostopic list`
- Ensure controller script is running

**Sensors not working:**
- Wait 5-10 seconds after launching Gazebo
- Check sensor topics: `rostopic echo /robot/laser/scan`

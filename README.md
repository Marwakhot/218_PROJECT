# CSCI 218 - Robot Navigation Using AI Techniques

A Gazebo simulation project implementing three AI approaches (Fuzzy Logic, Behavior Trees, Q-Learning) for autonomous robot navigation from start position (-4, -4) to goal position (-3.5, 3.5) while avoiding obstacles.

## Team Members
- Joslin Jolly (8964178)
- Marwa Khot (8963186)
- Raahim Ahmed (8699124)
- Saad Bin Waqas (8186388)
- Zobia Shaikh (8881820)

## Prerequisites

Before starting, install the following on your computer:

1. **Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. **MobaXterm** (Windows only): https://mobaxterm.mobatek.net/download.html
3. **Git**: https://git-scm.com/downloads

## Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd 218_PROJECT
```

### 2. Start Docker Desktop

- Open Docker Desktop and wait until it shows "Engine running"
- Leave it running in the background

### 3. Start MobaXterm (Windows)

- Open MobaXterm
- Leave it running in the background (this provides X11 display forwarding)

### 4. Build and Start the Container

Open PowerShell or Terminal in the project folder:
```bash
docker-compose up -d
```

**Note**: First-time setup takes 10-15 minutes to download all dependencies.

### 5. Connect to the Container
```bash
docker exec -it gazebo_sim bash
```

### 6. Source ROS Environment

Inside the container, run:
```bash
source /opt/ros/noetic/setup.bash
export DISPLAY=host.docker.internal:0
export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH
```

### 7. Launch Gazebo Simulation
```bash
LIBGL_ALWAYS_SOFTWARE=1 roslaunch gazebo_ros empty_world.launch world_name:=/root/robot_project/worlds/simple_world.world paused:=false gui:=true
```

A Gazebo window should open showing the robot (gray box with wheels) in the arena with obstacles and a green target cylinder at position (-3.5, 3.5).

## Running the AI Controllers

Open a **second terminal** and connect to the container:
```bash
docker exec -it gazebo_sim bash
source /opt/ros/noetic/setup.bash
cd /root/robot_project/scripts
```

### Option 1: Fuzzy Logic Controller
```bash
python3 controller.py
```

The robot will navigate using fuzzy logic rules with escape mechanisms for obstacle avoidance.

### Option 2: Behavior Tree Controller
```bash
python3 behavior_tree_complete.py
```

The robot will navigate using hierarchical behavior tree with priority-based decision making.

### Option 3: Q-Learning Controller

**Training Mode** (30 episodes):
```bash
python3 rl_navigation_final.py
```

**Training Mode** (custom episodes):
```bash
python3 rl_navigation_final.py 50
```

**Testing Mode** (using trained Q-table):
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
4. Stop Docker containers: `docker-compose down`

## Project Structure
```
218_PROJECT/
├── models/
│   └── simple_robot/
│       ├── model.config          # Robot metadata
│       └── model.sdf             # Robot definition with sensors
├── worlds/
│   └── simple_world.world        # Simulation environment
├── scripts/
│   ├── controller.py             # Fuzzy logic main controller
│   ├── fuzzy_brain.py            # Fuzzy logic inference system
│   ├── behavior_tree_complete.py # Behavior tree implementation
│   └── rl_navigation_final.py    # Q-learning agent
├── docker-compose.yml            # Docker configuration
└── README.md                     # This file
```

## Environment Details

- **Arena**: 10m × 10m bounded by walls
- **Start Position**: (-4, -4)
- **Goal Position**: (-3.5, 3.5) - Green cylinder
- **Robot Sensors**: 360° LIDAR (10m range), Odometry, IMU, Bumper
- **Obstacles**: Red box, yellow cylinder, narrow passage walls, purple obstacle, checkpoint sphere

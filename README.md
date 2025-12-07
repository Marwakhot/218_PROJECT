# Robot AI Navigation Project

A Gazebo simulation environment for testing different AI approaches to autonomous robot navigation.

## Overview

This project provides a two-wheeled differential drive robot in a simulated environment with obstacles. The robot is equipped with multiple sensors and must navigate from a starting position to a goal while avoiding obstacles.

**Environment:**
- Start Position: (-4, -4)
- Goal Position: (-3.5, 3.5) - Green cylinder
- Checkpoint: (0, -1) - Orange sphere
- Bounded 10x10m arena with walls and obstacles

**Robot Sensors:**
- LIDAR (360° laser scanner, 10m range)
- Front-facing camera (640x480)
- IMU (orientation and acceleration)
- Contact sensor (collision detection)
- Odometry (position and velocity)

---

## Setup Instructions

### Prerequisites
1. Install **Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. Install **MobaXterm**: https://mobaxterm.mobatek.net/download.html

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Marwakhot/218_PROJECT.git
cd 218_PROJECT
```

2. **Start Docker Desktop** (wait until it shows "running")

3. **Open MobaXterm** (keep it running in the background for X11 display)

4. **Open PowerShell in the project folder and run:**
```bash
docker-compose up -d
```
*(First time takes 10-15 minutes to download everything)*

5. **Connect to the container:**
```bash
docker exec -it gazebo_sim bash
```

6. **Set up the environment:**
```bash
export DISPLAY=host.docker.internal:0
export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH
```

7. **Launch the simulation (opens Gazebo window with robot and environment):**
```bash
gazebo /root/robot_project/worlds/simple_world.world
```
   *You should see the robot (gray box with wheels) in the arena with obstacles*

---

## Robot Control Interface

### Controlling the Robot

**Topic:** `/cmd_vel`  
**Type:** `geometry_msgs/Twist`

```python
from geometry_msgs.msg import Twist

twist = Twist()
twist.linear.x = 0.5   # Forward/backward speed (m/s)
twist.angular.z = 0.3  # Rotation speed (rad/s)
publisher.publish(twist)
```

### Available Sensor Topics

| Sensor | Topic | Message Type | Rate |
|--------|-------|--------------|------|
| LIDAR | `/robot/laser/scan` | `sensor_msgs/LaserScan` | 10 Hz |
| Camera | `/robot/camera/image_raw` | `sensor_msgs/Image` | 30 Hz |
| IMU | `/robot/imu` | `sensor_msgs/Imu` | 100 Hz |
| Odometry | `/odom` | `nav_msgs/Odometry` | 100 Hz |
| Ground Truth | `/robot/ground_truth/state` | `nav_msgs/Odometry` | 100 Hz |
| Bumper | `/robot/bumper` | `gazebo_msgs/ContactsState` | 10 Hz |

---

## For Team Members

### Your Task

Implement one of the following AI approaches to navigate the robot from start to goal:

1. **Reinforcement Learning** (DQN, PPO, SAC, etc.)
2. **Fuzzy Logic Control**
3. **Behavior Trees**

### Development Workflow

1. **Create your branch:**
```bash
git checkout -b your-name-approach
# Example: git checkout -b marwa-rl
```

2. **Create your script in the `scripts/` folder:**
```bash
cd scripts/
# Create your Python script
nano my_controller.py
```

3. **Make it executable:**
```bash
chmod +x my_controller.py
```

4. **Test your approach:**
```bash
# In the Docker container
cd /root/robot_project/scripts
python3 my_controller.py
```

5. **Commit and push your work:**
```bash
git add scripts/my_controller.py
git commit -m "Implemented [approach] navigation"
git push origin your-name-approach
```

6. **Create a Pull Request on GitHub**

---

## Understanding Sensor Data

### LIDAR (LaserScan)
```python
# 360 distance measurements (in meters)
msg.ranges  # Array[360]: distances at each angle
# Index 0 = right side, Index 180 = left side
# Front of robot = indices 0-30 and 330-359

# Example: Get minimum distance in front
front_ranges = np.concatenate([msg.ranges[:30], msg.ranges[-30:]])
min_distance = np.min(front_ranges)
```

### Odometry
```python
# Robot position
x = msg.pose.pose.position.x
y = msg.pose.pose.position.y

# Robot orientation (quaternion -> yaw conversion needed)
orientation = msg.pose.pose.orientation
# Use tf.transformations or manual conversion to get yaw angle

# Velocity
linear_velocity = msg.twist.twist.linear.x
angular_velocity = msg.twist.twist.angular.z
```

### IMU
```python
# Angular velocity (rad/s)
angular_vel_z = msg.angular_velocity.z

# Linear acceleration (m/s²)
accel_x = msg.linear_acceleration.x
accel_y = msg.linear_acceleration.y
```

---

## Testing & Debugging

### Test Manual Control
```bash
# Move forward
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.5}" -r 10

# Rotate
rostopic pub /cmd_vel geometry_msgs/Twist "angular: {z: 0.5}" -r 10

# Stop
rostopic pub /cmd_vel geometry_msgs/Twist "{}" -1
```

### Monitor Sensors
```bash
# List all available topics
rostopic list

# View LIDAR data
rostopic echo /robot/laser/scan

# View position
rostopic echo /odom

# Check topic frequency
rostopic hz /robot/laser/scan
```

### Reset Simulation
```bash
rosservice call /gazebo/reset_simulation
```

---

## Project Structure

```
218_PROJECT/
├── models/
│   └── simple_robot/
│       ├── model.config
│       └── model.sdf          # Robot definition with sensors
├── worlds/
│   └── simple_world.world     # Simulation environment
├── scripts/
│   └── [your_scripts_here]    # Add your AI controllers here
├── docker-compose.yml
└── README.md
```

---

## Daily Workflow

**Starting work:**
```bash
docker-compose up -d
docker exec -it gazebo_sim bash
export DISPLAY=host.docker.internal:0
export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH

# Launch Gazebo (opens simulation window)
gazebo /root/robot_project/worlds/simple_world.world
```
*Gazebo window will open showing the robot in the arena with obstacles*

**In a new terminal (for running your script):**
```bash
docker exec -it gazebo_sim bash
cd /root/robot_project/scripts
python3 your_script.py
```

**Stopping work:**
```bash
# Exit the container
exit

# Stop Docker
docker-compose down
```

---

## Success Criteria

- Navigate from start position to goal position
- Avoid all obstacles and walls
- Smooth, efficient path
- No collisions
- Reasonable completion time

---

## Useful ROS Commands

```bash
# View ROS computation graph
rqt_graph

# Monitor topics with GUI
rqt

# Record data for later analysis
rosbag record -a

# List all active nodes
rosnode list

# Get info about a topic
rostopic info /cmd_vel
```

---

## Troubleshooting

**Robot not moving?**
- Check if Gazebo is running
- Verify `/cmd_vel` topic exists: `rostopic list`
- Test manual control: `rostopic pub /cmd_vel ...`

**Sensors not publishing?**
- Check topic list: `rostopic list`
- Check topic frequency: `rostopic hz /robot/laser/scan`
- Restart Gazebo if needed

**Display issues?**
- Ensure MobaXterm is running
- Check DISPLAY variable: `echo $DISPLAY`
- Should be `host.docker.internal:0`

**Docker issues?**
- Restart Docker Desktop
- Remove containers: `docker-compose down`
- Rebuild: `docker-compose up -d`

---

## Resources

- **ROS Wiki:** http://wiki.ros.org/
- **ROS Tutorials:** http://wiki.ros.org/ROS/Tutorials
- **Gazebo Tutorials:** http://gazebosim.org/tutorials
- **geometry_msgs/Twist:** http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
- **sensor_msgs/LaserScan:** http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html

---

## Team Coordination

- Each member works on their own branch
- Test thoroughly before pushing
- Document your approach in comments
- Create Pull Requests for code review
- Communicate obstacles and progress

Good luck! 🚀

## Setup Instructions

### Prerequisites
1. Install **Docker Desktop**: https://www.docker.com/products/docker-desktop/
2. Install **MobaXterm**: https://mobaxterm.mobatek.net/download.html

### Installation Steps

1. **Clone this repository:**
```bash
   git clone https://github.com/Marwakhot/218_PROJECT.git
   cd robot-ai-simulation
```

2. **Start Docker Desktop** (wait until it says "running")

3. **Open MobaXterm** (keep it running in background)

4. **Open PowerShell in the project folder** and run:
```bash
   docker-compose up -d
```
   (First time takes 10-15 minutes to download everything)

5. **Connect to the container:**
```bash
   docker exec -it gazebo_sim bash
```

6. **Set up environment:**
```bash
   export DISPLAY=host.docker.internal:0
   export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH
```

7. **Launch the simulation:**
```bash
   gazebo /root/robot_project/worlds/simple_world.world
```

## Robot Sensors

- **LIDAR** (360° laser scanner) - Blue rays for obstacle detection
- **Camera** - Front-facing for visual navigation
- **IMU** - Orientation and movement tracking
- **Contact Sensor** - Collision detection

## Environment Features

- Bounded arena with walls
- Multiple obstacles (boxes, cylinders)
- Narrow passages
- Green target zone (goal)
- Orange checkpoint

## Project Structure
```
robot_project/
├── models/
│   └── simple_robot/      # Robot model with sensors
├── worlds/
│   └── simple_world.world # Simulation environment
├── scripts/               # AI control scripts (add yours here)
└── docker-compose.yml     # Docker configuration
```

## Daily Workflow

**Starting work:**
```bash
docker-compose up -d
docker exec -it gazebo_sim bash
export DISPLAY=host.docker.internal:0
export GAZEBO_MODEL_PATH=/root/robot_project/models:$GAZEBO_MODEL_PATH
gazebo /root/robot_project/worlds/simple_world.world
```

**Stopping work:**
```bash
docker-compose down
```

## Adding Your Code

Each team member should:
1. Create a branch: `git checkout -b your-name-feature`
2. Add your scripts to the `scripts/` folder
3. Test in Gazebo
4. Commit and push: `git push origin your-name-feature`
5. Create a Pull Request on GitHub

## Notes

- The robot starts at position (-4, -4)
- Goal: Reach the green target zone at (-3.5, 3.5)
- Avoid obstacles and walls
- Robot can be controlled via topics (to be documented by each AI approach)

#!/usr/bin/env python3
"""
FINAL FIX - Simple obstacle avoidance
"""

import math
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

import py_trees
from py_trees.behaviour import Behaviour
from py_trees.common import Status
from py_trees.composites import Sequence, Selector

# GOAL is green cylinder at top-left
GOAL_X = -3.5
GOAL_Y = 3.5
GOAL_TOLERANCE = 0.8
OBSTACLE_DISTANCE = 0.8 
MAX_LINEAR = 0.35
MAX_ANGULAR = 0.8
LASER_FRONT_WINDOW = 40

class RobotState:
    def __init__(self):
        self.odom = None
        self.laser = None

robot_state = RobotState()
cmd_pub = None

def publish_twist(linear_x=0.0, angular_z=0.0):
    global cmd_pub
    if cmd_pub is None:
        return
    t = Twist()
    t.linear.x = linear_x
    t.angular.z = angular_z
    cmd_pub.publish(t)

def get_robot_pose():
    od = robot_state.odom
    if od is None:
        return None
    x = od.pose.pose.position.x
    y = od.pose.pose.position.y
    q = od.pose.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny, cosy)
    return x, y, yaw

def front_min_distance():
    scan = robot_state.laser
    if scan is None:
        return float('inf')
    
    left = scan.ranges[:LASER_FRONT_WINDOW]
    right = scan.ranges[-LASER_FRONT_WINDOW:]
    front = list(left) + list(right)
    cleaned = [d for d in front if not (math.isinf(d) or math.isnan(d)) and d > 0.1]
    
    if not cleaned:
        return float('inf')
    return min(cleaned)

def get_side_distances():
    scan = robot_state.laser
    if scan is None:
        return float('inf'), float('inf')
    
    left = [scan.ranges[i] for i in range(70, 110) if i < len(scan.ranges)]
    left_clean = [d for d in left if not (math.isinf(d) or math.isnan(d)) and d > 0.1]
    left_min = min(left_clean) if left_clean else float('inf')
    
    right = [scan.ranges[i] for i in range(250, 290) if i < len(scan.ranges)]
    right_clean = [d for d in right if not (math.isinf(d) or math.isnan(d)) and d > 0.1]
    right_min = min(right_clean) if right_clean else float('inf')
    
    return left_min, right_min

class IsGoalReached(Behaviour):
    def __init__(self, name="IsGoalReached"):
        super(IsGoalReached, self).__init__(name)

    def update(self):
        pose = get_robot_pose()
        if pose is None:
            return Status.FAILURE
        
        x, y, _ = pose
        dist = math.hypot(GOAL_X - x, GOAL_Y - y)
        
        # Check if close to goal
        if dist <= GOAL_TOLERANCE:
            publish_twist(0.0, 0.0)
            print("\n" + "=" * 60)
            print("🎯 GOAL REACHED!")
            print(f"Final Position: ({x:.2f}, {y:.2f})")
            print(f"Goal Position: ({GOAL_X}, {GOAL_Y})")
            print(f"Distance: {dist:.2f}m")
            print("=" * 60 + "\n")
            return Status.SUCCESS
        
        return Status.FAILURE
    
class IsObstacleNear(Behaviour):
    def __init__(self, name="IsObstacleNear"):
        super(IsObstacleNear, self).__init__(name)

    def update(self):
        dist = front_min_distance()
        if dist < OBSTACLE_DISTANCE:
            print(f"⚠️ OBSTACLE {dist:.2f}m!")
            return Status.SUCCESS
        return Status.FAILURE

class AvoidObstacle(Behaviour):
    def __init__(self, name="AvoidObstacle"):
        super(AvoidObstacle, self).__init__(name)
        self.phase = 0
        self.start_time = None
        self.turn_direction = 1.0

    def initialise(self):
        self.phase = 0
        self.start_time = rospy.Time.now()
        
        left_dist, right_dist = get_side_distances()
        
        # Pick direction with MORE space
        if left_dist > right_dist:
            self.turn_direction = 1.0
            print(f"  TURN LEFT (L:{left_dist:.1f}m > R:{right_dist:.1f}m)")
        else:
            self.turn_direction = -1.0
            print(f"  TURN RIGHT (R:{right_dist:.1f}m > L:{left_dist:.1f}m)")

    def update(self):
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        
        # PHASE 0: TURN 90 degrees (4 seconds)
        if self.phase == 0:
            if elapsed < 4.0:
                publish_twist(linear_x=0.0, angular_z=self.turn_direction * 0.8)
                return Status.RUNNING
            else:
                print("  FORWARD")
                self.phase = 1
                self.start_time = rospy.Time.now()
        
        # PHASE 1: GO FORWARD (5 seconds)
        elif self.phase == 1:
            if elapsed < 5.0:
                publish_twist(linear_x=0.35, angular_z=0.0)
                return Status.RUNNING
            else:
                print("  ✓ AVOIDED\n")
                return Status.SUCCESS
        
        return Status.RUNNING

    def terminate(self, new_status):
        self.phase = 0

class MoveTowardsGoal(Behaviour):
    def __init__(self, name="MoveTowardsGoal"):
        super(MoveTowardsGoal, self).__init__(name)
        self.last_print = 0

    def update(self):
        pose = get_robot_pose()
        if pose is None:
            return Status.FAILURE
        
        x, y, yaw = pose
        distance = math.hypot(GOAL_X - x, GOAL_Y - y)
        front_dist = front_min_distance()
        
        # Check obstacle
        if front_dist < OBSTACLE_DISTANCE:
            return Status.FAILURE
        
        # Angle to goal
        goal_angle = math.atan2(GOAL_Y - y, GOAL_X - x)
        angle_diff = goal_angle - yaw
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # SIMPLIFIED: Just turn if needed, otherwise GO FORWARD
        if abs(angle_diff) > 0.5:
            # Turn
            linear_vel = 0.1
            angular_vel = 0.5 if angle_diff > 0 else -0.5
        else:
            # GO STRAIGHT
            linear_vel = 0.35
            angular_vel = 0.0
        
        publish_twist(linear_x=linear_vel, angular_z=angular_vel)
        
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_print > 2.0:
            print(f"[NAV] ({x:.1f},{y:.1f}) Goal:{distance:.1f}m Angle:{math.degrees(angle_diff):+.0f}°")
            self.last_print = current_time
        
        return Status.RUNNING
    
def create_behavior_tree():
    root = Selector(name="Root", memory=False)
    
    goal_check = IsGoalReached()
    
    obstacle_sequence = Sequence(name="Avoid", memory=False)
    obstacle_sequence.add_children([
        IsObstacleNear(),
        AvoidObstacle()
    ])
    
    navigation = MoveTowardsGoal()
    
    root.add_children([
        goal_check,
        obstacle_sequence,
        navigation
    ])
    
    return root

def odom_callback(msg):
    robot_state.odom = msg

def laser_callback(msg):
    robot_state.laser = msg

def main():
    global cmd_pub
    
    print("\n" + "=" * 60)
    print("BEHAVIOR TREE NAVIGATION - FINAL VERSION")
    print("=" * 60)
    
    rospy.init_node('bt_nav', anonymous=True)
    
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.Subscriber('/robot/laser/scan', LaserScan, laser_callback)
    
    print("⏳ Waiting for sensors...")
    rate = rospy.Rate(10)
    timeout = 0
    while (robot_state.odom is None or robot_state.laser is None) and timeout < 100:
        rate.sleep()
        timeout += 1
    
    if timeout >= 100:
        print("❌ Timeout!")
        return
    
    x = robot_state.odom.pose.pose.position.x
    y = robot_state.odom.pose.pose.position.y
    distance = math.hypot(GOAL_X - x, GOAL_Y - y)
    
    print(f"\n✓ Start: ({x:.1f}, {y:.1f})")
    print(f"✓ Goal: ({GOAL_X}, {GOAL_Y})")
    print(f"✓ Distance: {distance:.1f}m")
    print(f"✓ Obstacle detection: {OBSTACLE_DISTANCE}m\n")
    
    root = create_behavior_tree()
    rate = rospy.Rate(10)
    
    try:
        while not rospy.is_shutdown():
            root.tick_once()
            
            if root.status == Status.SUCCESS:
                print("\n🎉 MISSION COMPLETE!")
                publish_twist(0.0, 0.0)
                break
            
            rate.sleep()
    
    except KeyboardInterrupt:
        print("\n⛔ Stopped")
    
    finally:
        publish_twist(0.0, 0.0)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
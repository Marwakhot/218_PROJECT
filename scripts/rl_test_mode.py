#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math
import pickle
import os

class SimpleQLearningAgent:
    """Simple Q-Learning agent for robot navigation"""
    
    def __init__(self, n_states=100, n_actions=5):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.q_table = np.zeros((n_states, n_actions))
        
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.epsilon = 0.5  # Balanced exploration
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.05
        
        # SIMPLIFIED ACTIONS - Less likely to get stuck
        self.actions = [
            [0.5, 0.0],     # Forward
            [0.3, 0.5],     # Forward + left
            [0.3, -0.5],    # Forward + right
            [0.0, 0.8],     # Rotate left
            [0.0, -0.8],    # Rotate right
        ]
        
        self.action_names = [
            "FORWARD",
            "FWD+LEFT", 
            "FWD+RIGHT",
            "ROTATE LEFT",
            "ROTATE RIGHT"
        ]
    
    def discretize_state(self, lidar_data, distance_to_goal, angle_to_goal):
        front_distance = np.mean(lidar_data[0:3])
        
        if front_distance < 0.6:
            distance_bin = 0
        elif front_distance < 1.5:
            distance_bin = 1
        elif front_distance < 3.0:
            distance_bin = 2
        else:
            distance_bin = 3
        
        if angle_to_goal < -math.pi/3:
            angle_bin = 0
        elif angle_to_goal < math.pi/3:
            angle_bin = 1
        else:
            angle_bin = 2
        
        if distance_to_goal < 2.0:
            goal_dist_bin = 0
        elif distance_to_goal < 5.0:
            goal_dist_bin = 1
        else:
            goal_dist_bin = 2
        
        state = distance_bin * 9 + angle_bin * 3 + goal_dist_bin
        return min(state, self.n_states - 1)
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename="/root/robot_project/scripts/q_table.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filename="/root/robot_project/scripts/q_table.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        return False


class RobotNavigationEnv:
    
    def __init__(self):
        rospy.init_node('rl_navigation_final', anonymous=True)
        
        self.robot_x = -4.0
        self.robot_y = -4.0
        self.robot_theta = 0.0
        self.lidar_data = np.ones(24) * 10.0
        self.lidar_received = False
        self.odom_received = False
        
        self.goal_x = -3.5
        self.goal_y = 3.5
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/robot/laser/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        print("⏳ Waiting for sensor data...")
        rate = rospy.Rate(10)
        timeout = 0
        while (not self.lidar_received or not self.odom_received) and timeout < 100:
            rate.sleep()
            timeout += 1
        
        if timeout < 100:
            print(f"✓ Sensors connected!")
            print(f"✓ Start: ({self.robot_x:.2f}, {self.robot_y:.2f})")
            print(f"✓ Goal: ({self.goal_x}, {self.goal_y})")
    
    def lidar_callback(self, data):
        self.lidar_received = True
        ranges = np.array(data.ranges)
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.0)
        num_points = len(ranges)
        if num_points > 0:
            step = max(1, num_points // 24)
            self.lidar_data = ranges[::step][:24]
            if len(self.lidar_data) < 24:
                self.lidar_data = np.pad(self.lidar_data, (0, 24 - len(self.lidar_data)), constant_values=10.0)
    
    def odom_callback(self, data):
        self.odom_received = True
        self.robot_x = data.pose.pose.position.x
        self.robot_y = data.pose.pose.position.y
        
        orientation = data.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_theta = math.atan2(siny_cosp, cosy_cosp)
    
    def get_distance_to_goal(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        return math.sqrt(dx**2 + dy**2)
    
    def get_angle_to_goal(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - self.robot_theta
        return math.atan2(math.sin(angle_diff), math.cos(angle_diff))
    
    def execute_action(self, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.angular.z = action[1]
        self.cmd_vel_pub.publish(cmd)
        rospy.sleep(0.2)
    
    def calculate_reward(self, prev_distance):
        distance = self.get_distance_to_goal()
        reward = 0.0
        done = False
        
        # Big reward for getting closer
        if prev_distance is not None:
            progress = (prev_distance - distance)
            reward += progress * 20.0  # BIG reward for progress
        
        # GOAL!
        if distance < 0.8:
            reward += 500.0
            done = True
            return reward, done, distance
        
        # Collision
        min_distance = np.min(self.lidar_data)
        if min_distance < 0.25:
            reward -= 100.0
            done = True
            return reward, done, distance
        
        # Penalty for being too close
        if min_distance < 0.5:
            reward -= 5.0
        
        # Small time penalty
        reward -= 0.5
        
        return reward, done, distance
    
    def stop(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
    
    def reset_robot(self):
        """Reset robot to start position"""
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            
            state = ModelState()
            state.model_name = 'simple_robot'
            state.pose.position.x = -4.0
            state.pose.position.y = -4.0
            state.pose.position.z = 0
            state.pose.orientation.w = 1.0
            
            set_state(state)
            rospy.sleep(1.0)
            print("  🔄 Robot reset to start position")
        except:
            print("  ⚠️ Could not reset robot, continuing...")


def train_rl_final(episodes=25):
    
    print("\n" + "=" * 60)
    print("REINFORCEMENT LEARNING - FINAL RUN (25 EPISODES)")
    print("=" * 60)
    
    try:
        env = RobotNavigationEnv()
        
        if not env.lidar_received or not env.odom_received:
            print("❌ ERROR: No sensor data!")
            return
        
        agent = SimpleQLearningAgent()
        agent.load()
        
        print(f"\nStarting {episodes} episodes")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print("=" * 60 + "\n")
        
        success_count = 0
        
        for episode in range(episodes):
            print(f"\n{'='*60}")
            print(f"EPISODE {episode + 1}/{episodes}")
            print(f"{'='*60}")
            
            # Reset if stuck far from start
            if episode > 0:
                dist_from_start = math.sqrt((env.robot_x + 4)**2 + (env.robot_y + 4)**2)
                if dist_from_start > 8 or env.get_distance_to_goal() > 10:
                    env.reset_robot()
            
            env.stop()
            rospy.sleep(0.5)
            
            # INITIAL TURN if facing wall
            if episode == 0:
                initial_front = np.min(env.lidar_data[:3])
                if initial_front < 1.5:
                    print("🔄 Performing 180° turn...")
                    for i in range(50):
                        cmd = Twist()
                        cmd.angular.z = 0.8
                        env.cmd_vel_pub.publish(cmd)
                        rospy.sleep(0.1)
                    env.stop()
                    rospy.sleep(0.3)
                    print("✓ Turn complete!\n")
            
            distance = env.get_distance_to_goal()
            angle = env.get_angle_to_goal()
            state = agent.discretize_state(env.lidar_data, distance, angle)
            
            total_reward = 0
            prev_distance = distance
            max_steps = 200
            
            print(f"Start: ({env.robot_x:.1f},{env.robot_y:.1f}) → Goal: {distance:.1f}m away")
            
            for step in range(max_steps):
                action_idx = agent.choose_action(state)
                action = agent.actions[action_idx]
                action_name = agent.action_names[action_idx]
                
                env.execute_action(action)
                
                reward, done, distance = env.calculate_reward(prev_distance)
                total_reward += reward
                
                angle = env.get_angle_to_goal()
                next_state = agent.discretize_state(env.lidar_data, distance, angle)
                
                agent.update(state, action_idx, reward, next_state)
                state = next_state
                prev_distance = distance
                
                # Print every 20 steps
                if step % 20 == 0:
                    min_lidar = np.min(env.lidar_data)
                    print(f"  Step {step:3d}: {action_name:12s} | Dist={distance:.2f}m | Front={min_lidar:.2f}m | R={total_reward:6.1f}")
                
                if done:
                    if distance < 0.8:
                        print("\n" + "=" * 60)
                        print("🎯 GOAL REACHED!")
                        print(f"Position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
                        print(f"Goal: ({env.goal_x}, {env.goal_y})")
                        print(f"Distance: {distance:.2f}m")
                        print(f"Steps: {step+1}")
                        print("=" * 60 + "\n")
                        success_count += 1
                    else:
                        print(f"\n  💥 Collision at step {step+1}")
                    break
            
            agent.decay_epsilon()
            
            print(f"\n  Episode Summary:")
            print(f"    Reward: {total_reward:6.1f}")
            print(f"    Steps: {step + 1}")
            print(f"    Final Distance: {distance:.2f}m")
            print(f"    Epsilon: {agent.epsilon:.3f}")
            print(f"    Success Rate: {success_count}/{episode+1}")
            
            if (episode + 1) % 5 == 0:
                agent.save()
                print(f"    💾 Progress saved!")
        
        env.stop()
        agent.save()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print(f"Total Successes: {success_count}/{episodes}")
        print(f"Success Rate: {success_count/episodes*100:.1f}%")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⛔ Stopped by user")
        env.stop()
        agent.save()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_rl_final(episodes=25)
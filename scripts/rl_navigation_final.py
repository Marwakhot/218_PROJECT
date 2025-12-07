#!/usr/bin/env python3
"""
Reinforcement Learning Navigation - Complete Training & Testing
Trains agent over multiple episodes and tracks performance metrics
"""
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import pickle
import os
from datetime import datetime

class QLearningAgent:
    
    def __init__(self, n_states=108, n_actions=5):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.epsilon = 0.9  # Start with high exploration
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        
        # Actions: [linear_velocity, angular_velocity]
        self.actions = [
            [0.5, 0.0],      # Forward fast
            [0.35, 0.7],     # Forward + Left turn
            [0.35, -0.7],    # Forward + Right turn
            [0.2, 0.8],      # Slow forward + Hard left
            [0.2, -0.8],     # Slow forward + Hard right
        ]
        
        self.action_names = [
            "FORWARD",
            "FWD+LEFT", 
            "FWD+RIGHT",
            "SLOW+LEFT",
            "SLOW+RIGHT"
        ]
        
        # Escape mechanism
        self.escaping = False
        self.escape_step = 0
        self.escape_turn_direction = 0
        
        # Stuck detection
        self.prev_position = None
        self.stuck_counter = 0
        self.position_history = []
    
    def discretize_state(self, lidar_data, distance_to_goal, angle_to_goal):
        """Convert continuous state to discrete state index"""
        # Front obstacle distance (4 bins)
        front_distance = np.mean(lidar_data[0:5])
        if front_distance < 0.5:
            distance_bin = 0  # Very close
        elif front_distance < 1.0:
            distance_bin = 1  # Close
        elif front_distance < 2.0:
            distance_bin = 2  # Medium
        else:
            distance_bin = 3  # Far
        
        # Angle to goal (3 bins)
        if angle_to_goal < -math.pi/4:
            angle_bin = 0  # Goal is to the left
        elif angle_to_goal < math.pi/4:
            angle_bin = 1  # Goal is straight ahead
        else:
            angle_bin = 2  # Goal is to the right
        
        # Distance to goal (3 bins)
        if distance_to_goal < 2.0:
            goal_dist_bin = 0  # Close to goal
        elif distance_to_goal < 5.0:
            goal_dist_bin = 1  # Medium distance
        else:
            goal_dist_bin = 2  # Far from goal
        
        # Side obstacles (3 bins for left and right)
        left_distance = np.mean(lidar_data[5:10])
        right_distance = np.mean(lidar_data[-10:-5])
        
        left_bin = 0 if left_distance < 0.8 else (1 if left_distance < 1.5 else 2)
        right_bin = 0 if right_distance < 0.8 else (1 if right_distance < 1.5 else 2)
        
        # State = distance_bin(4) * angle_bin(3) * goal_dist_bin(3) * left_bin(3) * right_bin(3)
        # Total states = 4 * 3 * 3 * 3 * 3 = 324... let's simplify
        
        # Simplified: distance(4) * angle(3) * goal_dist(3) * sides(3)
        # sides_bin: 0=both close, 1=one close, 2=both clear
        if left_distance < 0.8 and right_distance < 0.8:
            sides_bin = 0
        elif left_distance < 0.8 or right_distance < 0.8:
            sides_bin = 1
        else:
            sides_bin = 2
        
        state = (distance_bin * 27) + (angle_bin * 9) + (goal_dist_bin * 3) + sides_bin
        return min(state, self.n_states - 1)
    
    def choose_action(self, state, lidar_data, angle_to_goal, robot_x, robot_y):
        """Choose action using epsilon-greedy policy with escape mechanism"""
        front_distance = np.mean(lidar_data[0:5])
        
        # Update position history for stuck detection
        current_pos = (robot_x, robot_y)
        self.position_history.append(current_pos)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        # Check if stuck (position not changing much)
        is_stuck = False
        if len(self.position_history) >= 20:
            recent_positions = self.position_history[-20:]
            x_positions = [p[0] for p in recent_positions]
            y_positions = [p[1] for p in recent_positions]
            x_range = max(x_positions) - min(x_positions)
            y_range = max(y_positions) - min(y_positions)
            
            # If robot hasn't moved much in last 20 steps, it's stuck
            if x_range < 0.3 and y_range < 0.3:
                is_stuck = True
        
        # Trigger escape if stuck OR too close to obstacle
        if (front_distance < 0.6 or is_stuck) and not self.escaping:
            self.escaping = True
            self.escape_step = 0
            self.position_history = []  # Clear history
            
            # Smarter turn direction
            left_distance = np.mean(lidar_data[5:10])
            right_distance = np.mean(lidar_data[-10:-5])
            
            # Turn toward the side with more space
            if left_distance > right_distance:
                self.escape_turn_direction = 1.0  # Turn left
            else:
                self.escape_turn_direction = -1.0  # Turn right
            
            if is_stuck:
                print("     STUCK DETECTED - Starting aggressive escape")
            return None, [-0.4, 0.0], "ESCAPE_BACKUP"
        
        if self.escaping:
            self.escape_step += 1
            if self.escape_step <= 20:  # Longer backup
                return None, [-0.4, 0.0], "ESCAPE_BACKUP"
            elif self.escape_step <= 45:  # Longer turn (90 degrees)
                return None, [0.0, self.escape_turn_direction * 0.8], "ESCAPE_TURN"
            elif self.escape_step <= 70:  # Longer forward motion
                return None, [0.45, 0.0], "ESCAPE_FORWARD"
            else:
                self.escaping = False
                self.escape_step = 0
                print("    ✓ Escape complete")
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            action_idx = np.argmax(self.q_table[state])
        
        return action_idx, self.actions[action_idx], self.action_names[action_idx]
    
    def update(self, state, action_idx, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        if action_idx is None:
            return
        
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next]
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.learning_rate * td_error
    
    def reset_episode(self):
        """Reset agent state for new episode"""
        self.escaping = False
        self.escape_step = 0
        self.stuck_counter = 0
        self.position_history = []
    
    def decay_epsilon(self):
        """Decay exploration rate"""
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


class RobotEnv:
    
    def __init__(self):
        rospy.init_node('rl_navigation', anonymous=True)
        
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
        
        # Wait for sensors
        rate = rospy.Rate(10)
        timeout = 0
        while (not self.lidar_received or not self.odom_received) and timeout < 100:
            rate.sleep()
            timeout += 1
    
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
        siny = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_theta = math.atan2(siny, cosy)
    
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
        rospy.sleep(0.1)
    
    def calculate_reward(self, prev_distance):
        """Calculate reward based on current state"""
        distance = self.get_distance_to_goal()
        reward = 0.0
        done = False
        
        # Reward for progress toward goal
        if prev_distance is not None:
            progress = (prev_distance - distance)
            reward += progress * 25.0  # High reward for getting closer
        
        # Large reward for reaching goal - FIXED THRESHOLD
        if distance < 0.75:  # Slightly tighter than 0.8 to ensure detection
            reward += 500.0
            done = True
            return reward, done, distance
        
        # Penalty for collision
        min_distance = np.min(self.lidar_data)
        if min_distance < 0.25:
            reward -= 100.0
            done = True
            return reward, done, distance
        
        # Small penalty for being too close to obstacles
        if min_distance < 0.5:
            reward -= 2.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.3
        
        return reward, done, distance
    
    def stop(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)


def train_agent(num_episodes=30, max_steps_per_episode=400):
    """Train the RL agent over multiple episodes"""
    
    print("\n" + "=" * 70)
    print("REINFORCEMENT LEARNING ROBOT NAVIGATION - TRAINING MODE")
    print("=" * 70)
    
    env = RobotEnv()
    
    if not env.lidar_received or not env.odom_received:
        print("ERROR: Sensors not connected!")
        return
    
    agent = QLearningAgent()
    
    # Try to load existing Q-table
    if agent.load():
        print("Loaded existing Q-table")
    else:
        print("Starting with new Q-table")
    
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Initial epsilon: {agent.epsilon:.2f}")
    print(f"  Start position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
    print(f"  Goal position: ({env.goal_x}, {env.goal_y})")
    print("=" * 70 + "\n")
    
    # Track performance metrics
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            print(f"\n{'='*70}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*70}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            
            # Reset agent state
            agent.reset_episode()
            
            # Initial turn if facing wall (only first episode)
            if episode == 0:
                env.stop()
                rospy.sleep(0.5)
                initial_front = np.mean(env.lidar_data[:5])
                if initial_front < 1.5:
                    print("Performing initial 180° turn...")
                    for _ in range(50):
                        cmd = Twist()
                        cmd.angular.z = 0.8
                        env.cmd_vel_pub.publish(cmd)
                        rospy.sleep(0.1)
                    env.stop()
                    rospy.sleep(0.3)
                    print("Initial turn complete\n")
            
            # Episode variables
            distance = env.get_distance_to_goal()
            angle = env.get_angle_to_goal()
            state = agent.discretize_state(env.lidar_data, distance, angle)
            
            total_reward = 0
            prev_distance = distance
            
            print(f"Starting distance: {distance:.2f}m\n")
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Choose and execute action
                angle = env.get_angle_to_goal()
                action_idx, action, action_name = agent.choose_action(
                    state, env.lidar_data, angle, env.robot_x, env.robot_y
                )
                env.execute_action(action)
                
                # Calculate reward and check if done
                reward, done, distance = env.calculate_reward(prev_distance)
                total_reward += reward
                
                # Get next state
                angle = env.get_angle_to_goal()
                next_state = agent.discretize_state(env.lidar_data, distance, angle)
                
                # Update Q-table
                agent.update(state, action_idx, reward, next_state)
                
                # Move to next state
                state = next_state
                prev_distance = distance
                
                # Print progress every 40 steps
                if step % 40 == 0 and not agent.escaping:
                    min_lidar = np.min(env.lidar_data)
                    print(f"  Step {step:3d}: {action_name:15s} | Distance={distance:.2f}m | "
                          f"Front={min_lidar:.2f}m | Reward={total_reward:7.1f}")
                
                # Check if episode is done
                if done:
                    if distance < 0.75:
                        print(f"\n✓ GOAL REACHED!")
                        print(f"  Final position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
                        print(f"  Steps taken: {step + 1}")
                        print(f"  Total reward: {total_reward:.1f}")
                        success_count += 1
                    else:
                        print(f"\n✗ Collision detected at step {step + 1}")
                        print(f"  Total reward: {total_reward:.1f}")
                    break
            
            if not done:
                print(f"\n⏱ Max steps reached")
                print(f"  Final distance: {distance:.2f}m")
                print(f"  Total reward: {total_reward:.1f}")
            
            # Record metrics
            episode_rewards.append(total_reward)
            episode_steps.append(step + 1)
            
            # Decay epsilon
            agent.decay_epsilon()
            
            # Stop robot
            env.stop()
            rospy.sleep(0.5)
            
            # Print episode summary
            print(f"\n  Episode Summary:")
            print(f"    Total Reward: {total_reward:.1f}")
            print(f"    Steps: {step + 1}")
            print(f"    Success Rate: {success_count}/{episode + 1} "
                  f"({100*success_count/(episode+1):.1f}%)")
            
            # Save Q-table periodically
            if (episode + 1) % 5 == 0:
                agent.save()
                print(f"     Q-table saved")
        
        # Final save
        env.stop()
        agent.save()
        
        # Print training summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nResults:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Successful episodes: {success_count}")
        print(f"  Success rate: {100*success_count/num_episodes:.1f}%")
        print(f"  Average reward: {np.mean(episode_rewards):.1f}")
        print(f"  Average steps: {np.mean(episode_steps):.1f}")
        print(f"  Final epsilon: {agent.epsilon:.3f}")
        
        # Show improvement over time
        if num_episodes >= 10:
            first_half_rewards = np.mean(episode_rewards[:num_episodes//2])
            second_half_rewards = np.mean(episode_rewards[num_episodes//2:])
            print(f"\nLearning Progress:")
            print(f"  First half avg reward: {first_half_rewards:.1f}")
            print(f"  Second half avg reward: {second_half_rewards:.1f}")
            print(f"  Improvement: {second_half_rewards - first_half_rewards:+.1f}")
        
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        env.stop()
        agent.save()
        print("Q-table saved")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        env.stop()


def test_agent(num_test_runs=3):
    """Test the trained agent"""
    
    print("\n" + "=" * 70)
    print("REINFORCEMENT LEARNING ROBOT NAVIGATION - TEST MODE")
    print("=" * 70)
    
    env = RobotEnv()
    
    if not env.lidar_received or not env.odom_received:
        print("ERROR: Sensors not connected!")
        return
    
    agent = QLearningAgent()
    
    if not agent.load():
        print("ERROR: No trained Q-table found! Please train first.")
        return
    
    agent.epsilon = 0.0  # No exploration during testing
    
    print(f"\nTesting trained agent (epsilon=0.0)")
    print(f"Number of test runs: {num_test_runs}")
    print(f"Start: ({env.robot_x:.2f}, {env.robot_y:.2f})")
    print(f"Goal: ({env.goal_x}, {env.goal_y})")
    print("=" * 70 + "\n")
    
    success_count = 0
    
    try:
        for run in range(num_test_runs):
            print(f"\n{'='*70}")
            print(f"TEST RUN {run + 1}/{num_test_runs}")
            print(f"{'='*70}\n")
            
            # Reset agent state
            agent.reset_episode()
            
            # Initial setup
            env.stop()
            rospy.sleep(0.5)
            
            if run == 0:
                initial_front = np.mean(env.lidar_data[:5])
                if initial_front < 1.5:
                    print("Turning away from wall...")
                    for _ in range(50):
                        cmd = Twist()
                        cmd.angular.z = 0.8
                        env.cmd_vel_pub.publish(cmd)
                        rospy.sleep(0.1)
                    env.stop()
                    rospy.sleep(0.3)
            
            distance = env.get_distance_to_goal()
            angle = env.get_angle_to_goal()
            state = agent.discretize_state(env.lidar_data, distance, angle)
            
            total_reward = 0
            prev_distance = distance
            
            print(f"Starting navigation (distance: {distance:.2f}m)\n")
            
            for step in range(400):
                angle = env.get_angle_to_goal()
                action_idx, action, action_name = agent.choose_action(
                    state, env.lidar_data, angle, env.robot_x, env.robot_y
                )
                
                env.execute_action(action)
                
                reward, done, distance = env.calculate_reward(prev_distance)
                total_reward += reward
                
                angle = env.get_angle_to_goal()
                next_state = agent.discretize_state(env.lidar_data, distance, angle)
                
                state = next_state
                prev_distance = distance
                
                if step % 30 == 0:
                    min_lidar = np.min(env.lidar_data)
                    print(f"Step {step:3d}: {action_name:15s} | Dist={distance:.2f}m | "
                          f"Front={min_lidar:.2f}m | Reward={total_reward:6.1f}")
                
                if done:
                    if distance < 0.75:
                        print(f"\n GOAL REACHED!")
                        print(f"   Position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
                        print(f"   Steps: {step + 1}")
                        print(f"   Total reward: {total_reward:.1f}")
                        success_count += 1
                    else:
                        print(f"\n Collision at step {step + 1}")
                    break
            
            if not done:
                print(f"\n⏱ Max steps reached")
                print(f"   Final distance: {distance:.2f}m")
            
            env.stop()
            rospy.sleep(1.0)
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print(f"\nSuccess rate: {success_count}/{num_test_runs} "
              f"({100*success_count/num_test_runs:.1f}%)")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted")
        env.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        num_tests = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        test_agent(num_test_runs=num_tests)
    else:
        # Training mode (default)
        num_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
        train_agent(num_episodes=num_eps)
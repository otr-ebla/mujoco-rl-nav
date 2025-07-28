import pygame
import numpy as np
import sys
import os

# Add the directory containing the environment to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment (assuming it's saved as robot_env.py)
from env4walls import RobotNavigationEnv

class KeyboardController:
    def __init__(self):
        self.env = RobotNavigationEnv(render_mode='human')
        self.running = True
        
        # Control parameters
        self.max_linear_speed = 0.01
        self.max_angular_speed = 0.5
        self.speed_step = 0.1
        
        # Current velocities
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        
        # Lidar printing control
        self.print_lidar = False
        
        # Key states
        self.keys_pressed = {
            'w': False,
            'a': False,
            's': False,
            'd': False
        }
        
        print("Keyboard Controls:")
        print("W - Move forward")
        print("S - Move backward")
        print("A - Turn left")
        print("D - Turn right")
        print("L - Toggle lidar visualization")
        print("P - Print lidar measurements")
        print("R - Reset environment")
        print("ESC - Exit")
        print("Space - Stop robot")
        print("\nRobot will accelerate/decelerate smoothly based on key presses.")
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_w:
                    self.keys_pressed['w'] = True
                elif event.key == pygame.K_s:
                    self.keys_pressed['s'] = True
                elif event.key == pygame.K_a:
                    self.keys_pressed['a'] = True
                elif event.key == pygame.K_d:
                    self.keys_pressed['d'] = True
                elif event.key == pygame.K_l:
                    self.env.toggle_lidar_visualization()
                elif event.key == pygame.K_p:
                    self.print_lidar = not self.print_lidar
                    status = "ON" if self.print_lidar else "OFF"
                    print(f"Lidar printing: {status}")
                elif event.key == pygame.K_r:
                    self.reset_environment()
                elif event.key == pygame.K_SPACE:
                    self.stop_robot()
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.keys_pressed['w'] = False
                elif event.key == pygame.K_s:
                    self.keys_pressed['s'] = False
                elif event.key == pygame.K_a:
                    self.keys_pressed['a'] = False
                elif event.key == pygame.K_d:
                    self.keys_pressed['d'] = False
    
    def update_velocities(self):
        """Update robot velocities based on key states"""
        # Linear velocity control
        if self.keys_pressed['w']:
            self.linear_vel = min(self.linear_vel + self.speed_step, self.max_linear_speed)
        elif self.keys_pressed['s']:
            self.linear_vel = max(self.linear_vel - self.speed_step, 0.0)
        else:
            # Gradual deceleration when no key is pressed
            self.linear_vel = max(self.linear_vel - self.speed_step * 0.5, 0.0)
        
        # Angular velocity control
        if self.keys_pressed['d']:
            self.angular_vel = min(self.angular_vel + self.speed_step, self.max_angular_speed)
        elif self.keys_pressed['a']:
            self.angular_vel = max(self.angular_vel - self.speed_step, -self.max_angular_speed)
        else:
            # Gradual deceleration when no key is pressed
            if self.angular_vel > 0:
                self.angular_vel = max(self.angular_vel - self.speed_step * 0.5, 0.0)
            elif self.angular_vel < 0:
                self.angular_vel = min(self.angular_vel + self.speed_step * 0.5, 0.0)
    
    def stop_robot(self):
        """Stop the robot immediately"""
        self.linear_vel = 0.0
        self.angular_vel = 0.0
    
    def reset_environment(self):
        """Reset the environment"""
        self.env.reset()
        self.stop_robot()
        print("Environment reset!")
    
    def print_lidar_data(self, observation, info):
        """Print lidar measurements in a readable format"""
        # Print summary statistics
        min_dist = np.min(observation)
        max_dist = np.max(observation)
        avg_dist = np.mean(observation)
        
        print(f"\n--- Lidar Measurements ---")
        print(f"Min distance: {min_dist:.2f}")
        print(f"Max distance: {max_dist:.2f}")
        print(f"Avg distance: {avg_dist:.2f}")
        
        # Print directional readings (8 directions)
        directions = ['Front', 'Front-Right', 'Right', 'Back-Right', 
                     'Back', 'Back-Left', 'Left', 'Front-Left']
        
        print("\nDirectional readings:")
        for i, direction in enumerate(directions):
            # Get reading for this direction (45 degree increments)
            angle_idx = i * 45  # 0, 45, 90, 135, 180, 225, 270, 315 degrees
            distance = observation[angle_idx]
            print(f"{direction:12}: {distance:6.2f}")
        
        # Print ranges with obstacle counts
        ranges = [(0, 20, "Very Close"), (20, 50, "Close"), 
                 (50, 100, "Medium"), (100, 200, "Far")]
        
        print("\nDistance distribution:")
        for min_r, max_r, label in ranges:
            count = np.sum((observation >= min_r) & (observation < max_r))
            percentage = (count / len(observation)) * 100
            print(f"{label:12}: {count:3d} rays ({percentage:5.1f}%)")
        
        # Print closest obstacles (top 5)
        sorted_indices = np.argsort(observation)[:5]
        print("\nClosest obstacles:")
        for i, idx in enumerate(sorted_indices):
            angle_deg = idx  # Since we have 1 degree per ray
            distance = observation[idx]
            print(f"{i+1}. Angle {angle_deg:3d}°: {distance:.2f}")
        
        print("-" * 30)
    
    def run(self):
        """Main game loop"""
        # Initialize environment
        observation, info = self.env.reset()
        
        try:
            while self.running:
                # Handle events
                self.handle_events()
                
                # Update velocities
                self.update_velocities()
                
                # Create action array
                action = np.array([self.linear_vel, self.angular_vel], dtype=np.float32)
                
                # Step environment
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Print lidar measurements if enabled
                if self.print_lidar:
                    self.print_lidar_data(observation, info)
                
                # Render environment
                self.env.render()
                
                # Check if episode ended
                if terminated or truncated:
                    robot_state = self.env.get_robot_state()
                    
                    if info.get('target_reached', False):
                        print(f"🎉 Target reached! Distance: {robot_state['distance_to_target']:.2f}")
                    elif info.get('collision', False):
                        print(f"💥 Collision detected! Min distance: {info['min_lidar_distance']:.3f}")
                    elif truncated:
                        print(f"⏰ Time limit reached. Distance to target: {robot_state['distance_to_target']:.2f}")
                    
                    # Reset environment after a short delay
                    pygame.time.wait(1000)
                    self.reset_environment()
                
                # Display current state information
                robot_state = self.env.get_robot_state()
                min_dist = info.get('min_lidar_distance', 0)
                
                # Color-coded distance warning
                if min_dist < 5:
                    status_color = "🔴"
                elif min_dist < 15:
                    status_color = "🟡"
                else:
                    status_color = "🟢"
                
                pygame.display.set_caption(
                    f"Robot Navigation {status_color} - Target: {robot_state['distance_to_target']:.1f} "
                    f"Min: {min_dist:.1f} - Lin: {self.linear_vel:.2f}, Ang: {self.angular_vel:.2f}"
                )
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.env.close()
            pygame.quit()

if __name__ == "__main__":
    # Check if pygame is available
    try:
        import pygame
        pygame.init()
    except ImportError:
        print("Error: pygame is not installed. Please install it with: pip install pygame")
        sys.exit(1)
    
    # Check if gymnasium is available
    try:
        import gymnasium as gym
    except ImportError:
        print("Error: gymnasium is not installed. Please install it with: pip install gymnasium")
        sys.exit(1)
    
    # Create and run the controller
    controller = KeyboardController()
    controller.run()
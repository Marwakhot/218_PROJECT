import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyNavigator:
    def __init__(self):
        print("Initializing Fuzzy Navigation System...")
        
        # Escape state machine
        self.escaping = False
        self.escape_phase = 0
        self.escape_counter = 0
        self.stuck_counter = 0
        self.turn_direction = 1.0  # Default left
        self.last_turn_direction = None  # Track last turn direction
        
        # Initial turn flag
        self.needs_initial_turn = True
        self.initial_turn_counter = 0
        
        # Stuck detection - IMPROVED
        self.last_front = 10.0
        self.position_stuck_counter = 0
        self.long_turn = False
        self.very_close_counter = 0  # NEW: Track how long we're very close to obstacle
        
        # Input variables
        self.front = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'front')
        self.heading = ctrl.Antecedent(np.arange(-180, 181, 1), 'heading')

        # Output variables
        self.speed = ctrl.Consequent(np.arange(0, 0.5, 0.01), 'speed', defuzzify_method='centroid')
        self.turn = ctrl.Consequent(np.arange(-1.0, 1.01, 0.01), 'turn', defuzzify_method='centroid')

        # Distance membership functions
        self.front['close'] = fuzz.trimf(self.front.universe, [0, 0, 0.8])
        self.front['medium'] = fuzz.trimf(self.front.universe, [0.5, 1.2, 2.0])
        self.front['far'] = fuzz.trimf(self.front.universe, [1.5, 5.0, 5.0])

        # Heading membership functions
        self.heading['hard_left'] = fuzz.trapmf(self.heading.universe, [-180, -180, -60, -30])
        self.heading['left'] = fuzz.trimf(self.heading.universe, [-60, -20, -5])
        self.heading['straight'] = fuzz.trimf(self.heading.universe, [-10, 0, 10])
        self.heading['right'] = fuzz.trimf(self.heading.universe, [5, 20, 60])
        self.heading['hard_right'] = fuzz.trapmf(self.heading.universe, [30, 60, 180, 180])

        # Speed outputs
        self.speed['slow'] = fuzz.trimf(self.speed.universe, [0.2, 0.25, 0.3])
        self.speed['medium'] = fuzz.trimf(self.speed.universe, [0.25, 0.35, 0.4])
        self.speed['fast'] = fuzz.trimf(self.speed.universe, [0.35, 0.45, 0.5])

        # Turn outputs
        self.turn['hard_left'] = fuzz.trimf(self.turn.universe, [-1.0, -0.8, -0.5])
        self.turn['left'] = fuzz.trimf(self.turn.universe, [-0.6, -0.4, -0.1])
        self.turn['straight'] = fuzz.trimf(self.turn.universe, [-0.15, 0, 0.15])
        self.turn['right'] = fuzz.trimf(self.turn.universe, [0.1, 0.4, 0.6])
        self.turn['hard_right'] = fuzz.trimf(self.turn.universe, [0.5, 0.8, 1.0])

        # Fuzzy Rules
        rule1 = ctrl.Rule(self.front['close'] & self.heading['hard_left'], 
                         [self.speed['slow'], self.turn['hard_left']])
        rule2 = ctrl.Rule(self.front['close'] & self.heading['left'], 
                         [self.speed['slow'], self.turn['hard_left']])
        rule3 = ctrl.Rule(self.front['close'] & self.heading['straight'], 
                         [self.speed['slow'], self.turn['hard_left']])
        rule4 = ctrl.Rule(self.front['close'] & self.heading['right'], 
                         [self.speed['slow'], self.turn['hard_right']])
        rule5 = ctrl.Rule(self.front['close'] & self.heading['hard_right'], 
                         [self.speed['slow'], self.turn['hard_right']])
        
        rule6 = ctrl.Rule(self.front['medium'] & self.heading['hard_left'], 
                         [self.speed['medium'], self.turn['hard_left']])
        rule7 = ctrl.Rule(self.front['medium'] & self.heading['left'], 
                         [self.speed['medium'], self.turn['left']])
        rule8 = ctrl.Rule(self.front['medium'] & self.heading['straight'], 
                         [self.speed['medium'], self.turn['straight']])
        rule9 = ctrl.Rule(self.front['medium'] & self.heading['right'], 
                         [self.speed['medium'], self.turn['right']])
        rule10 = ctrl.Rule(self.front['medium'] & self.heading['hard_right'], 
                          [self.speed['medium'], self.turn['hard_right']])
        
        rule11 = ctrl.Rule(self.front['far'] & self.heading['hard_left'], 
                          [self.speed['fast'], self.turn['hard_left']])
        rule12 = ctrl.Rule(self.front['far'] & self.heading['left'], 
                          [self.speed['fast'], self.turn['left']])
        rule13 = ctrl.Rule(self.front['far'] & self.heading['straight'], 
                          [self.speed['fast'], self.turn['straight']])
        rule14 = ctrl.Rule(self.front['far'] & self.heading['right'], 
                          [self.speed['fast'], self.turn['right']])
        rule15 = ctrl.Rule(self.front['far'] & self.heading['hard_right'], 
                          [self.speed['fast'], self.turn['hard_right']])

        self.system = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
            rule11, rule12, rule13, rule14, rule15
        ])
        self.sim = ctrl.ControlSystemSimulation(self.system)
        
        print("Fuzzy system initialized!")

    def get_action(self, f, l, r, h):
        """
        Navigation with: Initial turn → Normal nav → Escape when stuck
        """
        
        # INITIAL TURN - Do this once at the start
        if self.needs_initial_turn:
            self.initial_turn_counter += 1
            
            # Check if robot is facing obstacle at start
            if f < 1.5 and self.initial_turn_counter < 5:
                # Still checking...
                return 0.0, 0.0
            
            if f < 1.5:
                # Yes, facing obstacle - need to turn around
                if self.initial_turn_counter < 50:  # Turn for 5 seconds (180 degrees)
                    if self.initial_turn_counter % 10 == 5:
                        print(f" Initial turn to face away from wall ({self.initial_turn_counter}/50)...")
                    return 0.0, 0.8  # Turn in place
                else:
                    # Done with initial turn
                    self.needs_initial_turn = False
                    print("✓ Initial turn complete! Starting navigation...\n")
            else:
                # Path is clear, no initial turn needed
                self.needs_initial_turn = False
                print("✓ Path clear! Starting navigation...\n")
        
        # IMPROVED STUCK DETECTION
        if f < 0.7 and not self.escaping:
            self.stuck_counter += 1
            
            # Track if we're VERY close (less than 0.5m)
            if f < 0.5:
                self.very_close_counter += 1
            else:
                self.very_close_counter = 0
            
            # Check if position is actually stuck (front distance not changing much)
            if abs(f - self.last_front) < 0.08:  # Increased threshold slightly
                self.position_stuck_counter += 1
            else:
                self.position_stuck_counter = 0
            
            self.last_front = f
            
            # TRIGGER ESCAPE if any of these conditions:
            # 1. Stuck counter > 3 (getting close repeatedly)
            # 2. Position truly stuck (not moving away from obstacle)
            # 3. Very close for multiple iterations (immediate danger)
            if self.stuck_counter > 3 or self.position_stuck_counter > 4 or self.very_close_counter > 3:
                # START ESCAPE SEQUENCE
                self.escaping = True
                self.escape_phase = 0
                self.escape_counter = 0
                
                # If truly stuck or very close, do LONGER turn
                if self.position_stuck_counter > 4 or self.very_close_counter > 3:
                    self.long_turn = True
                    print(f"\n🆘 WEDGED/VERY CLOSE! Starting LONG escape sequence...")
                else:
                    self.long_turn = False
                    print(f"\n🆘 OBSTACLE DETECTED! Starting escape sequence...")
                
                self.position_stuck_counter = 0
                self.very_close_counter = 0
        else:
            if not self.escaping:
                # Only reset counters if we're far enough away
                if f > 1.0:
                    self.stuck_counter = 0
                    self.very_close_counter = 0
                self.last_front = f
        
        # EXECUTE ESCAPE SEQUENCE
        if self.escaping:
            self.escape_counter += 1
            
            # PHASE 0: Back up HARD (30 steps = 3 seconds)
            if self.escape_phase == 0:
                if self.escape_counter < 30:
                    if self.escape_counter % 10 == 0:
                        print(f"  Phase 1: Backing up ({self.escape_counter}/30)")
                    return -0.5, 0.0  # FASTER backup
                else:
                    self.escape_phase = 1
                    self.escape_counter = 0
                    
                    # SMART DIRECTION CHOICE:
                    # If we just escaped recently and hit obstacle again, turn OPPOSITE direction
                    if self.last_turn_direction is not None:
                        # Turn opposite to last time
                        self.turn_direction = -self.last_turn_direction
                        print(f"  Switching direction from last escape!")
                    else:
                        # First escape - choose based on which side has MORE space
                        self.turn_direction = 1.0 if l > r else -1.0
                    
                    # Remember this direction for next time
                    self.last_turn_direction = self.turn_direction
                    
                    direction = "LEFT" if self.turn_direction > 0 else "RIGHT"
                    print(f"  Phase 2: Turning {direction}...")
            
            # PHASE 1: Turn (duration depends on how stuck)
            if self.escape_phase == 1:
                turn_duration = 25 if self.long_turn else 8  # Long turn = 90°, short = 20°
                if self.escape_counter < turn_duration:
                    if self.escape_counter % 5 == 0:
                        print(f"  Turning ({self.escape_counter}/{turn_duration})...")
                    return 0.0, self.turn_direction * 0.8  # Turn in place
                else:
                    self.escape_phase = 2
                    self.escape_counter = 0
                    print("  Phase 3: Moving forward...")
            
            # PHASE 2: Move forward (50 steps = 5 seconds)
            if self.escape_phase == 2:
                if self.escape_counter < 50:
                    if self.escape_counter % 10 == 0:
                        print(f"  Moving forward ({self.escape_counter}/50)...")
                    return 0.45, 0.0  # FASTER forward
                else:
                    # DONE! Resume normal navigation
                    self.escaping = False
                    self.escape_phase = 0
                    self.escape_counter = 0
                    self.stuck_counter = 0
                    self.long_turn = False  # Reset this flag
                    print("  ✓ Escape complete! Resuming...\n")
                    # Keep last_turn_direction in memory for next escape
        
        # NORMAL NAVIGATION
        speed = 0.35
        turn = 0.0
        
        # Correct heading toward goal
        if abs(h) > 20:
            turn = 0.5 if h > 0 else -0.5
        elif abs(h) > 10:
            turn = 0.3 if h > 0 else -0.3
        elif abs(h) > 5:
            turn = 0.15 if h > 0 else -0.15
        
        # Gentle obstacle avoidance during normal navigation
        if f < 1.2 and f > 0.7:
            speed = 0.3
            if l < r:
                turn -= 0.2
            else:
                turn += 0.2
        
        return speed, turn
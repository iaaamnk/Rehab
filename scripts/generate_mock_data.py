import pandas as pd
import numpy as np
import os

# Kinect V2 standard has 25 joints. We'll simulate 25 joints * 3 coordinates (x, y, z)
NUM_JOINTS = 25
NUM_FRAMES_PER_SAMPLE = 100
FPS = 30

def generate_base_skeleton():
    """Generates a default standing skeleton pose (approximate real-world coordinates in meters)."""
    skeleton = np.zeros((NUM_JOINTS, 3))
    # Approximation of a standing human.
    # 0: SpineBase, 1: SpineMid, 2: Neck, 3: Head
    # 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
    # 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
    
    skeleton[0] = [0.0, 0.0, 2.0]        # Spine Base
    skeleton[1] = [0.0, 0.3, 2.0]        # Spine Mid
    skeleton[2] = [0.0, 0.6, 2.0]        # Neck
    skeleton[8] = [0.2, 0.5, 2.0]        # Right Shoulder
    skeleton[9] = [0.3, 0.2, 2.0]        # Right Elbow (hanging down)
    skeleton[10] = [0.3, -0.1, 2.0]      # Right Wrist (hanging down)
    
    return skeleton

def generate_movement(is_compensatory=False, num_frames=NUM_FRAMES_PER_SAMPLE):
    """
    Simulates a Right Shoulder Abduction (raising the arm to the side).
    If is_compensatory is True, adds excessive trunk lean (SpineMid/Neck moving to the left).
    """
    skeleton = generate_base_skeleton()
    frames = []
    
    # Simulating a sinusoidal motion up and down
    t = np.linspace(0, np.pi, num_frames)
    
    for val in t:
        current_pose = skeleton.copy()
        
        # 1. Healthy arm motion: Arm raises up to 90 degrees
        arm_raise_factor = np.sin(val) # 0 to 1 to 0
        
        # Right wrist moves up and out
        current_pose[10, 0] += arm_raise_factor * 0.5  # Move outwards (+x)
        current_pose[10, 1] += arm_raise_factor * 0.6  # Move upwards (+y)
        
        # Right elbow bends/moves slightly
        current_pose[9, 0] += arm_raise_factor * 0.25
        current_pose[9, 1] += arm_raise_factor * 0.3
        
        # 2. Compensatory motion (Trunk Sway)
        if is_compensatory:
            trunk_lean_factor = arm_raise_factor * 0.2 # 0.2 meters sway to the left
            current_pose[1, 0] -= trunk_lean_factor # Spine mid moves left (-x)
            current_pose[2, 0] -= trunk_lean_factor # Neck moves left
            current_pose[8, 0] -= trunk_lean_factor # Right shoulder drops/moves left
            current_pose[4, 0] -= trunk_lean_factor # Left shoulder moves left (index 4)
            
        # Add a tiny bit of Gaussian noise for realism
        noise = np.random.normal(0, 0.005, current_pose.shape)
        current_pose += noise
        
        frames.append(current_pose.flatten())
        
    return frames

def main():
    print("Generating simulated dataset...")
    
    # Define column names: j0_x, j0_y, j0_z, j1_x, ...
    columns = ['frame', 'label']
    for i in range(NUM_JOINTS):
        columns.extend([f'j{i}_x', f'j{i}_y', f'j{i}_z'])
        
    dataset = []
    global_frame = 0
    
    # Generate 50 healthy repetitions
    for _ in range(50):
        frames = generate_movement(is_compensatory=False)
        for frame_data in frames:
            row = [global_frame, 0] + list(frame_data) # label 0 = healthy
            dataset.append(row)
            global_frame += 1
            
    # Generate 50 compensatory repetitions
    for _ in range(50):
        frames = generate_movement(is_compensatory=True)
        for frame_data in frames:
            row = [global_frame, 1] + list(frame_data) # label 1 = compensatory
            dataset.append(row)
            global_frame += 1
            
    # Save to CSV
    df = pd.DataFrame(dataset, columns=columns)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'simulated_poses.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated dataset with {len(df)} frames and {len(columns)} columns.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()

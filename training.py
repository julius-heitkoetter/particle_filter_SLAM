import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import os

import tensorflow as tf
from tensorflow.keras import layers

# Constants
NUM_PARTICLES = 100
MAP_SIZE = (20, 20)  # Maze size (grid of 20x20)
FORWARD_MOVE_NOISE = 0.1
ROTATE_MOVE_NOISE = 0.05
TOUCH_SENSOR_NOISE = 0.05
SENSOR_RANGE = 1  # Touch sensor can detect walls 1 unit away
THETA_THRESHOLD = 0.5  # Maximum turn performed in one time step
FORWARD_THRESHOLD = 0.5  # Maximum movement forward in one time step.

NUM_OF_STEPS_BEFORE_RESAMPLE = 10 # we should only resample every X number of steps


class Map:
    """Class to handle map representation and coordinate transformations."""
    
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros(size)  # 0 = open space, 1 = wall

    def add_wall_vertical(self, start, end, col):
        """Add a vertical wall from (start, col) to (end, col)."""
        self.grid[start:end, col] = 1

    def add_wall_horizontal(self, row, start, end):
        """Add a horizontal wall from (row, start) to (row, end)."""
        self.grid[row, start:end] = 1

    def coord_to_index(self, x, y):
        """
        Convert robot coordinates (with (0,0) at center) to map indices.
        Positive x is to the right, positive y is upward.
        """
        half_x = self.size[0] // 2
        half_y = self.size[1] // 2
        # Flip y to match matplotlib's y-axis
        map_x = int(np.clip(x + half_x, 0, self.size[0] - 1))
        map_y = int(np.clip(half_y - y, 0, self.size[1] - 1))
        return map_x, map_y

    def index_to_coord(self, map_x, map_y):
        """Convert map indices to robot coordinates."""
        half_x = self.size[0] // 2
        half_y = self.size[1] // 2
        x = map_x - half_x
        y = half_y - map_y  # Flip y to match robot's upward direction
        return x, y

    def is_wall(self, x, y):
        """Check if the given robot coordinates have a wall."""
        map_x, map_y = self.coord_to_index(x, y)
        return self.grid[map_y, map_x] == 1


class RobotBase:
    """Base class to handle movement logic shared between real robot and particles."""
    
    def __init__(self, map_obj):
        self.map = map_obj
        self.x = 0.0  # Start at (0,0)
        self.y = 0.0
        self.theta = np.pi / 2  # Facing upward (positive y direction)

    def move(self, move_dist, move_theta):

        # Add noise to rotation
        self.theta += move_theta + np.random.normal(0, ROTATE_MOVE_NOISE)
        self.theta %= 2 * np.pi  # Keep theta within [0, 2π]

        # Add noise to the distance moved
        noisy_move_dist = move_dist + np.random.normal(0, FORWARD_MOVE_NOISE)

        # Update the robot's position based on its new heading (theta)
        self.x += noisy_move_dist * np.cos(self.theta)
        self.y += noisy_move_dist * np.sin(self.theta)

        # Keep within map bounds
        half_x = self.map.size[0] // 2
        half_y = self.map.size[1] // 2
        self.x = np.clip(self.x, -half_x + 0.5, half_x - 0.5)
        self.y = np.clip(self.y, -half_y + 0.5, half_y - 0.5)

    def get_position(self):
        """Return the robot's current position and orientation."""
        return self.x, self.y, self.theta


class Particle(RobotBase):
    """Particle class representing a possible state of the robot and the map estimate."""
    
    def __init__(self, map_obj):
        super().__init__(map_obj)
        self.map_estimate = np.zeros(map_obj.size)  # Initial unknown map

    def update_map(self, measurement):
        """Update the particle's map with the new measurement."""
        sensor_x = self.x + SENSOR_RANGE * np.cos(self.theta)
        sensor_y = self.y + SENSOR_RANGE * np.sin(self.theta)
        map_x, map_y = self.map.coord_to_index(sensor_x, sensor_y)

        # Update the map estimate with the measurement
        self.map_estimate[map_y, map_x] = measurement


class RealRobot(RobotBase):
    """RealRobot class that keeps track of the real robot's position."""
    
    def __init__(self, map_obj):
        super().__init__(map_obj)  # Inherit movement logic

    def sense_touch(self):
        """Simulate touch sensor: detect if there's a wall in front of the robot."""
        sensor_x = self.x + SENSOR_RANGE * np.cos(self.theta)
        sensor_y = self.y + SENSOR_RANGE * np.sin(self.theta)
        return 1 if self.map.is_wall(sensor_x, sensor_y) else 0

    def move(self, move_dist, move_theta):
        if abs(move_dist) > FORWARD_THRESHOLD:
            raise ValueError(f"move_dist {move_dist} exceeds FORWARD_THRESHOLD {FORWARD_THRESHOLD}")
        if abs(move_theta) > THETA_THRESHOLD:
            raise ValueError(f"move_theta {move_theta} exceeds THETA_THRESHOLD {THETA_THRESHOLD}")

        previous_theta = self.theta
        previous_x = self.x
        previous_y = self.y

        # Rotate the robot (with noise)
        self.theta += move_theta + np.random.normal(0, ROTATE_MOVE_NOISE)
        self.theta %= 2 * np.pi  # Keep theta within [0, 2π]

        # Simulate new position after moving
        new_x = self.x + (move_dist + np.random.normal(0, FORWARD_MOVE_NOISE)) * np.cos(self.theta)
        new_y = self.y + (move_dist + np.random.normal(0, FORWARD_MOVE_NOISE)) * np.sin(self.theta)

        # Check if the new position would place the robot inside a wall
        if self.map.is_wall(new_x, new_y):

            # Find the position where the robot would be just touching the wall
            # Binary search or step reduction method to find the point where the robot hits the wall
            touching_x, touching_y = self.find_touching_point(self.x, self.y, new_x, new_y)

            # Move the robot to the touching point
            self.x, self.y = touching_x, touching_y
        else:
            # No obstacle, proceed to the new position
            self.x, self.y = new_x, new_y
        

        # Keep within map bounds (boundary check)
        half_x = self.map.size[0] // 2
        half_y = self.map.size[1] // 2
        self.x = np.clip(self.x, -half_x + 0.5, half_x - 0.5)
        self.y = np.clip(self.y, -half_y + 0.5, half_y - 0.5)

        real_move_dist = np.sqrt((self.x - previous_x)**2  + (self.y - previous_y)**2 )
        real_move_theta = self.theta - previous_theta

        return real_move_dist, real_move_theta

    def find_touching_point(self, start_x, start_y, end_x, end_y):
        """
        Perform a step reduction or binary search along the path to find the point where
        the robot just touches the obstacle.
        """
        step_fraction = 0.1  # Start with a step size, reduce it until precise
        for i in range(10):  # Limit the number of refinement steps
            mid_x = start_x + step_fraction * (end_x - start_x)
            mid_y = start_y + step_fraction * (end_y - start_y)

            # Check if the midpoint is in a wall
            if self.map.is_wall(mid_x, mid_y):
                # If inside the wall, backtrack
                step_fraction -= step_fraction / 2
            else:
                # Move forward closer to the obstacle
                step_fraction += step_fraction / 2

        # Return the point where the robot touches the wall
        return start_x + step_fraction * (end_x - start_x), start_y + step_fraction * (end_y - start_y)


class RealWorld:
    """Class that contains the real world map and the real robot."""
    
    def __init__(self, map_size):
        self.map = Map(map_size)
        self.robot = RealRobot(self.map)

    def move_robot(self, move_dist, move_theta):
        """Move the real robot within the real world."""
        return self.robot.move(move_dist, move_theta)

    def get_sensor_reading(self):
        """Return the real robot's touch sensor reading."""
        return self.robot.sense_touch()


    def display_map_estimation(self, particles):
        """Visualize the aggregated map estimation."""
        plt.figure(figsize=(6, 6))
        
        # Display the consensus map
        consensus_map = aggregate_map_estimation(particles)
        plt.imshow(consensus_map, cmap='gray', origin='upper')
        
        # Set labels and grid
        plt.title('Consensus Map Estimation')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def display(self, particles = None):
        """Display the map, real robot, and particles."""
        plt.figure(figsize=(6, 6))
        
        # Define the extent based on the map size and the center origin
        extent = [-MAP_SIZE[0]//2, MAP_SIZE[0]//2, -MAP_SIZE[1]//2, MAP_SIZE[1]//2]
        
        # Display the map with the correct extent and flipped y-axis for proper orientation
        plt.imshow(self.map.grid, cmap='gray', origin='upper', extent=extent)

        # Plot particles (adjusted to match the coordinate system with center at (0, 0))
        if particles is not None:
            particle_x = [p.x for p in particles]
            particle_y = [p.y for p in particles]
            plt.scatter(particle_x, particle_y, color='red', s=1, label='Particles')

        # Plot the real robot
        plt.scatter(self.robot.x, self.robot.y, color='blue', s=50, label='Real Robot')
        plt.arrow(
            self.robot.x, self.robot.y,
            0.5 * np.cos(self.robot.theta),
            0.5 * np.sin(self.robot.theta),
            head_width=0.3, head_length=0.3, fc='blue', ec='blue'
        )

        # Set labels and limits
        plt.title('Particle Filter Localization')
        plt.legend()
        plt.grid(True)

        # Set the limits to match the map's coordinate system
        plt.xlim(-MAP_SIZE[0]//2, MAP_SIZE[0]//2)
        plt.ylim(-MAP_SIZE[1]//2, MAP_SIZE[1]//2)
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()



def particle_filter_step(particles, get_weights_fn, sensor_reading, move_dist, move_theta, resample = True):
    """Update particles based on motion, sensor data, and resampling."""
    # Predict step: move particles
    for particle in particles:
        particle.move(move_dist, move_theta)
        particle.update_map(sensor_reading)

    # Measurement update: compute weights based on touch sensor data
    weights = get_weights_fn(particles, sensor_reading) # TODO: for performance, when not training, may want to move this below into if statement

    if resample:

        #weights = []
        #consensus_map = aggregate_map_estimation(particles)
        #for particle in particles:
        #    weight = get_weights_fn(particles, sensor_reading, consensus_map)
        #    weights.append(weight)

        # Normalize weights
        #weights = np.array(weights)
        #if np.sum(weights) == 0:
        #    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
        #else:
        #    weights /= np.sum(weights)

        # Resample particles based on their weights
        indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights.numpy())
        particles = [copy.deepcopy(particles[i]) for i in indices]

    return particles, weights

def aggregate_map_estimation(particles, weights = None):
        """Aggregate the map estimates from all particles to build a consensus map."""
        consensus_map = np.zeros(MAP_SIZE)

        if weights is None:
            weights = np.ones(len(particles))

        assert len(particles) == len(weights)

        # Sum the map estimates from all particles
        for particle, weight in zip(particles, weights):
            consensus_map += particle.map_estimate * weight

        # Normalize the consensus map (0 to 1 range) to get probability of each cell being a wall
        consensus_map /= len(particles)

        return consensus_map

def run_step(real_world, step_number, particles, get_weights_fn, move_strategy_fn, visualize):

    # Decide movement based on strategy
    move_dist, move_theta = move_strategy_fn(particles, real_world.get_sensor_reading())

    # Move the real robot
    real_move_dist, real_move_theta = real_world.move_robot(move_dist, move_theta)

    # Update particles
    resample = step_number % NUM_OF_STEPS_BEFORE_RESAMPLE == step_number
    particles, weights = particle_filter_step(particles, get_weights_fn, real_world.get_sensor_reading(), real_move_dist, real_move_theta, resample)

    # Optionally visualize each step
    if visualize and ((step_number + 1) % 100 == 0 or step_number == NUM_STEPS - 1):
        print(f"Visualizing step {step_number + 1}")
        real_world.display(particles)
        real_world.display_map_estimation(particles)

    return weights

def run_simulation(real_world, weighting_NN, num_steps, visualize = True):

    # Initialize all particles at (0,0) facing upward
    particles = [Particle(real_world.map) for _ in range(NUM_PARTICLES)]

    # Visualize initial state
    if visualize:
        real_world.display(particles)

    for step_num in range(num_steps):

        print(f"Step {step_num + 1}/{num_steps}")

        with tf.GradientTape() as tape:
            # Run a single step
            weights = run_step(real_world, step_num, particles, weighting_NN.get_weights, weighting_NN.move_strategy, visualize)

            # Compute the loss
            ground_truth_position = (real_world.robot.x, real_world.robot.y, real_world.robot.theta)
            loss = weighting_NN.compute_loss(weights, particles, real_world.map, ground_truth_position)

        # Compute gradients and apply them using the optimizer
        gradients = tape.gradient(loss, weighting_NN.trainable_variables)
        if gradients is None or all(g is None for g in gradients):
            print("WARNING : No gradients were computed, skipping this step.")
        else:
            weighting_NN.optimizer.apply_gradients(zip(gradients, weighting_NN.trainable_variables))

        print(f"Loss at step {step_num + 1}: {loss.numpy()}")

class ParticleDistributionNN(tf.keras.Model):

    def __init__(self):
        super(ParticleDistributionNN, self).__init__()

        self.checkpoint_dir = "checkpoints"
        
        # Convolutional layers to process the map
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.fc_map = layers.Dense(128, activation='relu')

        # Transformer-like mechanism (simple dense layer to model interaction for simplicity)
        self.transformer = layers.Dense(128, activation='relu')

        # Final fully connected layer to output a weight for each particle
        self.fc_final = layers.Dense(1, activation='linear')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, particles, sensor_reading=None):
        """
        particles: a list of dictionaries where each particle has:
                   - map_estimate: a map of the environment (2D grid)
                   - x, y, theta: position and orientation of the robot
        sensor_reading: (Not used for now)
        """
        particle_maps = [particle.map_estimate for particle in particles]
        particle_positions = [[particle.x, particle.y, particle.theta] for particle in particles]

        # Stack the particle maps and positions into a tensor
        particle_maps = tf.convert_to_tensor(particle_maps, dtype=tf.float32)  # Shape: (num_particles, map_height, map_width)
        particle_positions = tf.convert_to_tensor(particle_positions, dtype=tf.float32)  # Shape: (num_particles, 3)

        # Add a batch dimension if needed
        particle_maps = tf.expand_dims(particle_maps, -1)  # Shape: (num_particles, map_height, map_width, 1)

        # Convolution layers to process maps
        conv_out = self.conv1(particle_maps)
        conv_out = self.conv2(conv_out)
        conv_out = self.flatten(conv_out)
        map_vectors = self.fc_map(conv_out)  # Shape: (num_particles, 128)

        # Concatenate map vectors with particle positions
        particle_features = tf.concat([map_vectors, particle_positions], axis=-1)  # Shape: (num_particles, 131)

        # Apply the transformer layer
        particle_features = self.transformer(particle_features)  # Shape: (num_particles, 128)

        # Final layer to output the weights
        weights = self.fc_final(particle_features)  # Shape: (num_particles, 1)
        weights = tf.squeeze(weights, axis=-1)  # Shape: (num_particles)
        weights = tf.nn.softmax(weights, axis=0) # Apply softmax to convert weights into probabilities

        return weights

    def compute_loss(self, weights, particles, ground_truth_map, ground_truth_position):
        """
        Compute the loss based on weighted map and position estimates compared to ground truth.
        """

        #weights = tf.convert_to_tensor(weights)

        # Extract map and positions from particles
        particle_positions = np.array([[p.x, p.y, p.theta] for p in particles])

        # Compute the weighted average of maps and positions
        weighted_map = aggregate_map_estimation(particles, weights)
        weighted_position = tf.reduce_sum(weights[:, None] * particle_positions, axis=0)

        # Compute losses
        map_loss = tf.reduce_mean(tf.square(weighted_map - ground_truth_map.grid))
        position_loss = tf.reduce_mean(tf.square(weighted_position - ground_truth_position))

        # Combine losses
        total_loss = map_loss + position_loss

        return total_loss

    def save_checkpoint(self, step_num):
        """
        Save model weights to a checkpoint inside the 'checkpoints' folder.
        Args:
            step_num: Current step number to differentiate checkpoints.
        """
        # Create the 'checkpoints' directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Define the checkpoint file path
        checkpoint_path = os.path.join(self.checkpoint_dir, f"particle_distribution_nn_step_{step_num}.weights.h5")

        # Save model weights
        self.save_weights(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, step_num):
        """
        Load model weights from a checkpoint.
        Args:
            step_num: Step number to identify which checkpoint to load.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"particle_distribution_nn_step_{step_num}.weights.h5")

        if os.path.exists(checkpoint_path):
            self.load_weights(checkpoint_path)
            print(f"Checkpoint loaded: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    def get_weights(self, particles, sensor_reading=None):
        """
        Return the weights for each particle.
        """
        return self.call(particles, sensor_reading)  # Convert the tensor to numpy for ease of use

    def move_strategy(self, particles, sensor_reading=None):
        """
        Decide a movement strategy based on particle distribution and sensor_reading.
        """
        if sensor_reading:
            return 0.5, 0.5
        return 0.5, 0  # Move forward by 1 unit, no rotation

def generate_random_world(map_size, num_vertical_walls, num_horizontal_walls):
    """
    Generate a random world with a given number of vertical and horizontal walls.
    
    Args:
        map_size: Tuple (width, height) of the map.
        num_vertical_walls: The number of vertical walls to generate.
        num_horizontal_walls: The number of horizontal walls to generate.

    Returns:
        real_world: A RealWorld object with randomly placed walls.
    """
    real_world = RealWorld(map_size)  # Create the real world with the real robot

    # Add boundaries
    real_world.map.add_wall_vertical(0,  map_size[1] - 1, 0)
    real_world.map.add_wall_vertical(0,  map_size[1] - 1, map_size[0] - 1)
    real_world.map.add_wall_horizontal(0, 0, map_size[0] - 1)
    real_world.map.add_wall_horizontal( map_size[1] - 1, 0, map_size[0] - 1)
    
    # Add vertical walls
    for _ in range(num_vertical_walls):
        col = random.randint(1, map_size[0] - 2)  # Random column within bounds
        start = random.randint(1, map_size[1] - 2)  # Random starting row within bounds
        end = random.randint(start + 1, map_size[1] - 1)  # Ensure end is after start
        real_world.map.add_wall_vertical(start, end, col)

    # Add horizontal walls
    for _ in range(num_horizontal_walls):
        row = random.randint(1, map_size[1] - 2)  # Random row within bounds
        start = random.randint(1, map_size[0] - 2)  # Random starting column within bounds
        end = random.randint(start + 1, map_size[0] - 1)  # Ensure end is after start
        real_world.map.add_wall_horizontal(row, start, end)

    return real_world

# Main loop

weighting_NN = ParticleDistributionNN()
#weighting_NN.load_checkpoint(337)

NUM_STEPS = 500
NUM_WORLDS = 1000

#for step_num in range(NUM_WORLDS):
#    #Create the world
#    test_world = generate_random_world(MAP_SIZE, 2, 2)
#
#    # Run the simulation
#    run_simulation(test_world, weighting_NN, NUM_STEPS, visualize = False)
#
#    # Save the model checkpoint
#    weighting_NN.save_checkpoint(step_num)



real_world = RealWorld(MAP_SIZE)  # Create the real world with the real robot
real_world.map.add_wall_horizontal(5, 5, 19)
real_world.map.add_wall_horizontal(19, 5, 19)
real_world.map.add_wall_vertical(1, 19, 5)
real_world.map.add_wall_vertical(1, 19, 19)
run_simulation(real_world, weighting_NN, NUM_STEPS, visualize = True)

    

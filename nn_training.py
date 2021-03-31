import os
import sys
import torch
import myModel
import datetime
import numpy as np
import torch.nn as nn
from tqdm import tqdm as tqdm
from vfields import AnalyticalVFields
from sim.geometry import Goal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset as BaseDataset



"""-------------------------------------------------------------------------"""
"""--------------Setting up the Directory, parameters and GPUs--------------"""
"""-------------------------------------------------------------------------"""


DATA_DIR = 'dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

load_model = False
loaded_model_name = 'Goal-one-obstacle-field-for-all-4-sigmoid'
save_model_name = 'Goal-one-obstacle-field-for-all-4-sigmoid'
tensorboard_name = save_model_name
tensorboard_logging = False
perform_training = True

# Number of epochs to train for, if None then until keyboard interrupt (ctrl+c)
# and training parameters
num_epochs = None
learning_rate = 1e-2
batch_size = 100
batch_size_doubling_epochs = [50000, 100000]

# How many epochs to save things
model_epochs = 1000
model_checkpoint_epochs = 10000

# Dimension of scenario
DIMENSIONS = 3


"""-------------------------------------------------------------------------"""
"""-Setting Train, Test and Validation Folders. Setting up the Tensorboard--"""
"""-logging location. Also getting a run datestamp to help distinguish file-"""
"""----------------------names----------------------------------------------"""
"""-------------------------------------------------------------------------"""


if perform_training is True and tensorboard_logging is True:
    writer = SummaryWriter(log_dir='logs')

save_dir = os.path.join('models', f'{save_model_name}.tar')
save_dir_backup = os.path.join('models', f'{save_model_name}_backup.tar')
load_dir = os.path.join('models', f'{loaded_model_name}.tar')
load_dir_backup = os.path.join('models', f'{loaded_model_name}_backup.tar')

print("Device Used:", DEVICE)


"""-------------------------------------------------------------------------"""
"""--------------Creating a Model and setting up optimiser------------------"""
"""-------------------------------------------------------------------------"""


# Loading a network
model = myModel.Net(dims=DIMENSIONS)

if load_model is True:
    # Loading the model weights
    checkpoint = torch.load(load_dir)
    epoch = checkpoint['epoch']
    load_dims = checkpoint['dims']
    if load_dims != DIMENSIONS:
        raise ValueError(f'Model is for {load_dims} dimensions and we are trying to train {DIMENSIONS} dimensions')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    # Loading the optimizer
    optimizer = torch.optim.SGD([
        dict(params=model.parameters(), lr=learning_rate, momentum=0.9),
    ])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    # Creating the optimiser
    optimizer = torch.optim.SGD([
        dict(params=model.parameters(), lr=learning_rate, momentum=0.9),
    ])
    epoch = 0


"""-------------------------------------------------------------------------"""
"""----------------------Generating data for training-----------------------"""
"""-------------------------------------------------------------------------"""


class Dataset(BaseDataset):
    """
    Randomly generating a scenario and returning the potential/velocity
    for it. We return the following using relative coordinates:
    
    goal_position - numpy array (dims)
    planets_position - numpy array (N, dims)
    planets_radii - numpy array (N, 1)
    spaceships - numpy array (N, dims*2)
    meteoroids - numpy array (N, dims*2)
    """

    def __init__(
            self,
            size_of_universe=2000,
            dims=2,
            decay_radius=20,
            convergence_radius=10,
            obstacle_scale=5,
            alpha=10
    ):
        self.size_of_universe = size_of_universe
        self.dims = dims
        self.vfields = AnalyticalVFields(decay_radius,
                                         convergence_radius,
                                         obstacle_scale,
                                         alpha)

    def __getitem__(self, i):
        # We randomly generate one of each of the objects that exist as they each have their own
        # sub-network that is trained
        # We use relative coordinates throughout
        position = np.zeros(self.dims)

        # Generating a random goal position using a uniform distribution and its associated velocity
        goal_position = np.random.uniform(-self.size_of_universe/2, self.size_of_universe/2, size=self.dims)
        goal = Goal(goal_position, goal_position/np.linalg.norm(goal_position))
        goal_velocity = self.vfields.goal(position, goal)

        # Generating a random planet position and radius using a weighted uniform distribution and its associated velocity
        planet_radius = np.random.uniform(40, 200)
        if np.random.uniform() < 0.75:
            dist_to_planet = np.random.uniform(planet_radius, self.size_of_universe/4)
        else:
            dist_to_planet = np.random.uniform(self.size_of_universe/4, self.size_of_universe/2)
        if self.dims == 2:
            theta = np.random.uniform(0, 2*np.pi)
            planet_position = np.array([dist_to_planet*np.cos(theta), dist_to_planet*np.sin(theta)])
        else:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            planet_position = np.array([dist_to_planet*np.cos(theta)*np.cos(phi), dist_to_planet*np.sin(theta)*np.cos(phi), dist_to_planet*np.sin(phi)])
        planet_velocity = self.vfields.obstacle(position, planet_position, planet_radius)

        # Generating a random spaceship position and radius using a weighted uniform distribution and its associated velocity
        spaceship_size = np.random.uniform(10, 20)
        if np.random.uniform() < 0.75:
            dist_to_spaceship = np.random.uniform(spaceship_size/2, self.size_of_universe/4)
        else:
            dist_to_spaceship = np.random.uniform(self.size_of_universe/4, self.size_of_universe/2)
        if self.dims == 2:
            theta = np.random.uniform(0, 2*np.pi)
            spaceship_position = np.array([dist_to_spaceship*np.cos(theta), dist_to_spaceship*np.sin(theta)])
        else:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            spaceship_position = np.array([dist_to_spaceship*np.cos(theta)*np.cos(phi), dist_to_spaceship*np.sin(theta)*np.cos(phi), dist_to_spaceship*np.sin(phi)])
        spaceship_velocity = self.vfields.obstacle(position, spaceship_position, spaceship_size)
        spaceship = np.concatenate((spaceship_position, spaceship_size), axis=None)

        # Generating a random meteoroid position and radius using a weighted uniform distribution and its associated velocity
        meteoroid_size = np.random.uniform(10, 20)
        if np.random.uniform() < 0.75:
            dist_to_meteoroid = np.random.uniform(meteoroid_size/2, self.size_of_universe/4)
        else:
            dist_to_meteoroid = np.random.uniform(self.size_of_universe/4, self.size_of_universe)
        if self.dims == 2:
            theta = np.random.uniform(0, 2*np.pi)
            meteoroid_position = np.array([dist_to_meteoroid*np.cos(theta), dist_to_meteoroid*np.sin(theta)])
        else:
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            meteoroid_position = np.array([dist_to_meteoroid*np.cos(theta)*np.cos(phi), dist_to_meteoroid*np.sin(theta)*np.cos(phi), dist_to_meteoroid*np.sin(phi)])
        meteoroid_velocity = self.vfields.obstacle(position, meteoroid_position, meteoroid_size)
        meteoroid = np.concatenate((meteoroid_position, meteoroid_size), axis=None)

        return goal_position, planet_position, planet_radius, spaceship, meteoroid, goal_velocity, planet_velocity, spaceship_velocity, meteoroid_velocity
        
    def __len__(self):
        return batch_size


train_dataset = Dataset(dims=DIMENSIONS)
dataloader = DataLoader(train_dataset, batch_size=batch_size)


"""-------------------------------------------------------------------------"""
"""-------------------------Training the Model------------------------------"""
"""-------------------------------------------------------------------------"""


if perform_training is True:
    print("Training")
    model.to(DEVICE)
    model.train()
    loss = nn.MSELoss().to(DEVICE)
    try:
        i = epoch + 1
        while True:
            # Perfoming training
            for index, (goal_position, planet_position, planet_radius, spaceship, meteoroid, goal_velocity, planet_velocity, spaceship_velocity, meteoroid_velocity) in enumerate(dataloader):
                if i == 1:
                    # performing data whitening
                    model.data_whitening(goal_position, planet_position, planet_radius, spaceship, meteoroid, DEVICE)
                # Could train the sub-networks in one go but this makes it more difficult
                """
                optimizer.zero_grad()
                prediction, goal_prediction, planets_prediction, spaceships_prediction, meteoroids_prediction = model.forward(goal_position.to(DEVICE).float(), 
                                                                                                                              planet_position.to(DEVICE).float(), 
                                                                                                                              planet_radii.to(DEVICE).float(), 
                                                                                                                              spaceship.to(DEVICE).float(), 
                                                                                                                              meteoroid.to(DEVICE).float())
                velocity = #INSERT SUM
                total_loss = loss(prediction, velocity.to(DEVICE).float())
                total_loss.backward()
                optimizer.step()
                """
                # We train the sub-networks seperately to make it as easy as possible
                optimizer.zero_grad()

                goal_prediction = model.forward_goal(goal_position.to(DEVICE).float())
                goal_loss = loss(goal_prediction, goal_velocity.to(DEVICE).float())

                planet_prediction = model.forward_planet(planet_position.to(DEVICE).float(), planet_radius.to(DEVICE).float())
                planet_loss = loss(planet_prediction, planet_velocity.to(DEVICE).float())

                spaceship_prediction = model.forward_spaceship(spaceship.to(DEVICE).float())
                spaceship_loss = loss(spaceship_prediction, spaceship_velocity.to(DEVICE).float())

                meteoroid_prediction = model.forward_meteoroid(meteoroid.to(DEVICE).float())
                meteoroid_loss = loss(meteoroid_prediction, meteoroid_velocity.to(DEVICE).float())

                total_loss = goal_loss + 5 * planet_loss + 5 * spaceship_loss + 5 * meteoroid_loss
                total_loss.backward()
                optimizer.step()
            
            # After 25K epochs, we switch the output non-linearity from
            # leaky_ReLU to ReLU
            if i % 25000 == 0:
                try:
                    model.switch_nonlinearity()
                except:
                    pass

            # After 200k epochs we reduce the learning rate
            if i % 200000 == 0:
                for g in optimizer.param_groups:
                    learning_rate = learning_rate / 10
                    g['lr'] = learning_rate
            
            # Writing values to tensorboard
            if (i % 100 == 0 or i == 1):
                print('\nEpoch: {}'.format(i))
                if tensorboard_logging is True:
                    goal_prediction = goal_prediction.cpu().detach().numpy()
                    planet_prediction = planet_prediction.cpu().detach().numpy()
                    spaceship_prediction = spaceship_prediction.cpu().detach().numpy()
                    meteoroid_prediction = meteoroid_prediction.cpu().detach().numpy()
                    prediction = goal_prediction + planet_prediction + spaceship_prediction + meteoroid_prediction
                    goal_velocity = goal_velocity.numpy()
                    planet_velocity = planet_velocity.numpy()
                    spaceship_velocity = spaceship_velocity.numpy()
                    meteoroid_velocity = meteoroid_velocity.numpy()
                    velocity = goal_velocity + planet_velocity + spaceship_velocity + meteoroid_velocity

                    writer.add_scalar(f'{tensorboard_name}/1 - Total Loss - MSE', total_loss, global_step=i)
                    writer.add_scalar(f'{tensorboard_name}/7 - Batch Size', batch_size, global_step=i)
                    writer.add_scalar(f'{tensorboard_name}/8 - Learning Rate', learning_rate, global_step=i)
                    if DIMENSIONS == 2:
                        writer.add_scalars(f'{tensorboard_name}/2 - Absolute Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(velocity[:, 0] - prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(velocity[:, 1] - prediction[:, 1])),
                                        'x - median': np.median(np.abs(velocity[:, 0] - prediction[:, 0])),
                                        'y - median': np.median(np.abs(velocity[:, 1] - prediction[:, 1]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/3 - Absolute Goal Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(goal_velocity[:, 0] - goal_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(goal_velocity[:, 1] - goal_prediction[:, 1])),
                                        'x - median': np.median(np.abs(goal_velocity[:, 0] - goal_prediction[:, 0])),
                                        'y - median': np.median(np.abs(goal_velocity[:, 1] - goal_prediction[:, 1]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/4 - Absolute Planets Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(planet_velocity[:, 0] - planet_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(planet_velocity[:, 1] - planet_prediction[:, 1])),
                                        'x - median': np.median(np.abs(planet_velocity[:, 0] - planet_prediction[:, 0])),
                                        'y - median': np.median(np.abs(planet_velocity[:, 1] - planet_prediction[:, 1]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/5 - Absolute Spaceships Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(spaceship_velocity[:, 0] - spaceship_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(spaceship_velocity[:, 1] - spaceship_prediction[:, 1])),
                                        'x - median': np.median(np.abs(spaceship_velocity[:, 0] - spaceship_prediction[:, 0])),
                                        'y - median': np.median(np.abs(spaceship_velocity[:, 1] - spaceship_prediction[:, 1]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/6 - Absolute Meteoroids Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(meteoroid_velocity[:, 0] - meteoroid_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(meteoroid_velocity[:, 1] - meteoroid_prediction[:, 1])),
                                        'x - median': np.median(np.abs(meteoroid_velocity[:, 0] - meteoroid_prediction[:, 0])),
                                        'y - median': np.median(np.abs(meteoroid_velocity[:, 1] - meteoroid_prediction[:, 1]))},
                                        global_step=i)
                    else:
                        writer.add_scalars(f'{tensorboard_name}/2 - Absolute Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(velocity[:, 0] - prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(velocity[:, 1] - prediction[:, 1])),
                                        'z - mean': np.mean(np.abs(velocity[:, 2] - prediction[:, 2])),
                                        'x - median': np.median(np.abs(velocity[:, 0] - prediction[:, 0])),
                                        'y - median': np.median(np.abs(velocity[:, 1] - prediction[:, 1])),
                                        'z - median': np.median(np.abs(velocity[:, 2] - prediction[:, 2]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/3 - Absolute Goal Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(goal_velocity[:, 0] - goal_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(goal_velocity[:, 1] - goal_prediction[:, 1])),
                                        'z - mean': np.mean(np.abs(goal_velocity[:, 2] - goal_prediction[:, 2])),
                                        'x - median': np.median(np.abs(goal_velocity[:, 0] - goal_prediction[:, 0])),
                                        'y - median': np.median(np.abs(goal_velocity[:, 1] - goal_prediction[:, 1])),
                                        'z - median': np.median(np.abs(goal_velocity[:, 2] - goal_prediction[:, 2]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/4 - Absolute Planets Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(planet_velocity[:, 0] - planet_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(planet_velocity[:, 1] - planet_prediction[:, 1])),
                                        'z - mean': np.mean(np.abs(planet_velocity[:, 2] - planet_prediction[:, 2])),
                                        'x - median': np.median(np.abs(planet_velocity[:, 0] - planet_prediction[:, 0])),
                                        'y - median': np.median(np.abs(planet_velocity[:, 1] - planet_prediction[:, 1])),
                                        'z - median': np.median(np.abs(planet_velocity[:, 2] - planet_prediction[:, 2]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/5 - Absolute Spaceships Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(spaceship_velocity[:, 0] - spaceship_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(spaceship_velocity[:, 1] - spaceship_prediction[:, 1])),
                                        'z - mean': np.mean(np.abs(spaceship_velocity[:, 2] - spaceship_prediction[:, 2])),
                                        'x - median': np.median(np.abs(spaceship_velocity[:, 0] - spaceship_prediction[:, 0])),
                                        'y - median': np.median(np.abs(spaceship_velocity[:, 1] - spaceship_prediction[:, 1])),
                                        'z - median': np.median(np.abs(spaceship_velocity[:, 2] - spaceship_prediction[:, 2]))},
                                        global_step=i)
                        writer.add_scalars(f'{tensorboard_name}/6 - Absolute Meteorites Velocity Errors (m/s)',
                                        {'x - mean': np.mean(np.abs(meteoroid_velocity[:, 0] - meteoroid_prediction[:, 0])),
                                        'y - mean': np.mean(np.abs(meteoroid_velocity[:, 1] - meteoroid_prediction[:, 1])),
                                        'z - mean': np.mean(np.abs(meteoroid_velocity[:, 2] - meteoroid_prediction[:, 2])),
                                        'x - median': np.median(np.abs(meteoroid_velocity[:, 0] - meteoroid_prediction[:, 0])),
                                        'y - median': np.median(np.abs(meteoroid_velocity[:, 1] - meteoroid_prediction[:, 1])),
                                        'z - median': np.median(np.abs(meteoroid_velocity[:, 2] - meteoroid_prediction[:, 2]))},
                                        global_step=i)
            
            # Doubling the batch size at fixed epochs
            if i in batch_size_doubling_epochs:
                batch_size = batch_size * 2

            # Saving the model every n epochs and saving a backup in
            # case of a model corruption when interrupting
            if i % model_epochs == 0:
                model_data = {'epoch': i,
                              'dims': DIMENSIONS,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(model_data, save_dir)
                torch.save(model_data, save_dir_backup)
                print('Model saved!')
            
            # Saving a model checkpoint at a fixed number of epochs to go
            # back and test different versions
            if i % model_checkpoint_epochs == 0:
                model_data = {'epoch': i,
                              'dims': DIMENSIONS,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                checkpoint_dir = os.path.join('models', f'{save_model_name}_{i}.tar')
                torch.save(model_data, checkpoint_dir)
                print('Model saved!')

            # Checking whether or not to stop training
            if num_epochs is not None:
                if i == num_epochs + epoch:
                    model_data = {'epoch': i,
                                  'dims': DIMENSIONS,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(model_data, save_dir)
                    print('Model saved!')
                    break
            i += 1
    except KeyboardInterrupt:
        model_data = {'epoch': i,
                      'dims': DIMENSIONS,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_data, save_dir)
        print('Model saved!')

if perform_training is True and tensorboard_logging is True:
    writer.flush()

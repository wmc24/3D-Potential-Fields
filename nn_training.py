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
loaded_model_name = 'Debugging'
save_model_name = 'Debugging'
tensorboard_name = save_model_name
tensorboard_logging = True
perform_training = True

# Number of epochs to train for, if None then until keyboard interrupt (ctrl+c)
# and training parameters
num_epochs = None
learning_rate = 1e-5
batch_size = 10
batch_size_doubling_epochs = [100000, 250000, 1000000, 10000000]

# How many epochs to save things
model_epochs = 1000
model_checkpoint_epochs = 50000

# Dimension of scenario
DIMENSIONS = 2


"""-------------------------------------------------------------------------"""
"""-Setting Train, Test and Validation Folders. Setting up the Tensorboard--"""
"""-logging location. Also getting a run datestamp to help distinguish file-"""
"""----------------------names----------------------------------------------"""
"""-------------------------------------------------------------------------"""


# Modifying default format to avoid characters that shouldn't be used in filenames
t = datetime.datetime.now()
date = t.date()
hour = str(t.hour).zfill(2)
minute = str(t.minute).zfill(2)
second = str(t.second).zfill(2)
run_datestamp = f'{date} {hour}_{minute}_{second}'

if perform_training is True and tensorboard_logging is True:
    tensorboard_name = f'{run_datestamp} {tensorboard_name}'
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
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    # Creating the optimiser
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
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
            planets_pot=False,
            spaceships_pot=False,
            meteoroids_pot=False,
            size_of_universe=2000,
            dims=2,
            decay_radius=20,
            convergence_radius=10,
            obstacle_scale=5,
            alpha=10
    ):
        self.planets_pot = planets_pot
        self.spaceships_pot = spaceships_pot
        self.meteoroids_pot = meteoroids_pot
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
        goal_position = np.random.uniform(-self.size_of_universe/2, self.size_of_universe, size=self.dims)
        goal = Goal(goal_position, goal_position)
        goal_velocity = self.vfields.goal(position, goal)

        # Generating a random planet position and radius using a uniform distribution and its associated velocity
        planet_position = np.random.uniform(-self.size_of_universe/2, self.size_of_universe, size=self.dims)
        planet_radius = np.random.uniform(40, 200)
        planet_velocity = self.vfields.obstacle(position, planet_position, planet_radius)

        # THESE ARE ALL TEMPORARY
        planet_velocity = np.random.uniform(-1, 1, size=self.dims)
        spaceship_velocity = np.random.uniform(-1, 1, size=self.dims)
        meteoroid_velocity = np.random.uniform(-1, 1, size=self.dims) 

        spaceship = np.random.uniform(-self.size_of_universe/2, self.size_of_universe, size=self.dims)
        spaceship_size = np.random.uniform(10, 20)
        spaceship_velocity = self.vfields.obstacle(position, spaceship, spaceship_size)

        meteoroid = np.random.uniform(-self.size_of_universe/2, self.size_of_universe, size=self.dims)
        meteoroid_size = np.random.uniform(10, 20)
        meteoroid_velocity = self.vfields.obstacle(position, meteoroid, meteoroid_size)

        # Converting to numpy arrays
        #goal_position = np.array(goal_position)
        #planet_position = np.array(planet_position)
        #planet_radius = np.array(planet_radius).reshape((-1, 1))
        #spaceship = np.array(spaceship).reshape((-1, self.dims*2))
        #meteoroid = np.array(meteoroid).reshape((-1, self.dims*2))

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
                planet_prediction = model.forward_goal(planet_position.to(DEVICE).float())
                planet_loss = loss(planet_prediction, planet_velocity.to(DEVICE).float())
                spaceship_prediction = model.forward_goal(spaceship.to(DEVICE).float())
                spaceship_loss = loss(spaceship_prediction, spaceship_velocity.to(DEVICE).float())
                meteoroid_prediction = model.forward_goal(meteoroid.to(DEVICE).float())
                meteoroid_loss = loss(meteoroid_prediction, meteoroid_velocity.to(DEVICE).float())
                total_loss = goal_loss + planet_loss + spaceship_loss + meteoroid_loss
                total_loss.backward()
                optimizer.step()
            
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
                    if DIMENSIONS == 2:
                        writer.add_scalar(f'{tensorboard_name}/1 - Total Loss - MSE', total_loss, global_step=i)
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
                        writer.add_scalar(f'{tensorboard_name}/7 - Batch Size', batch_size, global_step=i)
                    else:
                        writer.add_scalar(f'{tensorboard_name}/1 - Total Loss - MSE', total_loss, global_step=i)
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
                        writer.add_scalar(f'{tensorboard_name}/7 - Batch Size', batch_size, global_step=i)
            
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

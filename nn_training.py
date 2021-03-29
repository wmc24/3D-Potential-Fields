import os
import sys
import torch
import myModel
import datetime
import potential_field
import numpy as np
import torch.nn as nn
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset as BaseDataset



"""-------------------------------------------------------------------------"""
"""--------------Setting up the Directory, parameters and GPUs--------------"""
"""-------------------------------------------------------------------------"""


DATA_DIR = 'dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

load_model = False
loaded_model_name = 'Goal-Potential'
save_model_name = 'Goal-Potential'
tensorboard_name = save_model_name
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

if perform_training is True:
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
model = myModel.BasicGoalNet()

if load_model is True:
    # Loading the model weights
    checkpoint = torch.load(load_dir)
    epoch = checkpoint['epoch']
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
    planet_positions - numpy array (N, dims)
    planet_radii - numpy array (N, 1)
    spaceships - numpy array (N, dims*2)
    meteoroids - numpy array (N, dims*2)
    """

    def __init__(
            self,
            planets_pot=False,
            spaceships_pot=False,
            meteoroids_pot=False,
            size_of_universe=10,
            dims=2
    ):
        self.planets_pot = planets_pot
        self.spaceships_pot = spaceships_pot
        self.meteoroids_pot = meteoroids_pot
        self.size_of_universe = size_of_universe
        self.dims = 2

    def __getitem__(self, i):
        # Set the position as the centre as the net expects
        # relative coordinates
        position = np.array([0, 0, 0])

        # Generating a random goal position using a uniform distribution and its associated velocity
        goal_position = np.random.uniform(-self.size_of_universe/2, self.size_of_universe, size=3)
        goal_velocity = potential_field.get_velocity_to_reach_goal(position, goal_position, self.dims)

        # Defining the rest of the environment NEED TO MODIFY IN FUTURE
        planet_positions = []
        planet_radii = []
        spaceships = []
        meteoroids = []

        # Reshaping to the desired shapes and converting to numpy arrays
        goal_position = np.array(goal_position)
        planet_positions = np.array(planet_positions).reshape((-1, self.dims))
        planet_radii = np.array(planet_radii).reshape((-1, 1))
        spaceships = np.array(spaceships).reshape((-1, self.dims*2))
        meteoroids = np.array(meteoroids).reshape((-1, self.dims*2))

        # Getting the velocity array
        velocity = goal_velocity

        return goal_position, planet_positions, planet_radii, spaceships, meteoroids, velocity, goal_velocity
        
    def __len__(self):
        return batch_size


train_dataset = Dataset()
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

            for index, (goal_position, planet_positions, planet_radii, spaceships, meteoroids, velocity, goal_velocity) in enumerate(dataloader):
                #goal_position = goal_position.to(DEVICE).float()
                #planet_positions = planet_positions.to(DEVICE).float()
                #planet_radii = planet_radii.to(DEVICE).float()
                #spaceships = spaceships.to(DEVICE).float()
                #meteoroids = meteoroids.to(DEVICE).float()
                #velocity = velocity.to(DEVICE).float()

                optimizer.zero_grad()
                prediction, goal_prediction = model.forward(goal_position.to(DEVICE).float(), 
                                                            planet_positions.to(DEVICE).float(), 
                                                            planet_radii.to(DEVICE).float(), 
                                                            spaceships.to(DEVICE).float(), 
                                                            meteoroids.to(DEVICE).float())
                total_loss = loss(prediction, velocity.to(DEVICE).float())
                total_loss.backward()
                optimizer.step()
            
            # Writing values to tensorboard
            if i % 100 == 0 or i == 1:
                print('\nEpoch: {}'.format(i))
                prediction = prediction.cpu().detach().numpy()
                goal_prediction = goal_prediction.cpu().detach().numpy()
                velocity = velocity.numpy()
                goal_velocity = goal_velocity.numpy()
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
                writer.add_scalar(f'{tensorboard_name}/4 - Batch Size', batch_size, global_step=i)
            
            # Doubling the batch size at fixed epochs
            if i in batch_size_doubling_epochs:
                batch_size = batch_size * 2

            # Saving the model every n epochs and saving a backup in
            # case of a model corruption when interrupting
            if i % model_epochs == 0:
                model_data = {'epoch': i,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(model_data, save_dir)
                torch.save(model_data, save_dir_backup)
                print('Model saved!')
            
            # Saving a model checkpoint at a fixed number of epochs to go
            # back and test different versions
            if i % model_checkpoint_epochs == 0:
                model_data = {'epoch': i,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                checkpoint_dir = os.path.join('models', f'{save_model_name}_{i}.tar')
                torch.save(model_data, checkpoint_dir)
                print('Model saved!')

            # Checking whether or not to stop training
            if num_epochs is not None:
                if i == num_epochs + epoch:
                    model_data = {'epoch': i,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(model_data, save_dir)
                    print('Model saved!')
                    break
            i += 1
    except KeyboardInterrupt:
        model_data = {'epoch': i,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_data, save_dir)
        print('Model saved!')

if perform_training is True:
    writer.flush()
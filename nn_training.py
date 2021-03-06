import os
import sys
import torch
import myModel
import datetime
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from vfields import AnalyticalVFields
from sim.geometry import Goal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset as BaseDataset



'''-------------------------------------------------------------------------'''
'''--------------Setting up the Directory, parameters and GPUs--------------'''
'''-------------------------------------------------------------------------'''


DATA_DIR = 'dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

load_model = True
loaded_model_name = 'Potential-Field'
save_model_name = 'Potential-Field'
tensorboard_name = save_model_name
tensorboard_logging = False
perform_training = False
# Whether or not to train the subnetworks together or seperately
combined_training = False

# Number of epochs to train for, if None then until keyboard interrupt (ctrl+c)
# and training parameters
num_epochs = 100000
learning_rate = 1e-2
# Decrease the learning rate by a factor of 10
learning_rate_decrease_epochs = [75000]
batch_size = 250
# Increase the batch size by a factor of 4
batch_size_increase_epochs = [30000]

# How many epochs to save things
model_epochs = 1000
model_checkpoint_epochs = 5000

# Dimension of scenario
DIMENSIONS = 3


'''-------------------------------------------------------------------------'''
'''-Setting Train, Test and Validation Folders. Setting up the Tensorboard--'''
'''-logging location. Also getting a run datestamp to help distinguish file-'''
'''----------------------names----------------------------------------------'''
'''-------------------------------------------------------------------------'''


if perform_training is True and tensorboard_logging is True:
    writer = SummaryWriter(log_dir='logs')

save_dir = os.path.join('models', f'{save_model_name}.tar')
save_dir_backup = os.path.join('models', f'{save_model_name}_backup.tar')
load_dir = os.path.join('models', f'{loaded_model_name}.tar')
load_dir_backup = os.path.join('models', f'{loaded_model_name}_backup.tar')

print('Device Used:', DEVICE)


'''-------------------------------------------------------------------------'''
'''--------------Creating a Model and setting up optimiser------------------'''
'''-------------------------------------------------------------------------'''


# Loading a network
model = myModel.Net(dims=DIMENSIONS)

if load_model is True:
    # Loading the model weights
    checkpoint = torch.load(load_dir, map_location=DEVICE)
    epoch = checkpoint['epoch']
    epoch_start = epoch
    load_dims = checkpoint['dims']
    if load_dims != DIMENSIONS:
        raise ValueError(f'Model is for {load_dims} dimensions and we are trying to train {DIMENSIONS} dimensions')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.whitening = checkpoint['whitening']
    model.leaky = checkpoint['leaky']
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
    epoch_start = 0


'''-------------------------------------------------------------------------'''
'''----------------------Generating data for training-----------------------'''
'''-------------------------------------------------------------------------'''


class Dataset(BaseDataset):
    '''
    Randomly generating a scenario and returning the potential/speed
    for it. 
    '''

    def __init__(
            self,
            size_of_universe,
            dims,
            vfields
    ):
        self.size_of_universe = size_of_universe
        self.dims = dims
        self.vfields = vfields

    def __getitem__(self, i):
        # We randomly generate one of each of the objects that exist as they each have their own
        # sub-network that is trained
        # We use relative coordinates throughout

        # Generating a random goal dist using a weighted uniform distribution and its associated speed
        if epoch < 30000:
            if np.random.uniform() < 0.975:
                goal_dist = np.random.uniform(0, 100)
            else:
                goal_dist = np.random.uniform(100, self.size_of_universe/2)
        else:
            goal_dist = np.random.uniform(0, self.size_of_universe/2)
        angle = np.random.random()*2*np.pi
        direction = np.array([np.cos(angle), np.sin(angle)])
        if self.dims==3:
            vert_angle = (-0.5+np.random.random())*np.pi
            direction *= np.cos(vert_angle)
            direction = np.array([direction[0], direction[1], np.sin(vert_angle)])
        goal_disp = direction * goal_dist
        goal_velocity = self.vfields._goal(goal_disp)

        # Generating a random planet dist and radius using a weighted uniform distribution and its associated speed
        planet_radius = np.random.uniform(40, 200)
        if epoch < 30000:
            if np.random.uniform() < 0.8:
                planet_dist = np.random.uniform(planet_radius, planet_radius+200)
            else:
                planet_dist = np.random.uniform(planet_radius+200, self.size_of_universe/2)
        else:
            planet_dist = np.random.uniform(planet_radius, self.size_of_universe/2)
        planet_speed = self.vfields._obstacle(planet_dist, planet_radius)

        # Generating a random spaceship dist and radius using a weighted uniform distribution and its associated speed
        spaceship_size = np.random.uniform(10, 30)
        if epoch < 30000:
            if np.random.uniform() < 0.8:
                spaceship_dist = np.random.uniform(spaceship_size, spaceship_size+200)
            else:
                spaceship_dist = np.random.uniform(spaceship_size+200, self.size_of_universe/2)
        else:
            spaceship_dist = np.random.uniform(spaceship_size, self.size_of_universe/2)
        spaceship_speed = self.vfields._obstacle(spaceship_dist, spaceship_size)

        # Generating a random meteoroid dist, radius and speed using a weighted uniform distribution and its associated speed
        # Note that this potential is always passing a 2d arguement as it considers distance along direction of velocity
        # and distance perpendicular to direction of velocity (angle ignored)
        meteoroid_size = np.random.uniform(20, 80)
        meteoroid_input_speed = np.random.uniform(500, 700)
        if epoch < 30000:
            if np.random.uniform() < 0.975:
                meteoroids_disp_1 = np.random.uniform(0, meteoroid_size+self.size_of_universe/4)
                meteoroids_disp_2 = np.random.uniform(-meteoroid_size-200, meteoroid_size+200)
            else:
                meteoroids_disp_1 = np.random.uniform(0, self.size_of_universe)
                meteoroids_disp_2 = np.random.uniform(-self.size_of_universe, self.size_of_universe)
        else:
            meteoroids_disp_1 = np.random.uniform(0, self.size_of_universe)
            meteoroids_disp_2 = np.random.uniform(-self.size_of_universe, self.size_of_universe)
        meteoroids_disp = np.array((meteoroids_disp_1, meteoroids_disp_2))
        meteoroid_speed = self.vfields._moving_obstacle(meteoroids_disp, meteoroid_size, meteoroid_input_speed)

        return goal_disp, planet_dist, planet_radius, spaceship_dist, spaceship_size, meteoroids_disp, meteoroid_size, meteoroid_input_speed, goal_velocity, planet_speed, spaceship_speed, meteoroid_speed
        
    def __len__(self):
        return batch_size


size_of_universe=2000
decay_radius=30
convergence_radius=20
alpha=10
weights = [0.5, 1.1, 1]
vfields = AnalyticalVFields(weights=weights,
                            decay_radius=decay_radius,
                            convergence_radius=convergence_radius,
                            alpha=alpha)
train_dataset = Dataset(size_of_universe, DIMENSIONS, vfields)
dataloader = DataLoader(train_dataset, batch_size=batch_size)


'''-------------------------------------------------------------------------'''
'''-------------------------Training the Model------------------------------'''
'''-------------------------------------------------------------------------'''


if perform_training is True:
    print('Training')
    model.to(DEVICE)
    model.train()
    loss = nn.MSELoss().to(DEVICE)
    try:
        epoch = epoch + 1
        while True:
            # Perfoming training
            for index, (goal_disp, planet_dist, planet_radius, spaceship_dist, spaceship_size, meteoroids_disp, meteoroid_size, meteoroid_input_speed, goal_velocity, planet_speed, spaceship_speed, meteoroid_speed) in enumerate(dataloader):
                if epoch == 1:
                    # performing data whitening
                    print('Calculating Data Whitening')
                    model.data_whitening(goal_disp, planet_dist, planet_radius, spaceship_dist, spaceship_size, meteoroids_disp, meteoroid_size, meteoroid_input_speed, DEVICE)

                if combined_training is True:
                    raise NotImplementedError
                else:
                    # We train the sub-networks seperately to make it as easy as possible
                    optimizer.zero_grad()

                    goal_prediction = model.forward_goal(goal_disp.to(DEVICE).float())
                    goal_loss = loss(goal_prediction, goal_velocity.to(DEVICE).float())

                    planet_prediction = model.forward_obstacle(planet_dist.to(DEVICE).float(), planet_radius.to(DEVICE).float())
                    planet_loss = loss(planet_prediction.squeeze(), planet_speed.to(DEVICE).float())

                    spaceship_prediction = model.forward_obstacle(spaceship_dist.to(DEVICE).float(), spaceship_size.to(DEVICE).float())
                    spaceship_loss = loss(spaceship_prediction.squeeze(), spaceship_speed.to(DEVICE).float())

                    meteoroid_prediction = model.forward_moving_obstacle(meteoroids_disp.to(DEVICE).float(), meteoroid_size.to(DEVICE).float(), meteoroid_input_speed.to(DEVICE).float())
                    meteoroid_loss = loss(meteoroid_prediction.squeeze(), meteoroid_speed.to(DEVICE).float())

                    total_loss = goal_loss + planet_loss + spaceship_loss + meteoroid_loss
                    total_loss.backward()
                    optimizer.step()
            
            # After 50K epochs, we switch the output non-linearity from
            # leaky_ReLU to ReLU
            if epoch % 50000 == 0:
                try:
                    model.switch_nonlinearity()
                except:
                    pass

            # Decreasing the learning rate by a factor of 10
            if epoch in learning_rate_decrease_epochs:
                for g in optimizer.param_groups:
                    learning_rate = learning_rate / 10
                    g['lr'] = learning_rate
            
            # Writing values to tensorboard
            if (epoch % 10 == 0 or epoch == 1):
                print('\nEpoch: {}'.format(epoch))
                if tensorboard_logging is True:
                    goal_prediction = goal_prediction.cpu().detach().numpy()
                    planet_prediction = planet_prediction.squeeze().cpu().detach().numpy()
                    spaceship_prediction = spaceship_prediction.squeeze().cpu().detach().numpy()
                    meteoroid_prediction = meteoroid_prediction.squeeze().cpu().detach().numpy()
                    goal_velocity = goal_velocity.numpy()
                    planet_speed = planet_speed.numpy()
                    spaceship_speed = spaceship_speed.numpy()
                    meteoroid_speed = meteoroid_speed.numpy()

                    writer.add_scalar(f'{tensorboard_name}/1 - Total Loss - MSE', total_loss, global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/2 - Goal Loss - MSE', goal_loss, global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/3 - Planet Loss - MSE', planet_loss, global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/4 - Spaceship Loss - MSE', spaceship_loss, global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/5 - Meteoroid Loss - MSE', meteoroid_loss, global_step=epoch)
                    writer.add_scalars(f'{tensorboard_name}/6 - Absolute Goal Velocity Errors (m/s)',
                                    {'mean': np.mean(np.abs(goal_velocity - goal_prediction)),
                                     'median': np.median(np.abs(goal_velocity - goal_prediction))},
                                    global_step=epoch)
                    writer.add_scalars(f'{tensorboard_name}/7 - Absolute Planets Speed Errors (m/s)',
                                    {'mean': np.mean(np.abs(planet_speed - planet_prediction)),
                                     'median': np.median(np.abs(planet_speed - planet_prediction))},
                                    global_step=epoch)
                    writer.add_scalars(f'{tensorboard_name}/8 - Absolute Spaceships Speed Errors (m/s)',
                                    {'mean': np.mean(np.abs(spaceship_speed - spaceship_prediction)),
                                    'median': np.median(np.abs(spaceship_speed - spaceship_prediction))},
                                    global_step=epoch)
                    writer.add_scalars(f'{tensorboard_name}/9 - Absolute Meteorites Speed Errors (m/s)',
                                    {'mean': np.mean(np.abs(meteoroid_speed - meteoroid_prediction)),
                                     'median': np.median(np.abs(meteoroid_speed - meteoroid_prediction))},
                                    global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/10 - Batch Size', batch_size, global_step=epoch)
                    writer.add_scalar(f'{tensorboard_name}/11 - Learning Rate', learning_rate, global_step=epoch)

            # Saving a plot of the results
            if epoch % 5000 == 0 or epoch == 1:
                x = np.linspace(0, size_of_universe/2, 1000).reshape((-1, 1))
                y = np.zeros(np.shape(x))
                disp = np.concatenate((-x, y), axis=1)
                if DIMENSIONS == 3:
                    disp = np.concatenate((disp, y), axis=1)
                velocity = model.forward_goal(torch.tensor(disp).to(DEVICE).float()).detach().cpu().numpy()
                gt_velocity = vfields._goal_list(disp)
                fig, axs = plt.subplots(2, figsize=(10, 6.666))
                axs[0].plot(x, velocity[:, 0], label=f'Predicted')
                axs[0].plot(x, gt_velocity[:, 0], label=f'Ground Truth')
                axs[1].plot(x, velocity[:, 0], label=f'Predicted')
                axs[1].plot(x, gt_velocity[:, 0], label=f'Ground Truth')
                plt.xlabel('Distance')
                axs[0].set_ylabel('Speed')
                axs[1].set_ylabel('Speed')
                axs[0].set_xlim([0, 1000])
                axs[1].set_xlim([0, 40])
                axs[0].set_ylim([-0.1, 1.5])
                axs[1].set_ylim([-0.1, 1.5])
                axs[1].legend(loc=7)
                axs[0].legend(loc=7)
                plot_dir = os.path.join('plots', f'{epoch}_goal_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

                fig, axs = plt.subplots(3, figsize=(10, 10))
                j = 0
                for radius in [40, 100, 200]:
                    radii = torch.ones(np.shape(x)).to(DEVICE).float() * radius
                    speeds = model.forward_obstacle(torch.tensor(x).to(DEVICE).float(), radii).detach().cpu().numpy()
                    gt_speeds = vfields._obstacle_list(x, radius)
                    axs[j].plot(x, speeds, label=f'Predicted, radius = {radius}')
                    axs[j].plot(x, gt_speeds, label=f'Ground Truth, radius = {radius}')
                    axs[j].set_ylabel('Speed')
                    axs[j].set_xlim((0, 1000))
                    axs[j].legend(loc=1)
                    j += 1
                plt.xlabel('Distance')
                plot_dir = os.path.join('plots', f'{epoch}_planets_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

                fig, axs = plt.subplots(2, figsize=(10, 6.666))
                j = 0
                for radius in [10, 20]:
                    radii = torch.ones(np.shape(x)).to(DEVICE).float() * radius
                    speeds = model.forward_obstacle(torch.tensor(x).to(DEVICE).float(), radii).detach().cpu().numpy()
                    gt_speeds = vfields._obstacle_list(x, radius)
                    axs[j].plot(x, speeds, label=f'Predicted, radius = {radius}')
                    axs[j].plot(x, gt_speeds, label=f'Ground Truth, radius = {radius}')
                    axs[j].set_ylabel('Speed')
                    axs[j].set_xlim((0, 1000))
                    axs[j].legend(loc=1)
                    j += 1
                plt.xlabel('Distance')
                plot_dir = os.path.join('plots', f'{epoch}_spaceships_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

                fig, axs = plt.subplots(2, figsize=(10, 6.666))
                j = 0
                x = np.linspace(-size_of_universe/2, size_of_universe/2, 1000).reshape((-1, 1))
                y = np.ones(np.shape(x)) * 100
                disp = np.concatenate((y, x), axis=1)
                for radius, input_speed in [(30, 500), (70, 700)]:
                    radii = torch.ones(np.shape(x)).to(DEVICE).float() * radius
                    input_speeds = torch.ones(np.shape(x)).to(DEVICE).float() * input_speed
                    speeds = model.forward_moving_obstacle(torch.tensor(disp).to(DEVICE).float(), radii, input_speeds).detach().cpu().numpy()
                    gt_speeds = vfields._moving_obstacle_list(disp, radius, input_speed)
                    axs[j].plot(x, speeds, label=f'Predicted,\nradius = {radius},\nmeteoroid speed = {input_speed}')
                    axs[j].plot(x, gt_speeds, label=f'Ground Truth,\nradius = {radius},\nmeteoroid speed = {input_speed}')
                    axs[j].set_ylabel('Speed')
                    axs[j].set_xlim((-size_of_universe/2, size_of_universe/2))
                    axs[j].legend(loc=4)
                    j += 1
                plt.xlabel('Distance Perpendicular to Meteoroid Velocity')
                plot_dir = os.path.join('plots', f'{epoch}_meteoroids_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

                x = np.arange(-2, 251).reshape((-1, 1))
                y = np.arange(-126, 127).reshape((-1, 1))
                X, Y = np.meshgrid(x, y)
                DISP = np.concatenate((np.expand_dims(X, axis =2), np.expand_dims(Y, axis =2)), axis=2).reshape((-1, 2))
                radius = 40
                input_speed = 600
                radii = np.ones(np.shape(x)) * radius
                input_speeds = np.ones(np.shape(x)) * input_speed
                RADII, INPUT_SPEEDS = torch.tensor(np.meshgrid(radii, input_speeds)).to(DEVICE).float()
                SPEEDS = model.forward_moving_obstacle(torch.tensor(DISP).to(DEVICE).float(), RADII, INPUT_SPEEDS).detach().cpu().numpy().reshape(np.shape(X))
                GT_SPEEDS = vfields._moving_obstacle_list(DISP, radius, input_speed).reshape(np.shape(X))

                fig = plt.figure(figsize=(8, 6.666))
                ax = plt.axes()
                plt.contourf(X, Y, SPEEDS, np.arange(GT_SPEEDS.min(), GT_SPEEDS.max(), .01), cmap='jet')
                plt.xlabel('Distance Parallel to\nMeteoroid Velocity')
                plt.ylabel('Distance Perpendicular to\nMeteoroid Velocity')
                plt.xlim([0, 250])
                plt.ylim([-125, 125])
                cb = plt.colorbar()
                cb.set_label('Speed')
                plot_dir = os.path.join('plots', f'{epoch}_meteoroids_3D_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

                fig = plt.figure(figsize=(8, 6.666))
                ax = plt.axes()
                plt.contourf(X, Y, GT_SPEEDS, np.arange(GT_SPEEDS.min(), GT_SPEEDS.max(), .01), cmap='jet')
                plt.xlabel('Distance Parallel to\nMeteoroid Velocity')
                plt.ylabel('Distance Perpendicular to\nMeteoroid Velocity')
                plt.xlim([0, 250])
                plt.ylim([-125, 125])
                cb = plt.colorbar()
                cb.set_label('Speed')
                plot_dir = os.path.join('plots', f'{epoch}_meteoroids_3D_gt_{save_model_name}')
                plt.savefig(plot_dir)
                plt.close()

            
            # Increasing the batch size at fixed epochs
            if epoch in batch_size_increase_epochs:
                batch_size = batch_size * 4

            # Saving the model every n epochs and saving a backup in
            # case of a model corruption when interrupting
            if epoch % model_epochs == 0:
                model_data = {'epoch': epoch,
                              'dims': DIMENSIONS,
                              'whitening': model.whitening,
                              'leaky': model.leaky,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(model_data, save_dir)
                torch.save(model_data, save_dir_backup)
                print('Model saved!')
            
            # Saving a model checkpoint at a fixed number of epochs to go
            # back and test different versions
            if epoch % model_checkpoint_epochs == 0:
                model_data = {'epoch': epoch,
                              'dims': DIMENSIONS,
                              'whitening': model.whitening,
                              'leaky': model.leaky,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                checkpoint_dir = os.path.join('models', f'{save_model_name}_{epoch}.tar')
                torch.save(model_data, checkpoint_dir)
                print('Model saved!')

            # Checking whether or not to stop training
            if num_epochs is not None:
                if epoch == num_epochs + epoch_start:
                    model_data = {'epoch': epoch,
                                  'dims': DIMENSIONS,
                                  'whitening': model.whitening,
                                  'leaky': model.leaky,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(model_data, save_dir)
                    print('Model saved!')
                    break
            epoch += 1
    except KeyboardInterrupt:
        model_data = {'epoch': epoch,
                      'dims': DIMENSIONS,
                      'whitening': model.whitening,
                      'leaky': model.leaky,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_data, save_dir)
        print('Model saved!')

if perform_training is True and tensorboard_logging is True:
    writer.flush()

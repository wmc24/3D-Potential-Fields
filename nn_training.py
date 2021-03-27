import os
import torch
import myModel


"""-------------------------------------------------------------------------"""
"""--------------Setting up the Directory, parameters and GPUs--------------"""
"""-------------------------------------------------------------------------"""


DATA_DIR = 'dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

load_model = True
loaded_model_name = 'Debugging'
save_model_name = 'Debugging'
perform_training = True

# Number of epochs to train for, if None then until keyboard interrupt (ctrl+c)
num_epochs = 1 
learning_rate = 1e-5

# How many epochs to save things
model_epochs = 100
model_checkpoint_epochs = 2000
training_vis_epochs = 1


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
    writer = SummaryWriter(log_dir=os.path.join(EXPERIMENT_DIR, 'logs'))

json_dir = 'models'
save_dir = os.path.join('models', f'{save_model_name}.tar')
save_dir_backup = os.path.join('models', f'{save_model_name}_backup.tar')
load_dir = os.path.join('models', f'{loaded_model_name}.tar')
load_dir_backup = os.path.join('models', f'{loaded_model_name}_backup.tar')

print("Device Used:", DEVICE)


"""-------------------------------------------------------------------------"""
"""--------------------------Creating a Model-------------------------------"""
"""-------------------------------------------------------------------------"""


# Loading a network
model = myModel.Net()

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
"""-------------------------Training the Model------------------------------"""
"""-------------------------------------------------------------------------"""


if perform_training is True:
    print("Training")
    try:
        i = epoch + 1
        while True:
            print('\nEpoch: {}'.format(i))
            # INSERT TRAINING

            # Saving the model every n epochs and saving a backup in
            # case of a model corruption when interrupting
            if i % model_epochs == 0:
                model_data = {'epoch': epoch,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                torch.save(model_data, save_dir)
                torch.save(model_data, save_dir_backup)
                print('Model saved!')
            
            # Saving a model checkpoint at a fixed number of epochs to go
            # back and test different versions
            if i % model_checkpoint_epochs == 0:
                model_data = {'epoch': epoch,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()}
                checkpoint_dir = os.path.join('models', f'{save_model_name}_{i}.tar')
                torch.save(model_data, checkpoint_dir)
                print('Model saved!')

            # Checking whether or not to stop training
            if num_epochs is not None:
                if i == num_epochs + epoch:
                    model_data = {'epoch': epoch,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict()}
                    torch.save(model_data, save_dir)
                    print('Model saved!')
                    break
            i += 1
    except KeyboardInterrupt:
        model_data = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(model_data, save_dir)
        print('Model saved!')
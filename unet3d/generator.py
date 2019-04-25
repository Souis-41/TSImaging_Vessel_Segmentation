import copy
from random import shuffle
import numpy as np

def get_training_and_validation_generators(data_file, batch_size, data_split=0.8):
    
    
    training_list, validation_list = get_validation_split(data_file,data_split=0.8)
    training_generator = data_generator(data_file, training_list,batch_size=batch_size)
    validation_generator = data_generator(data_file, validation_list,batch_size=batch_size)

    num_training_steps = get_number_of_steps(len(training_list), batch_size)
    print("Number of training steps: ", num_training_steps)
    num_validation_steps = get_number_of_steps(len(validation_list), batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps

def get_number_of_steps(n_samples, batch_size):
    
    if np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1

def get_validation_split(data_file, data_split=0.8):
    
    print("Creating validation split...")
    nb_samples = data_file.root.data.shape[0]
    input_list = list(range(nb_samples))
    shuffle(input_list)
    n_training = int(len(input_list) * data_split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def data_generator(data_file, index_list, batch_size):
    
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)
        shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            data, truth = data_file.root.data[index], data_file.root.truth[index]
            x_list.append(data)
            y_list.append(truth)
            
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield np.asarray(x_list), np.asarray(y_list)
                x_list = list()
                y_list = list()

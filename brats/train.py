import sys
sys.path.append('..')
import tables
import os
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d, isensee2017_model
from brats.utils import get_callbacks

config = dict()
config["if_isensee2017_model"] = False
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (288, 288, 16)  # This determines what shape the images will be cropped/resampled to.
config["labels"] = (0,1,2,3,4,5)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["nb_channels"] = 4
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["batch_size"] = 2
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["data_file"] = os.path.abspath('data_storage.h5')
config["primary_pretrained_model"] = os.path.abspath("pt_tumor_segmentation_model.h5")
config["primary_model_file"] = os.path.abspath("pt_isensee_2017_model.h5")

def main():
    # convert input images into an hdf5 file
    data_file_opened = tables.open_file(config["data_file"], "r")
    
    if config["if_isensee2017_model"] == False:
        config["initial_learning_rate"] = 0.00001
        config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"])
    else:
        config["initial_learning_rate"] = 5e-4
        config["n_base_filters"] = 16
        config["model_file"] = os.path.abspath("isensee_2017_model.h5")
        
        model = isensee2017_model(input_shape=config["input_shape"], 
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              n_base_filters=config["n_base_filters"])

    # get training and testing generators
    train_generator, validation_generator, train_step, val_step = get_training_and_validation_generators(data_file_opened,
                                                                                                         config["batch_size"], 
                                                                                                         data_split=config["validation_split"])
    # run training
    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_step,
                        epochs=config["n_epochs"],
                        validation_data=validation_generator,
                        validation_steps=val_step,
                        workers=0,
                        callbacks=get_callbacks(config["model_file"],
                                                initial_learning_rate=config["initial_learning_rate"],
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]))

    data_file_opened.close()

if __name__ == "__main__":
    main()

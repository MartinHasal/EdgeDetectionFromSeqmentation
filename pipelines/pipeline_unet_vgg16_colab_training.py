import os

import pandas as pd
from cv2 import imread
import numpy as np

from pipelines.args import cli_argument_parser
from funcs.inference import predictDataset, predictImg, predictListOfFiles
from funcs.train import trainSegmentationModel
from models.unet import UNet
from procs.adapter import getDatasets
from utils.plots import plotTrainingHistory, plotColorizedVessels, plotPredictedImg, clean_image, plotListofImages
from utils.plots import plotHistogramImgSlicer, plotPredictedImgSlicer
from utils.timer import elapsed_timer
from utils.model import save_model
from utils.smooth_blender_predicitions import predict_img_with_smooth_windowing


def buildModel(input_shape, nclasses: int = 2, encoder_type: str = 'vgg16', trainable_encoder: bool = False):

    unet = UNet(input_shape, nclasses=nclasses, encoder_type=encoder_type, trainable_encoder=trainable_encoder)
    nn_unet = unet.model
    nn_unet.summary()

    return nn_unet


if __name__ == '__main__':

    kwargs = cli_argument_parser()

    # pipeline running
    
    with elapsed_timer('Creating datasets'):

        ds_train, ds_test = getDatasets(
            db_name=kwargs['db_name'],
            patch_size=kwargs['patch_size'],
            patch_overlap_ratio=kwargs['patch_overlap_ratio'],
            ds_test_ratio=kwargs['ds_test_ratio'],
            ds_augmentation_ratio=kwargs['ds_augmentation_ratio'],
            ds_augmentation_ratio_clahe=kwargs['clahe_augmentation_ratio'],
            ds_augmentation_ops=kwargs['ds_augmentation_ops'],
            crop_threshold=kwargs['crop_threshold']
        )

    with elapsed_timer('Build models'):

        IMG_SHAPE = (kwargs['patch_size'], kwargs['patch_size'], 3)
        NCLASSES = 2

        nn_unet_vgg16 = buildModel(IMG_SHAPE, NCLASSES, trainable_encoder=kwargs['trainable_encoder'])

    with elapsed_timer('Training model'):

        history = trainSegmentationModel(nn_model=nn_unet_vgg16,
                                         nclasses=NCLASSES,
                                         ds_train=ds_train,
                                         ds_val=ds_test,
                                         nepochs=kwargs['nepochs'],
                                         batch_size=kwargs['batch_size'],
                                         loss_type=kwargs['loss_type'],
                                         decay=kwargs['lr_decay_type'])

    # convert the history.history dict to a pandas DataFrame:
    df_history = pd.DataFrame(history.history)
    # plot training history
    plotTrainingHistory(df_history)
    
    # save to csv: 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)      


    # predict on test data set
    NSAMPLES = 4
    predictDataset(ds_test, nsamples_to_plot=NSAMPLES, nn_model=nn_unet_vgg16)

    # saving model
    OUTPUT_MODEL_PATH = kwargs['output_model_path']
    OUTPUT_MODEL_NAME = kwargs['output_model_name']

    if OUTPUT_MODEL_PATH is not None:
        with elapsed_timer('Saving model (UNetVGG16)'):
            save_model(nn_unet_vgg16, OUTPUT_MODEL_PATH, OUTPUT_MODEL_NAME)

    
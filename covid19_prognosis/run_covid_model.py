# Note this code is not adapted for recent Pytorch releases (Pytorch 1.6)

import numpy as np
import pandas as pd
import argparse, logging, random, os, torch, sys

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from covid19_prognosis.models import gmic
from covid19_prognosis.data_loading import image_loading
from covid19_prognosis.data_loading import structured_data_loading 
from lightgbm import LGBMClassifier

def covid_gmic_loss_function(labels, pred, model, parameters):
    """
    Loss function of the COVID-GMIC model
    :param labels:
    :param pred:
    :param model:
    :param parameters:
    :return:
    """
    # BCE loss
    bce = torch.nn.BCELoss()
    loss = bce(model.y_global, labels) + bce(model.y_local, labels) + bce(model.y_fusion, labels)
    # cam loss
    if parameters["beta"] != 0:
        loss += (parameters["beta"] * torch.abs(model.saliency_map).sum())
    return loss


def covid_gmic_drc_loss_function(labels, pred, model, parameters):
    """
    Loss function of the COVID-GMIC-DRC model
    :param labels:
    :param pred:
    :param model:
    :param parameters:
    :return:
    """
    # Negative log likelihood of the data
    sl, fl, _ = image_loading.discritize_time_array(labels[:,:2], parameters["buckets"])
    loss = -torch.mean(torch.log(torch.clamp(1+sl*(pred-1),1e-7,None)) + torch.log(torch.clamp(1 - fl*pred,1e-7,None)))
    # cam loss
    if parameters["beta"] != 0:
        loss += (parameters["beta"] * torch.abs(model.saliency_map).sum())
    return loss

def epoch(parameters, data_loader, model, loss_function, optimizer, device, training=True):
    """
    Performing a training / inference epoch
    :param parameters:
    :param data_loader:
    :param model:
    :param loss_function:
    :param optimizer:
    :param device:
    :param training:
    :return:
    """
    # create epoch document unit
    output_dict = {"index":[], "label":[], "prediction":[]}
    
    if parameters["drc"]:
        output_dict = {"index":[], "label_t2e":[], "label_event":[]}
        for x in parameters["buckets"]:
            output_dict["pred_"+str(x)] = []

    # update model phase
    if training:
        model.train()
    else:
        model.eval()

    # start the epoch
    for i, (index, imgs, labels) in enumerate(data_loader):
        # transform the images
        input_img_variable = Variable(imgs.to(device)).float()
        input_label_variable = Variable(labels.squeeze(1).to(device)).float()

        # forward propagation
        if training:
            pred = model(input_img_variable)
        else:
            with torch.no_grad():
                pred = model(input_img_variable)

        # calculate loss
        loss = loss_function(input_label_variable, pred, model, parameters)

        # backward propagaton
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update dictionary
        pred_val = pred.data.cpu().numpy()
        label_val = input_label_variable.data.cpu().numpy()
        index_val = index.data.cpu().numpy()
        if not parameters["drc"]:
            for j in range(len(index_val)):
                output_dict["index"].append(index_val[j])
                output_dict["label"].append(label_val[j])
                output_dict["prediction"].append(pred_val[j])
        else:
            for j in range(len(index_val)):
                output_dict["index"].append(index_val[j])
                output_dict["label_t2e"].append(label_val[j,0])
                output_dict["label_event"].append(label_val[j,1])
                for k,x in enumerate(parameters["buckets"]):
                    output_dict["pred_"+str(x)].append(pred_val[j,k])
        
        # log minibatch statistics
        logging.info("minibatch_number = {0}".format(i))
        logging.info("labels = {0}".format(label_val))
        logging.info("pred = {0}".format(pred_val))
            
    return pd.DataFrame(output_dict)

def start_experiment(parameters):
    """
    Entry level function that starts the experiment
    :param parameters:
    :return:
    """
    # set random seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    np.random.seed(parameters["seed"])
    if parameters["device_type"] == "gpu":
        torch.cuda.manual_seed(parameters["seed"])

    # create logger
    if parameters["output_path"] is not None:
        if not os.path.exists(parameters["output_path"]):
            os.mkdir(parameters["output_path"])
        logging.basicConfig(filename=os.path.join(parameters["output_path"], "log"), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("parameters = {0}".format(parameters))

    # create data loader
    data_set = image_loading.RandomImageLoader(parameters)
    data_loader = DataLoader(data_set, num_workers=parameters["number_of_loader"], batch_size=parameters["batch_size"], pin_memory=True)

    # create model
    model = gmic.GMIC(parameters)

    # allocate gpu
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["gmic_learning_rate"])
    loss_function = covid_gmic_loss_function
    if parameters["drc"]:
        loss_function = covid_gmic_drc_loss_function
        
    # start experiments
    logging.info("Start sample training epoch {}.".format(str(datetime.now())))
    training_results = epoch(parameters, data_loader, model, loss_function, optimizer, device, training=True)
    logging.info("Start sample inference epoch {}.".format(str(datetime.now())))
    inference_results = epoch(parameters, data_loader, model, loss_function, optimizer, device, training=False)
    if parameters["output_path"] is not None:
        if parameters["drc"]:
            train_out_path="training_results_drc.csv"
            inf_out_path="inference_results_drc.csv"
        else:
            train_out_path="training_results.csv"
            inf_out_path="inference_results.csv"
        training_results.to_csv(os.path.join(parameters["output_path"], train_out_path))
        inference_results.to_csv(os.path.join(parameters["output_path"], inf_out_path))
        logging.info("output saved to {} at {}".format(parameters["output_path"], str(datetime.now())))

def start_experiment_gbm(parameters):
    """
    Function that starts the experiments for the GBM model and stores the inference results 
    :param parameters:
    :return:
    """
    # generate a random training set 
    data_set = structured_data_loading.RandomDataLoader(parameters)
    X = data_set.features
    
    # For each class
    for c in range(0, parameters['number_classes']):
        y = [i.tolist()[c] for i in data_set.labels]

        # create model
        model = LGBMClassifier(n_estimators=parameters['gbm_n_estimators'],
            learning_rate=parameters['gbm_learning_rate'],num_leaves=parameters['gbm_num_leaves'],\
            colsample_bytree=.5,reg_alpha=1,reg_lambda=2, min_split_gain=.05, min_child_weight=5,
            silent=-1,verbose=-1, class_weight='balanced', random_state=parameters['seed'])

        # Train model
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:,1].ravel()
        output_dict_train = {"index":np.arange(len(y)), "label":y, "prediction":y_pred}

        # save results 
        training_inference_gbm = pd.DataFrame(output_dict_train)

        if parameters["output_path"] is not None:
            training_inference_gbm.to_csv(os.path.join(parameters["output_path"], 
                                                           "training_inference_results_GBM_{}.csv".format(c)),index=False)

if __name__ == "__main__":
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='sample script for training and evaluating models')
    parser.add_argument('--seed', type=int, default=random.randint(0, 99999))
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument('--drc', action="store_true", default=False)
    args = parser.parse_args()

    # construct the experiment parameters
    parameters = dict(
        seed = args.seed,                                       # Random seed to be used throughout the script.
        output_path = args.output_path,                         # Output path where all predictions will be saved.
        device_type = args.device_type,                         # Device type to use in heatmap generation and classifiers.
        gpu_number = args.gpu_number,                           # GPU number when multiple GPUs are available.
        number_of_loader=20,                                    # Number of subprocesses to use for data loading for torch.utils.data.DataLoader
        batch_size=4,                                           # Batch size.
        image_size=(1024, 1024),                                # Image size.
        data_num=10,                                            # Number of samples to generate in the random data loaders of the chest x-rays and clinical variables.
        # COVID-GMIC hyperparameters
        gmic_learning_rate=1e-5,                                # GMIC learning rate.
        number_classes=9 if args.drc else 4,                    # Number of classes (4 in the case of COVID-GMIC and 9 in the case of COVID-DRC)
        cam_size=(32, 32),                                      # Size of the saliency maps.
        percent_r=0.05,                                         # Hyperparameter representing the r% largest largest values in the saliency maps. 
        K=6,                                                    # Number of ROI patches.
        crop_shape=(224, 224),                                  # Size of ROI patches.
        beta=1e-5,                                              # Hyperparameter representing the weight on the l1-norm regularization on the saliency maps.
        # COVID-GMIC-DRC hyperparameters
        drc=args.drc,                                           # Boolean flag whether to run the outcome classification task (False) or the DRC task (True). 
        buckets=[0., 3., 12., 24., 48., 72., 96., 144., 192.],  # Time intervals for the survival analysis. 
        # COVID-GBM hyperparameters
        gbm_learning_rate=1e-2,                                 # GBM learning rate.
        gbm_n_estimators = 1000,                                # GBM number of estimators.
        gbm_num_leaves = 10,                                    # GBM number of leaves
        number_features = 10                                    # Number of features to generate in the random data loader of the clinical variables.
    )
    
    # start the experiments
    start_experiment(parameters)
    start_experiment_gbm(parameters)

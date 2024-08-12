'''A collection of my hand-written functions which can help save time and energy while coding.
'''
#Importing relevant dependencies
from typing import Any, Tuple, List, Dict
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchmetrics import Precision, Recall, F1Score, AUROC
from math import ceil
from tqdm.notebook import tqdm_notebook
import os
from pathlib import WindowsPath



def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



def forward_pass(model: nn.Module, features: torch.Tensor, model_type: str, device, memory_format: torch.memory_format = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a single forward pass of the features through the model.

    Args:
        model (nn.Module): The neural network model to perform the forward pass.
        features (torch.Tensor): The input features to be passed through the model.
        model_type (str): Type of the model, one of ['binary', 'multiclass', 'multilabel'].
        device: Device on which to perform the forward pass (e.g., 'cpu', 'cuda').
        memory_format (torch.memory_format, optional): Memory format for tensors. One of [torch.contiguous_format, torch.channels_last]. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing logits, probabilities, and predictions.

    Raises:
        ValueError: If the model_type or memory_format is invalid.
    """
    valid_model_types = ['binary', 'multiclass', 'multilabel']
    valid_memory_formats = [torch.contiguous_format, torch.channels_last]
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type '{model_type}'. Expected one of: {valid_model_types}")
    if memory_format is not None and memory_format not in valid_memory_formats:
        raise ValueError(f"Invalid memory_format '{memory_format}'. Expected one of: {valid_memory_formats}")

    model = model.to(device=device, memory_format=memory_format)
    features = features.to(device=device, memory_format=memory_format)
    if model_type == 'binary':
        logits = model(features).squeeze()
        prob = torch.sigmoid(logits)
        pred = torch.round(prob)
    elif model_type == 'multiclass':
        logits = model(features)
        prob = logits.softmax(dim=1)
        pred = torch.argmax(prob, dim=1)
    elif model_type == 'multilabel':
        print('Forward pass for multilabel model not yet made!')

    return logits, prob, pred



def eval_model(model: nn.Module, model_type: str, criterion, device, num_classes:int = 2, test_dataloader: DataLoader = None, X_test: torch.Tensor = None, y_test: torch.Tensor = None, memory_format: torch.memory_format = None, metrics:str = 'default', model_name:str = None) -> Dict[str, Any]:
    """
    Evaluates the model on the given test data and extracts meaningful metrics on the model's performance such as Accuracy and AUROC.

    Args:
        model (nn.Module): The neural network model to be evaluated.
        model_type (str): Type of the model, one of ['binary', 'multiclass', 'multilabel'].
        criterion: Loss function used for evaluation.
        device: Device on which to perform evaluation (e.g., 'cpu', 'cuda').
        num_classes (int, optional): Number of classes in the dataset. Default is 2 (for binary classification).
        test_dataloader (DataLoader, optional): DataLoader for testing data. Defaults to None.
        X_test (torch.Tensor, optional): Testing data tensor. Defaults to None.
        y_test (torch.Tensor, optional): Testing labels tensor. Defaults to None.
        memory_format (torch.memory_format, optional): Memory format for tensors. One of [torch.channels_last, torch.contiguous_format]. Defaults to None.
        metrics (str, optional): Metrics to calculate. Either 'default' to calculate Loss and Accuracy only or 'all' for additional metrics like AUROC. Defaults to 'default'.
        model_name (str, optional): Name of the model. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the model's name and the metrics obtained from evaluating the model.

    Raises:
        ValueError: If the model_type or metrics value is invalid.
    """
    model_name = model_name if model_name is not None else model.__class__.__name__
    model_type, metrics = model_type.lower(), metrics.lower()
    valid_model_types = ['binary', 'multiclass', 'multilabel']
    valid_metrics_values = ['default', 'all']
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type {model_type}. Expected one of: {valid_model_types}")
    if metrics not in valid_metrics_values:
        raise ValueError(f"Invalid value entered for parameter `metrics` {metrics}. Expected one of: {valid_metrics_values}")
    if metrics == 'all':
        precision_metric = Precision(task=model_type, num_classes=num_classes).to(device)
        recall_metric = Recall(task=model_type, num_classes=num_classes).to(device)
        f1score_metric = F1Score(task=model_type, num_classes=num_classes).to(device)
        auroc_metric = AUROC(task=model_type, num_classes=num_classes).to(device)
        f1score, auroc, precision, recall = 0.0, 0.0, 0.0, 0.0
    loss, acc = 0.0, 0.0
    model.eval()
    # Testing loop
    with torch.inference_mode():
        if test_dataloader is not None:
            for features, target in test_dataloader:
                logits, prob, pred = forward_pass(model=model, features=features, model_type=model_type, device=device, memory_format=memory_format)
                target = target.squeeze().to(device)
                loss += criterion(logits, target).detach().item()
                acc += accuracy_fn(y_true=target,y_pred=pred)
                if metrics == 'all':
                    f1score += f1score_metric(pred, target).item()
                    precision += precision_metric(pred, target).item()
                    recall += recall_metric(pred, target).item()
                    auroc += auroc_metric(prob, target).item()

            #Calculate the average test loss and average test accuracy per batch
            loss /= len(test_dataloader)
            acc /= len(test_dataloader)
            if metrics == 'all':
                f1score /= len(test_dataloader)
                auroc /= len(test_dataloader)
                precision /= len(test_dataloader)
                recall /= len(test_dataloader) 

        elif X_test is not None:
            logits, prob, pred = forward_pass(model=model, features=X_test, model_type=model_type, device=device, memory_format=memory_format)
            y_test = y_test.squeeze().to(device)
            loss = criterion(logits, y_test).item()
            acc = accuracy_fn(y_true=y_test,y_pred=pred)
            if metrics == 'all':
                f1score += f1score_metric(pred, y_test).item()
                precision += precision_metric(pred, y_test).item()
                recall += recall_metric(pred, y_test).item()
                auroc += auroc_metric(prob, y_test).item()

    return {'Model name': model_name, 'test_loss': loss, 'test_accuracy': acc, 'test_F1Score': f1score, 'test_auroc': auroc, 'test_precision': precision, 'test_recall': recall} if metrics == 'all' else {'Model name': model_name, 'test_loss': loss, 'test_accuracy': acc}



def train_step(model: nn.Module, model_type: str, criterion, optimizer: torch.optim.Optimizer, verbose: bool, device, train_dataloader: DataLoader = None, X_train: torch.Tensor = None, y_train: torch.Tensor = None, memory_format: torch.memory_format=None) -> Dict[str, Any]:
    """
    Performs a single training step on the model using the given DataLoader or training tensors and returns the training loss and accuracy.

    Args:
        model (nn.Module): The neural network model to be trained.
        model_type (str): Type of the model, one of ['binary', 'multiclass', 'multilabel'].
        criterion: Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        verbose (bool): Verbosity level (True = detailed, False = silent).
        device: Device on which to perform training (e.g., 'cpu', 'cuda').
        train_dataloader (DataLoader, optional): DataLoader for training data. Defaults to None.
        X_train (torch.Tensor, optional): Training data tensor. Defaults to None.
        y_train (torch.Tensor, optional): Training labels tensor. Defaults to None.
        memory_format (torch.memory_format, optional): Memory format for tensors. One of [torch.channels_last, torch.contiguous_format]. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the training loss and accuracy.

    Raises:
        ValueError: If the model_type is invalid or if the training data type is invalid.
    """
    model_type = model_type.lower()
    valid_model_types = ['binary', 'multiclass', 'multilabel']
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type '{model_type}'. Expected one of: {valid_model_types}")
    
    train_loss, train_acc = 0.0, 0.0
    model.train()
    if train_dataloader is not None:
        for train_features, train_target in train_dataloader:
            logits, prob, pred = forward_pass(model=model, features=train_features, model_type=model_type, device=device, memory_format=memory_format)
            train_target = train_target.squeeze().to(device)
            loss = criterion(logits, train_target)
            acc = accuracy_fn(y_true=train_target,y_pred= pred)
            train_loss += loss.detach().item()
            train_acc += acc

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        #Calculate the average training loss and average training accuracy per batch
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)


    elif X_train is not None:
        logits, prob, pred = forward_pass(model=model, features=X_train, model_type=model_type, device=device, memory_format = memory_format)
        y_train = y_train.squeeze().to(device)
        loss = criterion(logits, y_train)
        acc = accuracy_fn(y_true=y_train,y_pred= pred)
        train_loss = loss.detach().item()
        train_acc = acc

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    else:
        raise ValueError(f'Invalid train_data type of: `{train_dataloader.__class__ if train_dataloader is not None else X_train.__class__}` inputed. Type: `torch.Tensor` or `torch.utils.data.DataLoader` expected')

    if verbose:
        print(f"Loss: {train_loss:.5f} | Accuracy: {train_acc:.2f}%")

    return {'train_loss': train_loss, 'train_accuracy': train_acc}



def fit_model(epochs: int, model: nn.Module, model_type: str, criterion, optimizer: torch.optim.Optimizer, verbose:int, device, train_dataloader: DataLoader = None, X_train: torch.Tensor = None, y_train: torch.Tensor = None, memory_format: torch.memory_format = None) -> Dict[str, List]:
    """
    Trains a neural network model for a specified number of epochs.

    Args:
        epochs (int): Number of epochs to train the model.
        model (nn.Module): The neural network model to be trained.
        model_type (str): Type of the model (e.g., 'multiclass', 'binary', 'multilabel').
        criterion: Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        verbose (int): Verbosity level (0 = silent, 1 = progress, 2 = detailed).
        device: Device on which to perform training (e.g., 'cpu', 'cuda').
        train_dataloader (DataLoader, optional): DataLoader for training data. Defaults to None.
        X_train (torch.Tensor, optional): Training data tensor. Defaults to None.
        y_train (torch.Tensor, optional): Training labels tensor. Defaults to None.
        memory_format (torch.memory_format, optional): Memory format for tensors (e.g 'torch.channels_last', 'torch.contiguous_format'). Defaults to None.

    Returns:
        Dict[str, List]: Dictionary containing training loss and accuracy history.
    """
    print_intervals = ceil(epochs/10)
    model_type = model_type.lower()
    model.train()
    train_history = {'train_loss':[], 'train_accuracy':[]}
    for epoch in tqdm_notebook(range(epochs)):
        if train_dataloader is not None:
            train_loss, train_acc = train_step(model=model, model_type=model_type, criterion=criterion, optimizer=optimizer, train_dataloader=train_dataloader,verbose=False, device=device, memory_format=memory_format)
        elif X_train is not None:
            train_loss, train_acc = train_step(model=model, model_type=model_type, criterion=criterion, optimizer=optimizer, X_train=X_train, y_train=y_train,verbose=False, device=device, memory_format=memory_format)

        train_history['train_loss'].append(train_loss)
        train_history['train_accuracy'].append(train_acc)

        # Print out what's happening
        if verbose == 1:
            if epoch % print_intervals == 0 or print_intervals == 1:
                print(f"Epoch: {epoch} | Loss: {train_loss:.5f} | Accuracy: {train_acc:.2f}%")
        elif verbose == 2:
            print(f"Epoch: {epoch} | Loss: {train_loss:.5f} | Accuracy: {train_acc:.2f}%")

    return train_history



def combine_metrics_dicts(metrics_dicts: List)->Dict[str, Any]:
    """
    Combines individual dictionaries (which have been put into a single `List`) containing any number of metrics into a single dictionary.

    Args:
        metrics_dicts (List): A list containing dictionaries which all have the same keys: the metrics.

    Returns:
        Dict[str, Any]: A single dictionary where each key is a metric and the value is a list of all values for that metric combined from the input dictionaries.

    Raises:
        ValueError: If a metric in any of the input dictionaries does not match the expected metrics.
    """
    metrics_dict = {metric:[] for metric in metrics_dicts[0].keys()}
    for i in range(len(metrics_dicts)):
        for metric, value in metrics_dicts[i].items():
            if metric not in metrics_dict.keys():
                raise ValueError(f'Invalid metric of name: {metric} in `Dict` {metrics_dicts[i]} inputed. Expected one of: {metrics_dict.keys()}')
            metrics_dict[metric].append(value)
    return metrics_dict



def train_and_test_model(epochs: int, model: nn.Module, model_type: str, num_classes:int, criterion:nn.Module, optimizer: torch.optim.Optimizer, verbose:bool, device, model_name: str = None, train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,X_train: torch.Tensor = None, y_train: torch.Tensor = None, X_test: torch.Tensor = None, y_test: torch.Tensor = None, memory_format: torch.memory_format = None, metrics: str = 'default', checkpoint_path = "C:/dev/model_checkpoints") -> Tuple[Dict, Dict]:
    """
    Trains and tests a neural network model for a specified number of epochs.

    Args:
        epochs (int): Number of epochs to train the model.
        model (nn.Module): The neural network model to be trained and tested.
        model_type (str): Type of the model (e.g., 'binary', 'multiclass', 'multilabel').
        num_classes (int): Number of classes for classification.
        criterion (nn.Module): Loss function used for training and testing.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        verbose (bool): Verbosity level (False = silent, True = detailed).
        device: Device on which to perform training and testing (e.g., 'cpu', 'cuda').
        model_name (str, optional): Name of the model. Defaults to None.
        train_dataloader (DataLoader, optional): DataLoader for training data. Defaults to None.
        test_dataloader (DataLoader, optional): DataLoader for testing data. Defaults to None.
        X_train (torch.Tensor, optional): Training data tensor. Defaults to None.
        y_train (torch.Tensor, optional): Training labels tensor. Defaults to None.
        X_test (torch.Tensor, optional): Testing data tensor. Defaults to None.
        y_test (torch.Tensor, optional): Testing labels tensor. Defaults to None.
        memory_format (torch.memory_format, optional): Memory format for tensors (e.g 'torch.channels_last', 'torch.contiguous_format'). Defaults to None.
        metrics (str, optional): Metrics to evaluate (e.g 'default' for Loss and Accuracy metrics only, 'all' for additional metrics like AUROC). Defaults to 'default'.
        checkpoint_path (str, optional): Path to save model checkpoints. Defaults to "C:/dev/model_checkpoints".

    Returns:
        Tuple[Dict, Dict]: Two dictionaries containing the training and testing metrics history, respectively.

    Raises:
        ValueError: If the checkpoint path is not acceptable.
    """
    model_name = model_name if model_name is not None else model.__class__.__name__
    train_metrics_list = []
    test_metrics_list = []
    best_test_loss = float('inf')

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    model_chkpt_path = checkpoint_path/model_name if type(checkpoint_path) is WindowsPath else checkpoint_path + '/' + model_name if type(checkpoint_path) is str else None

    if model_chkpt_path is None:
        raise ValueError('Inputed checkpoint path not acceptable, expected one of type [str, pathlib.WindowsPath]')

    os.mkdir(model_chkpt_path) if not os.path.isdir(model_chkpt_path) else None
    last_chkpt_path = None

    if os.listdir(model_chkpt_path) != []:
        last_chkpt_path = model_chkpt_path/os.listdir(model_chkpt_path)[0] if type(checkpoint_path) is WindowsPath else model_chkpt_path + '/' + os.listdir(model_chkpt_path)[0]

    if train_dataloader is not None:
        for epoch in tqdm_notebook(range(epochs)):
            print('Epoch: ', epoch)
            train_metrics = train_step(model=model,
            model_type=model_type, criterion=criterion, optimizer=optimizer, verbose=verbose, train_dataloader=train_dataloader, device=device, memory_format=memory_format)
            train_metrics_list.append(train_metrics)
            test_metrics = eval_model(model=model, model_type=model_type, criterion=criterion, num_classes=num_classes, test_dataloader=test_dataloader, device=device, memory_format=memory_format, metrics=metrics, model_name=model_name)
            if test_metrics['test_loss'] < best_test_loss:
                best_test_loss = test_metrics['test_loss']
                os.remove(last_chkpt_path) if last_chkpt_path is not None else None
                last_chkpt_path = model_chkpt_path/f'epoch_{epoch+1}_loss_{best_test_loss:.3f}.pth' if type(checkpoint_path) is WindowsPath else f'{model_chkpt_path}/epoch_{epoch+1}_loss_{best_test_loss:.3f}.pth'
                torch.save(model.state_dict(), f=last_chkpt_path)
            test_metrics_list.append(test_metrics)
            print(test_metrics,'\n--------------------')
    elif X_train is not None:
        for epoch in tqdm_notebook(range(epochs)):
            print('Epoch: ', epoch)
            train_metrics = train_step(model=model,
            model_type=model_type, criterion=criterion, optimizer=optimizer, verbose=verbose, X_train=X_train, y_train=y_train, device=device, memory_format=memory_format)
            train_metrics_list.append(train_metrics)
            test_metrics = eval_model(model=model, model_type=model_type, criterion=criterion, num_classes=num_classes, X_test=X_test, y_test=y_test, device=device, memory_format=memory_format, metrics=metrics, model_name= model_name)
            if test_metrics['test_loss'] < best_test_loss:
                best_test_loss = test_metrics['test_loss']
                os.remove(last_chkpt_path) if os.listdir(model_chkpt_path) != [] else None
                last_chkpt_path = model_chkpt_path/f'epoch_{epoch+1}_loss_{best_test_loss:.3f}.pth' if type(checkpoint_path) is WindowsPath else f'{model_chkpt_path}/epoch_{epoch+1}_loss_{best_test_loss:.3f}.pth'
                torch.save(model.state_dict(), f=last_chkpt_path)
            test_metrics_list.append(test_metrics)
            print(test_metrics,'\n--------------------')

    return combine_metrics_dicts(train_metrics_list), combine_metrics_dicts(test_metrics_list)



def plot_metrics_history(train_metrics_dict, test_metrics_dict, comment: str = None):
    """
    Plots the training and testing metrics history.

    Args:
        train_metrics_dict (dict): Dictionary containing training metrics history.
        test_metrics_dict (dict): Dictionary containing testing metrics history.
        comment (str, optional): Comment to be added as the title of the plot. Defaults to None.

    Returns:
        None
    """
    train_metrics = list(train_metrics_dict.keys())
    test_metrics = list(test_metrics_dict.keys())
    plt.figure(figsize=(12,6))
    for i in range(len(train_metrics)):
        plt.subplot(1,2,i+1)
        plt.plot(train_metrics_dict[train_metrics[i]], label=train_metrics[i])
        plt.plot(test_metrics_dict[test_metrics[i+1]], label=test_metrics[i+1])
        plt.xlabel('Epoch')
        plt.ylabel(f'{train_metrics[i]} & {test_metrics[i+1]}')
        plt.legend()
    if comment is not None:
        plt.suptitle(comment)



def compare_models(*model_metrics) -> pd.DataFrame:
    """
    Compares multiple models by creating a DataFrame from their metrics.

    Args:
        *model_metrics: Variable length argument list of dictionaries, each containing metrics for a model.

    Returns:
        pd.DataFrame: A DataFrame where each row represents a model and each column represents a metric. The DataFrame is indexed by the model names.
    """
    compare_models_df = pd.DataFrame(model_metrics, columns=model_metrics[0].keys())
    compare_models_df.set_index('Model_name', inplace=True)
    return compare_models_df

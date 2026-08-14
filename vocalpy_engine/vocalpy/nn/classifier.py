# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import torch

import numpy as np
import torch.nn as nn
import torchvision.models as models

from torch.nn.functional import softmax

from vocalpy.errors import ValidationError
from vocalpy.utils.io import load_checkpoint
from vocalpy.nn import datasets
from vocalpy.nn.pretrained_models import get_pretrained_model_spec, validate_pretrained_model_file


class VocalClassifier(object):
    """
    Vocalization classifier

    Parameters
    ----------
    network_type : str
        vocal classifier network_type ('noise', or 'class')
    source : str or numpy.ndarray
        path to directory with spectrograms or array with data to be classified
    batch_size : str, optional
        batch size to use with the neural network
    path_to_checkpoint : str, optional
        path to checkpoint to laod pretrained neural network model
    """

    def __init__(self, network_type, source, batch_size=32, path_to_checkpoint=None):
        if network_type not in ["noise", "class"]:
            raise ValidationError(
                "VocalClassifier network_type must be 'noise' or 'class'. "
                f"provided value: {network_type}"
            )
        self.network_type = network_type

        # self.source = source
        self.batch_size = batch_size
        self.path_to_checkpoint = path_to_checkpoint

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        if self.network_type == "noise":
            self.model = self.load_pretrained_noise_model(self.device, self.path_to_checkpoint)
        else:
            self.model = self.load_pretrained_class_model(self.device, self.path_to_checkpoint)

        self.dataset = self.create_dataset(source)
        self.dataloader = datasets.create_dataloader(self.dataset, self.batch_size) if len(self.dataset) > 0 else []

    @staticmethod
    def build_mobilenet_v2_classifier(num_classes):
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 1024), nn.ReLU(inplace=True), nn.Linear(1024, num_classes),
        )
        return model

    def _load_pretrained_model(self, device, model_path, network_type):
        spec = get_pretrained_model_spec(network_type)
        resolved_model_path = spec.path if model_path is None else model_path
        expected_sha256 = spec.sha256 if model_path is None else None
        self.checkpoint_sha256 = validate_pretrained_model_file(resolved_model_path, expected_sha256=expected_sha256)
        self.checkpoint_path = str(resolved_model_path)

        classifier_model = self.build_mobilenet_v2_classifier(spec.num_classes)
        load_checkpoint(self.checkpoint_path, classifier_model, device)
        classifier_model = classifier_model.to(device)
        classifier_model.eval()

        self.classes = list(spec.classes)
        return classifier_model

    def load_pretrained_noise_model(self, device, model=None):
        """
        Loads pretrained Class CNN model by default, trained to classify spectrograms
        as Vocal or Noise; or model at path provided by the user

        Parameters
        ----------
        device : torch.device
            device to run (CPU or GPU)
        model : str, optional
            path to checkpoint for a neural network model
        """
        return self._load_pretrained_model(device=device, model_path=model, network_type="noise")

    def load_pretrained_class_model(self, device, model=None):
        """
        Loads pretrained Class CNN model by default, trained to classify spectrograms
        as one of eleven classes:
            chevron, complex, down_fm, flat, mult_steps, rev_chevron,
            short, step_down, step_up, two_steps, up_fm
        or model at path provided by the user

        Parameters
        ----------
        device : torch.device
            device to run (CPU or GPU)
        model : str, optional
            path to checkpoint for a neural network model
        """
        return self._load_pretrained_model(device=device, model_path=model, network_type="class")

    def create_dataset(self, source):
        """
        Creates a dataset by instantiating the VocalDatasetFromFolder class

        Parameters
        ----------
        source : str or numpy.ndarray
            if path -> directory that contains the spectrogram images used to create the dataset
            if ndarray -> return dataset from array
        """
        if isinstance(source, np.ndarray):
            return datasets.VocalDatasetFromArray(source)

        return datasets.VocalDatasetFromFolder(source)

    def empty_predictions(self):
        if self.network_type == "noise":
            return np.asarray([], dtype=bool)
        return np.empty((0, len(self.classes)), dtype=float)

    def classify_list_of_vocals(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        # -- is list of vocals is empty, just return
        if list_of_vocals.number_of_vocals < 1:
            return self.empty_predictions()

        if self.network_type == "noise":
            return self.classify_list_of_vocals_noise(list_of_vocals)

        return self.classify_list_of_vocals_class(list_of_vocals)

    def classify_list_of_vocals_class(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` into vocal classes using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        predictions = []

        # compute metrics over the dataset
        with torch.no_grad():
            for itr, image in enumerate(self.dataloader):
                image = image.to(self.device)
                score = self.model(image)
                predicted = softmax(score, dim=1)
                predictions.append(predicted.cpu().numpy())

        return np.vstack(predictions)

    def classify_list_of_vocals_noise(self, list_of_vocals):
        """
        Classify a :class:`ListOfVocals` as Vocal or Noise using a Neural Network

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`
            list of vocals to be classified
        """
        predictions = []

        # compute metrics over the dataset
        with torch.no_grad():
            for itr, image in enumerate(self.dataloader):
                image = image.to(self.device)
                score = self.model(image)
                _, predicted = torch.max(score, 1)
                predictions.append(predicted.cpu().numpy())

        return np.hstack(predictions).astype("bool")

# -*- coding: utf-8 -*-
"""Neural-network segmentation for individual vocal spectrogram crops."""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from vocalpy.errors import ValidationError
from vocalpy.nn.datasets import build_image_transform, create_dataloader
from vocalpy.nn.pretrained_models import get_pretrained_model_spec, validate_pretrained_model_file
from vocalpy.nn.squeakout import DEFAULT_IMAGE_SIZE, load_squeakout_checkpoint


class VocalSegmenter(object):
    """
    Segment individual vocal spectrogram crops with a PyTorch model.

    The segmenter operates on the same per-vocal crop surface used by the
    classifier. The model is expected to return a binary mask logits/probability
    map for each crop.
    """

    def __init__(self, source, batch_size=32, path_to_model=None, threshold=None, model=None):
        self.batch_size = batch_size
        self.path_to_model = path_to_model
        self.model_spec = self.resolve_model_spec()
        self.input_shape = tuple(self.model_spec.input_shape)
        self.prediction_type = self.model_spec.prediction_type or "logits"
        self.threshold = self._resolve_threshold(threshold)

        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")

        self.model = self.load_segmentation_model(self.device, path_to_model=path_to_model, model=model)
        self.dataset = self.create_dataset(source)
        self.dataloader = create_dataloader(self.dataset, self.batch_size) if len(self.dataset) > 0 else []
        self.output_shape = (self.dataset.height, self.dataset.width)

    def resolve_model_spec(self):
        return get_pretrained_model_spec("segment")

    @staticmethod
    def _validate_threshold(threshold):
        if threshold <= 0 or threshold >= 1:
            raise ValidationError(f"segmentation threshold must be between 0 and 1. provided value: {threshold}")
        return threshold

    def _resolve_threshold(self, threshold):
        threshold = self.model_spec.default_threshold if threshold is None else float(threshold)
        if threshold is None:
            raise ValidationError("segmentation threshold is not configured for the selected model")
        return self._validate_threshold(threshold)

    def load_segmentation_model(self, device, path_to_model=None, model=None):
        if not hasattr(self, "model_spec"):
            self.model_spec = self.resolve_model_spec()
            self.input_shape = tuple(self.model_spec.input_shape)
            self.prediction_type = self.model_spec.prediction_type or "logits"

        if model is not None:
            if not isinstance(model, nn.Module):
                raise ValidationError(
                    "segmentation model must resolve to a torch.nn.Module. "
                    f"received type: {model.__class__.__name__}"
                )
            return model.to(device).eval()

        resolved_model_path = self.model_spec.path if path_to_model is None else path_to_model
        expected_sha256 = self.model_spec.sha256 if path_to_model is None else None
        self.checkpoint_sha256 = validate_pretrained_model_file(
            resolved_model_path,
            expected_sha256=expected_sha256,
        )
        self.checkpoint_path = str(resolved_model_path)
        if self.model_spec.architecture != "squeakout":
            raise ValidationError(
                "unsupported bundled segmentation architecture. "
                f"provided value: {self.model_spec.architecture}"
            )
        return load_squeakout_checkpoint(resolved_model_path, device=device)

    def create_dataset(self, source):
        if not isinstance(source, np.ndarray):
            raise ValidationError("segmentation source must be a numpy.ndarray of vocal spectrogram crops")
        return VocalSegmentationDatasetFromArray(source, image_size=self.input_shape[-2:])

    def empty_predictions(self):
        height, width = self.output_shape
        return np.empty((0, height, width), dtype=np.uint8)

    def segment_list_of_vocals(self, list_of_vocals):
        if list_of_vocals.number_of_vocals < 1:
            return self.empty_predictions()

        predictions = []
        with torch.no_grad():
            for image in self.dataloader:
                image = image.to(self.device)
                probabilities = self._prediction_to_probability_map(self.model(image))
                predictions.append(self._resize_and_threshold(probabilities))

        return np.vstack(predictions) if predictions else self.empty_predictions()

    def _prediction_to_probability_map(self, prediction):
        if prediction.ndim == 4:
            if prediction.shape[1] == 1:
                return self._to_probability_map(prediction[:, 0])
            if prediction.shape[1] == 2:
                return torch.softmax(prediction, dim=1)[:, 1]

        if prediction.ndim == 3:
            return self._to_probability_map(prediction)

        raise ValidationError(
            "segmentation model output must have shape (N,H,W), (N,1,H,W), or (N,2,H,W). "
            f"received shape: {tuple(prediction.shape)}"
        )

    def _to_probability_map(self, prediction):
        if self.prediction_type == "probabilities":
            return prediction
        if self.prediction_type == "logits":
            return torch.sigmoid(prediction)
        raise ValidationError(
            "unsupported segmentation prediction_type. "
            f"provided value: {self.prediction_type}"
        )

    def _resize_and_threshold(self, prediction):
        if self.output_shape == (0, 0):
            return np.empty((prediction.shape[0], 0, 0), dtype=np.uint8)

        resized = F.interpolate(
            prediction.unsqueeze(1),
            size=self.output_shape,
            mode="nearest",
        ).squeeze(1)
        return ((resized >= self.threshold).to(dtype=torch.uint8) * 255).cpu().numpy()


class VocalSegmentationDatasetFromArray(torch.utils.data.Dataset):
    def __init__(self, data, image_size=DEFAULT_IMAGE_SIZE, transform=None):
        self.data = np.asarray(data, dtype=np.uint8)
        if self.data.size == 0:
            self.len = 0
            self.height = 0
            self.width = 0
        else:
            self.len, self.height, self.width = self.data.shape
        self.transform = transform or transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                *build_image_transform(image_size=image_size, repeat_channels=False).transforms,
            ]
        )

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return len(self.data)

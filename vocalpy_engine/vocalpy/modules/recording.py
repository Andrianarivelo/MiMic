# -*- coding: utf-8 -*-
"""VocalPy - Vocal analysis framework"""

__license__ = "Apache License, Version 2.0"
__copyright__ = "2020 Dietrich Lab - Yale University School of Medicine"


from time import time
from logging import getLogger
from joblib import Parallel, delayed
from os.path import join, split, splitext

from vocalpy.errors import ConfigurationError, RecordingStateError
from vocalpy.modules.audio import Audio
from vocalpy.modules.list_of_vocals import ListOfVocals
from vocalpy.configs.configs import load_user_parameters, write_user_parameters
from vocalpy.utils.misc import create_dataframe_from_list_of_vocals
from vocalpy.utils.io import (
    write_pickle_file,
    load_pickle_file,
    create_directory,
    remove_directory,
    save_dataframe_as_csv,
    get_output_directory_for_audio_file,
)


class Recording(object):
    """
    Recording Object which contains auxiliary functions to process the recording
    Concentrates all objects associated with the recording, including: :class:`Audio` and :class:`ListOfVocals`

    Parameters
    ----------
    recording_path : str
        Path to audio file to open
    args : args
        Struct with arguments that customizes the execution with parameters such as:
        animal pipeline, bin size, frequency range, threads for parallelization, verbose output

    Returns
    -------
    recording : Object
        The recording object containing the read audio file and its metadata

    Examples
    --------
    >>> audio_path = 'path/to/audio'
    >>> args = p.parse_args()
    >>> recording = Recording(recording_path=audio_path, args=args)
    """

    def __init__(self, recording_path, args):
        self.recording_path = recording_path
        self.recording_dir = None
        self.recording_name = None
        self.output_dir = None
        self.spectrogram_dir = None
        self.mask_dir = None
        self.create_paths(recording_path)
        self.params = self.create_user_parameters_yaml(args)
        self.audio = Audio(self.recording_path, self.output_dir, self.spectrogram_dir, self.mask_dir, self.params["bin_size"])
        self._group_name = "not set"
        self._list_of_vocals = None
        self._has_list_of_vocals = None
        self._animal_name = args.animal
        self._animal = self.create_animal_pipeline()

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n duration: {self.audio.audio_duration} \n sampling rate: {self.audio.sampling_rate}"
        )

    @property
    def has_list_of_vocals(self):
        """
        Checks if recording has a list of vocals

        Returns
        ------
        _has_list_of_vocals : Bool
        """
        return self._has_list_of_vocals

    @has_list_of_vocals.setter
    def has_list_of_vocals(self, new_has_list_of_vocals):
        self._has_list_of_vocals = new_has_list_of_vocals

    @property
    def list_of_vocals(self):
        """
        Returns the list of vocals for this recording

        Returns
        ------
        _list_of_vocals : :class:`ListOfVocals`
        """
        return self._list_of_vocals

    @list_of_vocals.setter
    def list_of_vocals(self, new_list_of_vocals):
        self._list_of_vocals = new_list_of_vocals

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, new_group_name):
        self._group_name = new_group_name

    def save_recording_object(self, path, filename="recording"):
        write_pickle_file(self, filename, path, object_type="recording")

    def create_user_parameters_yaml(self, args):
        user_params_yaml = load_user_parameters(args)
        write_user_parameters(user_params_yaml, self.output_dir)
        return user_params_yaml

    def create_paths(self, recording_path):
        """
        Creates directory structure metadata for output files.
        Image subdirectories are created lazily by the save paths that
        actually write into them.

        Parameters
        ----------
        recording_path : str
        """
        basepath, filename = split(recording_path)
        self.recording_dir = basepath
        self.recording_name = filename
        filename = splitext(filename)[0]
        self.output_dir = get_output_directory_for_audio_file(recording_path)
        self.spectrogram_dir = join(self.output_dir, "spectrogram")
        self.mask_dir = join(self.output_dir, "mask")

        create_directory(self.output_dir)

    def create_animal_pipeline(self):
        """
        Creates an instance of :class:`Animal` that implements the pipeline for the animal selected by the user

        Parameters
        ----------
        animal : str
            animal pipeline selected by the user

        Returns
        -------
        class instance : :class:`Animal`
            animal pipeline selected by the user
        """
        import importlib

        if not self.params["identifier"]:
            raise ConfigurationError(f"animal pipeline is not available for {self._animal_name}")

        AnimalClass = getattr(importlib.import_module("vocalpy.pipelines." + self._animal_name), self._animal_name.title())

        return AnimalClass(self._animal_name.lower(), self.params)

    def _require_list_of_vocals(self, list_of_vocals=None):
        resolved_list_of_vocals = list_of_vocals if list_of_vocals is not None else self._list_of_vocals
        if resolved_list_of_vocals is None:
            raise RecordingStateError("recording has no list of vocals")
        return resolved_list_of_vocals

    def identify_vocalizations(self):
        """
        Process recording by calling appropriate animal pipeline functions
        """

        logger = getLogger()
        timeAParallel = time()

        # -- distribute Recording chunks to available cores
        # -- process each chunk and find candidate vocalizations
        results = Parallel(n_jobs=self.params["threads"], require="sharedmem")(
            delayed(self._animal.identify_vocalizations)(chunk=i) for i in self.audio.chunks
        )

        # -- create list of vocals found in the recording
        logger.info("combining list of vocals from each bin")
        timeAcombining = time()
        list_of_vocals = ListOfVocals()
        list_of_vocals.combine_list_of_list_of_vocals(list_of_list_of_vocals=results)

        # -- only needs to fix first vocal in each segment
        list_of_vocals.update_intervals()
        self._has_list_of_vocals = True
        self.list_of_vocals = list_of_vocals
        logger.info(f"done combining ({time() - timeAcombining:.0f}s)")
        logger.info(list_of_vocals)

        self.identify_vocalizations_finished()
        logger.info(
            f"recording parallel processing ({(time() - timeAParallel) // 60:.0f}m {(time() - timeAParallel) % 60:.0f}s)"
        )

    def identify_vocalizations_finished(self):
        """
        Recording has already been processed -> clear segments
        """
        self.audio.chunks = None
        return 0

    def classify_vocalizations(self):
        """
        Classify identified vocalizations using the appropiate animal pipeline
        """
        logger = getLogger()

        if self.params["classifier"]:

            # -- classify candidate vocalizations as Vocal or Noise; remove Noise
            logger.info("classifying candidate vocalizations as vocal or noise")
            timeAclassification = time()
            predictions, classes = self._animal.classify_vocalizations(
                network_type="noise", list_of_vocals=self.list_of_vocals
            )
            logger.info("removing candidates classified as noise")
            self.remove_vocals_classified_as_noise_from_list_of_vocals(predictions)
            logger.info(f"done classifying and removing ({time() - timeAclassification:.0f}s)")
            logger.info(self._list_of_vocals)

            # -- classify vocalizations into vocal types
            logger.info("classifying vocalizations")
            timeAclassification = time()
            predictions, classes = self._animal.classify_vocalizations(
                network_type="class", list_of_vocals=self.list_of_vocals
            )
            logger.info("adding classification to vocals")
            self.update_vocals_with_class_classification(predictions, classes)
            logger.info(f"done classifying and updating vocals ({time() - timeAclassification:.0f}s)")
        else:
            logger.info(f"no classifier available for animal type: {self._animal._animal}")

    def segment_vocalizations(self):
        """
        Segment identified vocalizations using the configured neural network model.
        """
        logger = getLogger()

        if not self.params.get("segmenter", False):
            logger.info(f"no segmenter available for animal type: {self._animal._animal}")
            return 0

        list_of_vocals = self._require_list_of_vocals()

        logger.info("segmenting vocalizations")
        time_segmentation = time()
        predictions = self._animal.segment_vocalizations(
            list_of_vocals=list_of_vocals,
            path_to_model=self.params.get("segmentation_model_path"),
            threshold=self.params.get("segmentation_threshold"),
        )
        logger.info("adding segmentation masks to vocals")
        self.update_vocals_with_segmentation_masks(predictions)
        logger.info(f"done segmenting vocals ({time() - time_segmentation:.0f}s)")
        return 0

    def load_list_of_vocals(self):
        """
        Loads :class:`ListOfVocals`  from a python object file
        """
        return load_pickle_file("list_of_vocals", self.output_dir, expected_object_type="list_of_vocals")

    def save_recording_data_to_csv(self, list_of_vocals=None, path=None):
        """
        Saves recording metadata to a CSV file. The file will contain information
        regarding each vocalization identified in the recording.

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        path = path if path is not None else self.output_dir

        if list_of_vocals.intervals_fixed is False:
            list_of_vocals.update_intervals()

        recording_df = create_dataframe_from_list_of_vocals(list_of_vocals)

        save_dataframe_as_csv(dataframe=recording_df, path=path, filename=self.recording_name)
        return 0

    def save_spectrograms(self, list_of_vocals=None, path=None):
        """
        Saves the spectrogram images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram"))
        create_directory(join(path, "spectrogram"))
        list_of_vocals.save_spectrograms(output_dir=path)
        return 0

    def save_validation_images(self, list_of_vocals=None, path=None):
        """
        Saves the spectrogram overlaid with the segmentation images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram_validation"))
        create_directory(join(path, "spectrogram_validation"))
        list_of_vocals.save_validation_images(output_dir=path)
        return 0

    def save_spectrograms_and_masks(self, list_of_vocals=None, path=None):
        """
        Saves the spectrograms and segmentation images to the output directory for this Recording Object

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "spectrogram"))
        create_directory(join(path, "spectrogram"))
        list_of_vocals.save_spectrograms(output_dir=path)
        remove_directory(join(path, "mask"))
        create_directory(join(path, "mask"))
        list_of_vocals.save_masks(output_dir=path)
        return 0

    def save_cnn_masks(self, list_of_vocals=None, path=None):
        """
        Saves the neural-network segmentation masks to the output directory.

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        path : str, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        path = path if path is not None else self.output_dir
        remove_directory(join(path, "cnn_mask"))
        create_directory(join(path, "cnn_mask"))
        list_of_vocals.save_cnn_masks(output_dir=path)
        return 0

    def remove_spectrograms_and_masks_from_object(self, list_of_vocals=None):
        """
        Removes transient image payloads from the in-memory Vocal objects.

        Parameters
        ----------
        list_of_vocals : :class:`ListOfVocals`, optional
        """
        list_of_vocals = self._require_list_of_vocals(list_of_vocals)
        list_of_vocals.remove_visual_data()
        return 0

    def create_dataset(self, list_of_vocals=None):
        """
        Creates an image dataset to be used by the Neural Networks
        """
        # ToDo
        # create from list of vocals or save spectrograms
        # create filename list etc and create from folder path
        self._require_list_of_vocals(list_of_vocals)
        raise NotImplementedError("create_dataset is not implemented")

    def remove_vocals_classified_as_noise_from_list_of_vocals(self, predictions):
        """
        Removes vocals that were classified as noise from the :class:`ListOfVocals`

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        """

        # -- remove vocals that were classifier as noise
        self._list_of_vocals.remove_vocals_classified_as_noise(predictions)
        # # -- update inter-vocal intervals after removing noise
        # self._list_of_vocals.update_intervals()
        return 0

    def update_vocals_with_class_classification(self, predictions, classes):
        """
        Updates :class:`ListOfVocals` with the class and probability distribution obtained
        using the Neural Network

        Parameters
        ----------
        predictions : List[float]
            Neural Network classification predictions for the :class:`ListOfVocals`
        classes : List[str]
            Labels used for classifying vocalizations
        """

        # -- make sure number of predictions is the same as number of vocals
        if self._list_of_vocals.number_of_vocals != predictions.shape[0]:
            raise RecordingStateError(
                "number of vocals and classifier predictions differ. "
                f"number of vocals: {self._list_of_vocals.number_of_vocals}; "
                f"number of predictions: {predictions.shape[0]}"
            )

        # -- add probability distribution for each vocal, top1 and top2 classes
        self._list_of_vocals.add_classification_to_vocals(predictions, classes)
        return 0

    def update_vocals_with_segmentation_masks(self, predictions):
        """
        Updates :class:`ListOfVocals` with neural-network segmentation masks.

        Parameters
        ----------
        predictions : numpy.ndarray
            segmentation mask predictions for each vocal
        """
        if self._list_of_vocals.number_of_vocals != predictions.shape[0]:
            raise RecordingStateError(
                "number of vocals and segmenter predictions differ. "
                f"number of vocals: {self._list_of_vocals.number_of_vocals}; "
                f"number of predictions: {predictions.shape[0]}"
            )

        self._list_of_vocals.add_segmentation_masks_to_vocals(predictions)
        return 0

    def save_outputs(self, validation_flag):
        logger = getLogger()

        logger.info("saving recording object, vocalizations, and csv file")
        timeAsaving = time()
        if validation_flag is True:
            self.save_validation_images(path=self.output_dir)
        if self._list_of_vocals is not None and self._list_of_vocals.has_cnn_masks():
            self.save_cnn_masks(path=self.output_dir)
        # self.save_recording_object(path=self.output_dir)
        self.remove_spectrograms_and_masks_from_object()
        self.save_recording_object(filename="recording_without_spectrograms", path=self.output_dir)
        self.save_recording_data_to_csv(path=self.output_dir)
        logger.info(f"done saving ({time() - timeAsaving:.0f}s)")

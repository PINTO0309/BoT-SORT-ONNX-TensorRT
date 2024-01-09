#!/usr/bin/env python

"""
runtime: https://github.com/microsoft/onnxruntime

pip install onnxruntime or pip install onnxruntime-gpu
pip install lap==0.4.0 scipy==1.10.1 opencv-contrib-python==4.9.0.80
"""
from __future__ import annotations
import os
import re
import sys
import copy
import cv2
import time
import lap
import requests
import subprocess
import numpy as np
import scipy.linalg
from enum import Enum
from collections import OrderedDict, deque
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

# https://developer.nvidia.com/cuda-gpus
NVIDIA_GPU_MODELS_CC = [
    'RTX 3050', 'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090',
]

ONNX_TRTENGINE_SETS = {
    'yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_910520829314548387_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_910520829314548387_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_910520829314548387_1_1_fp16_sm86.profile',
    ],
    'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_0_0_fp16_sm86.profile',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_377269473329240331_1_1_fp16_sm86.profile',
    ],
}

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, w, h, vx, vy, vw, vh

    contains the bounding box center position (x, y), width w, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, w, h) is taken as direct observation of the state space (linear
    observation model).
    """

    """
    Table for the 0.95 quantile of the chi-square distribution with N degrees of
    freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
    function and used as Mahalanobis gating threshold.
    """
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean: np.ndarray = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 2],
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 2],
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, w, h), where (x, y)
            is the center position, w the width, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain: np.ndarray = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean: np.ndarray = mean + np.dot(innovation, kalman_gain.T)
        new_covariance: np.ndarray = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4

class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_long_lost(self):
        self.state = TrackState.LongLost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh: np.ndarray, score: float, classid: int, feature_history: int, feature: np.ndarray=None):
        """STrack

        Parameters
        ----------
        tlwh: np.ndarray
            Top-left, width, height. [x1, y1, w, h]

        score: float
            Object detection score.

        classid: int
            Class ID.

        feature_history: int
            Number of features to be retained in history.

        feature: Optional[np.ndarray]
            Features obtained from the feature extractor.
        """
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.classid = classid

        self.smooth_feature = None
        self.curr_feature = None
        if feature is not None:
            self.update_features(feature)
        self.features = deque([], maxlen=feature_history)
        self.feature_history = feature_history
        self.alpha = 0.9

    def update_features(self, feature: np.ndarray):
        # Skip processing because it has already been
        # normalized in the post-processing process of ONNX.
        # feature /= np.linalg.norm(feature)
        self.curr_feature = feature
        if self.smooth_feature is None:
            self.smooth_feature = feature
        else:
            self.smooth_feature = self.alpha * self.smooth_feature + (1 - self.alpha) * feature
        self.features.append(feature)
        self.smooth_feature /= np.linalg.norm(self.smooth_feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: List[STrack]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: List[STrack], H: np.ndarray=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track: STrack, frame_id: int):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feature is not None:
            self.update_features(new_track.curr_feature)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh: np.ndarray):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh: np.ndarray):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    _mean: np.ndarray = np.array([0.000, 0.000, 0.000], dtype=np.float32)
    _std: np.ndarray = np.array([1.000, 1.000, 1.000], dtype=np.float32)

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap: Tuple = (2, 0, 1)
    _h_index: int = 2
    _w_index: int = 3
    _norm_shape: List = [1,3,1,1]
    _class_score_th: float

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        mean: Optional[np.ndarray] = np.array([0.000, 0.000, 0.000], dtype=np.float32),
        std: Optional[np.ndarray] = np.array([1.000, 1.000, 1.000], dtype=np.float32),
        class_score_th: float = 0.35,
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            self._input_shapes = [
                input.shape for input in self._interpreter.get_inputs()
            ]
            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3
            self._norm_shape = [1,3,1,1]

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_shapes = [
                input.get('shape', None) for input in self._input_details
            ]
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2
            self._norm_shape = [1,1,1,3]

        self._mean = mean.reshape(self._norm_shape)
        self._std = std.reshape(self._norm_shape)
        self._class_score_th = class_score_th

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

class YOLOX(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolox_n_body_head_hand_post_0461_0.4428_1x3x256x320_float32.tflite',
        class_score_th: Optional[float] = 0.35,
        providers: Optional[List] = None,
    ):
        """YOLOX

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOX. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOX

        class_score_th: Optional[float]
            Score threshold. Default: 0.35

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            class_score_th=class_score_th,
            providers=providers,
        )

    def __call__(
        self,
        image: np.ndarray,
    ) -> List[Box]:
        """YOLOX

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        boxes: np.ndarray
            Predicted boxes: [N, x1, y1, x2, y2]

        scores: np.ndarray
            Predicted box scores: [N, score]
        """
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )

        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]

        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
            )

        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_image = cv2.resize(
            image,
            (
                int(self._input_shapes[0][self._w_index]),
                int(self._input_shapes[0][self._h_index]),
            )
        )
        resized_image = resized_image.transpose(self._swap)
        resized_image = \
            np.ascontiguousarray(
                resized_image,
                dtype=np.float32,
            )

        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """

        """
        Detector is
            N -> Number of boxes detected
            batchno -> always 0: BatchNo.0

        batchno_classid_score_x1y1x2y2: float32[N,7]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box, score in zip(boxes_keep, scores_keep):
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    result_boxes.append(
                        Box(
                            classid=int(box[1]),
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                        )
                    )

        return result_boxes

class FastReID(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
        providers: Optional[List] = None,
    ):
        """FastReID

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for FastReID. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for FastReID

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
            mean=np.array([0.485, 0.456, 0.406], dtype=np.float32),
            std=np.array([0.229, 0.224, 0.225], dtype=np.float32),
        )
        self.feature_size = self._output_shapes[1][-1]

    def __call__(
        self,
        *,
        base_images: List[np.ndarray],
        target_features: List[np.ndarray]
    ) -> np.ndarray:
        """FastReID

        Parameters
        ----------
        base_image: List[np.ndarray]
            Object images [N, 3, H, W]

        target_features: List[np.ndarray]
            features [M, 2048]

        Returns
        -------
        similarities: np.ndarray
            features [N, M]

        base_features: np.ndarray
            features [M, 2048]
        """
        temp_base_images = copy.deepcopy(base_images)
        temp_target_features = copy.deepcopy(target_features)

        # PreProcess
        temp_base_images = \
            self._preprocess(
                base_images=temp_base_images,
            )

        # Inference
        outputs = super().__call__(input_datas=[temp_base_images, temp_target_features])
        similarities = outputs[0]
        base_features = outputs[1]
        return similarities, base_features

    def _preprocess(
        self,
        *,
        base_images: List[np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        """_preprocess

        Parameters
        ----------
        base_images: List[np.ndarray]
            Entire image

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        stacked_images_N: np.ndarray
            Resized and normalized image. [N, 3, H, W]
        """
        # Resize + Transpose
        resized_base_images_np: np.ndarray = None
        resized_base_images_list: List[np.ndarray] = []
        for base_image in base_images:
            resized_base_image: np.ndarray = \
                cv2.resize(
                    src=base_image,
                    dsize=(
                        int(self._input_shapes[0][self._w_index]),
                        int(self._input_shapes[0][self._h_index]),
                    )
                )
            resized_base_image = resized_base_image[..., ::-1] # BGR to RGB
            resized_base_image = resized_base_image.transpose(self._swap)
            resized_base_images_list.append(resized_base_image)
        resized_base_images_np = np.asarray(resized_base_images_list)
        resized_base_images_np = (resized_base_images_np / 255.0 - self._mean) / self._std
        resized_base_images_np = resized_base_images_np.astype(self._input_dtypes[0])
        return resized_base_images_np

class BoTSORT(object):
    def __init__(self, object_detection_model, feature_extractor_model, track_target_classes: List[int], frame_rate: int=30):

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh: float = 0.6 # tracking confidence threshold Default: 0.6
        self.track_low_thresh: float = 0.1 # lowest detection threshold valid for tracks Default: 0.1
        self.new_track_thresh: float = 0.9 # new track thresh Default: 0.7
        self.match_thresh: float = 0.8 # matching threshold for tracking Default: 0.8
        self.track_buffer: int = 300 # the frames for keep lost tracks Default: 30
        self.feature_history: int = 300 # the frames for keep features Default: 50

        self.buffer_size: int = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost: int = self.buffer_size
        self.kalman_filter: KalmanFilter = KalmanFilter()

        # Object detection module
        self.detector: YOLOX = object_detection_model
        self.detections: List[Box] = []
        self.strack_features: List[np.ndarray] = []

        # ReID module
        self.proximity_thresh: float = 0.5 # threshold for rejecting low overlap reid matches Default: 0.5
        self.appearance_thresh: float = 0.25 # threshold for rejecting low appearance similarity reid matches Default: 0.25
        self.encoder: FastReID = feature_extractor_model

        self.track_target_classes = track_target_classes

    def update(self, image: np.ndarray):
        self.frame_id += 1
        activated_starcks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        debug_image = copy.deepcopy(image)

        # Object detection =========================================================
        detected_boxes: List[Box] = self.detector(image=debug_image)
        boxes = copy.deepcopy([box for box in detected_boxes if box.classid in self.track_target_classes])
        non_tracked_boxes = copy.deepcopy([box for box in detected_boxes if box.classid not in self.track_target_classes])
        # Object detection =========================================================

        # Add newly detected tracklets to tracked_stracks
        unconfirmed_stracks: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Extract embeddings
        # Feature extraction of detected objects and
        # calculation of similarity between the extracted features and the previous feature.
        # At the first calculation, the previous features are treated as all zeros.
        # similarities: [N, M]
        # current_features: [N, 2048]
        person_images: List[np.ndarray] = [debug_image[box.y1:box.y2, box.x1:box.x2, :] for box in boxes]

        strack_features: List[np.ndarray] = []
        current_stracks: List[STrack] = []
        similarities: np.ndarray = None
        current_features: np.ndarray = None

        strack_features = [
            strack.curr_feature for strack in strack_pool
        ] if len(strack_pool) > 0 else np.zeros([0, self.encoder.feature_size], dtype=np.float32)
        if len(person_images) > 0:
            similarities_and_current_features: Tuple[np.ndarray, np.ndarray] = \
                self.encoder(
                    base_images=person_images,
                    target_features=strack_features,
                )
            similarities = similarities_and_current_features[0]
            similarities = similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
            current_features = similarities_and_current_features[1]
        else:
            similarities = np.zeros([0, len(strack_pool)], dtype=np.float32).transpose(1, 0)
            current_features = np.zeros([0, self.encoder.feature_size], dtype=np.float32)
        current_stracks: List[STrack] = []
        current_similarities: np.ndarray = copy.deepcopy(similarities)
        low_score_current_stracks: List[STrack] = []
        if len(boxes) > 0:
            current_stracks: List[STrack] = [
                STrack(
                    tlwh=STrack.tlbr_to_tlwh(np.asarray([box.x1,box.y1,box.x2,box.y2])),
                    score=box.score,
                    classid=box.classid,
                    feature=base_feature,
                    feature_history=self.feature_history
                ) for box, base_feature in zip(boxes, current_features) if box.score > self.track_high_thresh
            ]
            if len(boxes) != len(current_stracks) and len(current_stracks) > 0 and len(current_similarities) > 0:
                current_similarities = current_similarities.transpose(1, 0) # M: stracks N: boxes, [M, N] -> [N, M]
                current_similarities = np.asarray([
                    current_similarity for box, current_similarity in zip(boxes, current_similarities) if box.score > self.track_high_thresh
                ], dtype=np.float32)
                current_similarities = current_similarities.transpose(1, 0) # N: boxes M: stracks, [N, M] -> [M, N]
            elif len(current_stracks) == 0 and len(current_similarities) > 0:
                current_similarities = np.zeros([0, len(strack_pool)], dtype=np.float32).transpose(1, 0)
            elif len(current_stracks) > 0 and len(current_similarities) == 0:
                current_similarities = np.zeros([0, len(current_stracks)], dtype=np.float32)
            low_score_current_stracks: List[STrack] = [
                STrack(
                    tlwh=STrack.tlbr_to_tlwh(np.asarray([box.x1,box.y1,box.x2,box.y2])),
                    score=box.score,
                    classid=box.classid,
                    feature=base_feature,
                    feature_history=self.feature_history
                ) for box, base_feature in zip(boxes, current_features) if box.score <= self.track_high_thresh and box.score >= self.track_low_thresh
            ]

        # Calibration by camera motion is not performed.
        # STrack.multi_gmc(strack_pool, np.eye(2, 3, dtype=np.float32))
        # STrack.multi_gmc(unconfirmed, np.eye(2, 3, dtype=np.float32))

        # First association, with high score detection boxes
        ious_dists = iou_distance(strack_pool, current_stracks)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        emb_dists = current_similarities
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track: STrack = strack_pool[itracked]
            det: STrack = current_stracks[idet]
            if track.state == TrackState.Tracked:
                track.update(current_stracks[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(new_track=det, frame_id=self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Second association, with low score detection boxes
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, low_score_current_stracks)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det: STrack = low_score_current_stracks[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(new_track=det, frame_id=self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        unconfirmed_boxes = [current_stracks[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed_stracks, unconfirmed_boxes)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        unconfirmed_strack_curr_features = \
            np.asarray([unconfirmed_strack.curr_feature for unconfirmed_strack in unconfirmed_stracks], dtype=np.float32) \
                if len(unconfirmed_stracks) > 0 else np.zeros([0, self.encoder.feature_size], dtype=np.float32)
        unconfirmed_boxes_features = \
            np.asarray([unconfirmed_box.curr_feature for unconfirmed_box in unconfirmed_boxes], dtype=np.float32) \
                if len(unconfirmed_boxes) > 0 else np.zeros([0, self.encoder.feature_size], dtype=np.float32)
        emb_dists = np.maximum(0.0, np.matmul(unconfirmed_strack_curr_features, unconfirmed_boxes_features.transpose(1, 0)))
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed_track: STrack = unconfirmed_stracks[itracked]
            unconfirmed_track.update(unconfirmed_boxes[idet], self.frame_id)
            activated_starcks.append(unconfirmed_track)
        for it in u_unconfirmed:
            track = unconfirmed_stracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Init new stracks
        for inew in u_detection:
            track = unconfirmed_boxes[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Merge
        self.tracked_stracks: List[STrack] = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks: List[STrack] = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks: List[STrack] = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks, non_tracked_boxes


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]):
    exists: Dict[int, int] = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista: List[STrack], tlistb: List[STrack]):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa: List[STrack], stracksb: List[STrack]):
    pdist: np.ndarray = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        strackp: STrack =stracksa[p]
        timep = strackp.frame_id - strackp.start_frame
        strackq: STrack =stracksb[q]
        timeq = strackq.frame_id - strackq.start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def bbox_iou(atlbr: np.ndarray, btlbr: np.ndarray) -> float:
    # atlbr: [x1, y1, x2, y2]
    # btlbr: [x1, y1, x2, y2]

    # Calculate areas of overlap
    inter_xmin = max(atlbr[0], btlbr[0])
    inter_ymin = max(atlbr[1], btlbr[1])
    inter_xmax = min(atlbr[2], btlbr[2])
    inter_ymax = min(atlbr[3], btlbr[3])
    # If there is no overlap
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    # Calculate area of overlap and area of each bounding box
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
    area2 = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
    # Calculate IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    return iou

def bbox_ious(atlbrs: List[np.ndarray], btlbrs: List[np.ndarray]) -> np.ndarray:
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    ious = np.array([[bbox_iou(atlbr=atlbr, btlbr=btlbr) for btlbr in btlbrs] for atlbr in atlbrs])
    return ious

def iou_distance(atracks: List[STrack], btracks: List[STrack]):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = bbox_ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def gate_cost_matrix(kf: KalmanFilter, cost_matrix: np.ndarray, tracks: List[STrack], detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = KalmanFilter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix

def fuse_motion(kf: KalmanFilter, cost_matrix: np.ndarray, tracks: List[STrack], detections: List[STrack], only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = KalmanFilter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def fuse_iou(cost_matrix: np.ndarray, tracks: List[STrack], detections: List[STrack]):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def fuse_score(cost_matrix: np.ndarray, detections: List[STrack]):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def gate(cost_matrix: np.ndarray, emb_cost):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix
    index = emb_cost > 0.3
    cost_matrix[index] = 1
    return cost_matrix

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.
    If the folder does not exist, it is created.

    :param url: URL of the file to download.
    :param folder: Folder where the file will be saved.
    :param filename: Filename to save the file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Full path for the file
    file_path = os.path.join(folder, filename)
    # Download the file
    print(f"{Color.GREEN('Downloading...')} {url} to {file_path}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{Color.GREEN('Download completed:')} {file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def get_nvidia_gpu_model() -> List[str]:
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)

        # Extract GPU model numbers using regular expressions
        models = re.findall(r'GPU \d+: (.*?)(?= \(UUID)', output)
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-odm',
        '--object_detection_model',
        type=str,
        default='yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx',
        choices=[
            'yolox_n_body_head_hand_post_0461_0.4428_1x3x384x640.onnx',
            'yolox_t_body_head_hand_post_0299_0.4522_1x3x384x640.onnx',
            'yolox_s_body_head_hand_post_0299_0.4983_1x3x384x640.onnx',
            'yolox_m_body_head_hand_post_0299_0.5263_1x3x384x640.onnx',
            'yolox_l_body_head_hand_post_0299_0.5420_1x3x384x640.onnx',
            'yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx',
        ],
        help='ONNX/TFLite file path for YOLOX.',
    )
    parser.add_argument(
        '-fem',
        '--feature_extractor_model',
        type=str,
        default='mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
        choices=[
            'mot17_sbs_S50_NMx3x256x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x288x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x320x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x352x128_post_feature_only.onnx',
            'mot17_sbs_S50_NMx3x384x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x256x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x288x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x320x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x352x128_post_feature_only.onnx',
            'mot20_sbs_S50_NMx3x384x128_post_feature_only.onnx',
        ],
        help='ONNX/TFLite file path for FastReID.',
    )
    parser.add_argument(
        '-tc',
        '--track_target_classes',
        type=int,
        nargs='+',
        default=[0],
        help='List of class IDs to be tracked. 0:Body, 1: Head, 2: Hand',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
        help='Video file path or camera index.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='tensorrt',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    args = parser.parse_args()

    # runtime check
    object_detection_model_file: str = args.object_detection_model
    feature_extractor_model_file: str = args.feature_extractor_model
    object_detection_model_ext: str = os.path.splitext(object_detection_model_file)[1][1:].lower()
    feature_extractor_model_ext: str = os.path.splitext(feature_extractor_model_file)[1][1:].lower()
    runtime: str = None
    if object_detection_model_ext != feature_extractor_model_ext:
        print(Color.RED('ERROR: object_detection_model and feature_extractor_model must be files with the same extension.'))
        sys.exit(0)
    if object_detection_model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif object_detection_model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)

    WEIGHT_FOLDER_PATH = '.'
    gpu_models = get_nvidia_gpu_model()
    default_supported_gpu_model = False
    if len(gpu_models) == 1:
        gpu_model = gpu_models[0]
        for target_gpu_model in NVIDIA_GPU_MODELS_CC:
            if target_gpu_model in gpu_model:
                default_supported_gpu_model = True
                break

    # Download object detection onnx
    weight_file = os.path.basename(object_detection_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download object detection tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    # Download reid onnx
    weight_file = os.path.basename(feature_extractor_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download reid tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    track_target_classes: List[int] = args.track_target_classes
    video: str = args.video
    execution_provider: str = args.execution_provider
    providers: List[Tuple[str, Dict] | str] = None
    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    # Model initialization
    object_detection_model = \
        YOLOX(
            runtime=runtime,
            model_path=object_detection_model_file,
            providers=providers,
        )
    feature_extractor_model = \
        FastReID(
            runtime=runtime,
            model_path=feature_extractor_model_file,
            providers=providers,
        )
    botsort = \
        BoTSORT(
            object_detection_model=object_detection_model,
            feature_extractor_model=feature_extractor_model,
            track_target_classes=track_target_classes,
            frame_rate=30,
        )

    cap = cv2.VideoCapture(
        int(video) if is_parsable_to_int(video) else video
    )
    disable_video_writer: bool = args.disable_video_writer
    video_writer = None
    if not disable_video_writer:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=fourcc,
            fps=cap_fps,
            frameSize=(w, h),
        )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)
        # debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()
        stracks, boxes = botsort.update(image=debug_image)
        elapsed_time = time.perf_counter() - start_time
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        for strack in stracks:
            color = (255,255,255)
            if strack.classid == 0:
                color = (255,0,0)
            elif strack.classid == 1:
                color = (0,0,255)
            elif strack.classid == 2:
                color = (0,255,0)
            cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), (255,255,255), 2)
            cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), color, 1)
            ptx = int(strack.tlbr[0]) if int(strack.tlbr[0])+50 < debug_image_w else debug_image_w-50
            pty = int(strack.tlbr[1])-10 if int(strack.tlbr[1])-25 > 0 else 20
            cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        for box in boxes:
            if box.classid not in track_target_classes:
                color = (255,255,255)
                if box.classid == 0:
                    color = (255,0,0)
                elif box.classid == 1:
                    color = (168,87,167)
                elif box.classid == 2:
                    color = (0,255,0)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), (255,255,255), 2)
                cv2.rectangle(debug_image, (box.x1, box.y1), (box.x2, box.y2), color, 1)
                ptx = box.x1 if box.x1+50 < debug_image_w else debug_image_w-50
                pty = box.y1-10 if box.y1-25 > 0 else 20

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", debug_image)
        if video_writer is not None:
            video_writer.write(debug_image)

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()

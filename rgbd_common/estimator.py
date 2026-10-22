"""Filter-neutral protocol and saved temporal records."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class Step:
    predicted:object
    predicted_covariance:np.ndarray
    filtered:object
    filtered_covariance:np.ndarray
    transition:object=None
    cross_covariance:object=None
    diagnostics:object=None


class MotionEstimator(Protocol):
    name:str
    def predict(self,state,covariance,dt):...
    def correct(self,state,covariance,problem,reference=False,nominal=None):...
    def smooth(self,steps):...
    def refine_reverse_hypothesis(self,replay):...

from gymnasium import Env
import numpy as np
from collections import deque
import random
import math
from gymnasium import error, spaces, utils

from .fogg_behavioral_model import Patient

PROFILES = {
    'closed': dict(social_influence=-1.0,
                   likes_challenge=-1.0,
                   competetiveness=-1.0,
                   stress_resistance=1.0,
                   fatigue_resistance=1.0),  # profile 0
    'unmotivated': dict(social_influence=-1.0,
                        likes_challenge=-1.0,
                        competetiveness=-1.0,
                        stress_resistance=-1.0,
                        fatigue_resistance=-1.0),  # profile 1
    'social': dict(social_influence=1.0,
                   likes_challenge=1.0,
                   competetiveness=1.0,
                   stress_resistance=0.0,
                   fatigue_resistance=0.0),  # profile 2
    'mixed': dict(social_influence=0.0,
                  likes_challenge=1.0,
                  competetiveness=0.0,
                  stress_resistance=0.0,
                  fatigue_resistance=-1.0)  # profile 3
}


class ProfiledPatient(Patient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            profile_name = kwargs.get('profile')
        except KeyError:
            raise ValueError("Profile must be provided as a keyword argument 'profile'")

        try:
            self.profile = PROFILES[profile_name]
        except KeyError:
            raise ValueError(f"Profile must be one of {list(PROFILES.keys())}")

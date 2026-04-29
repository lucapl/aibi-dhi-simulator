import numpy as np
from collections import deque
import random
import math
from typing import override

from .fogg_behavioral_model import Patient

PROFILES = {
    'neutral': dict(social_influence=0.0,
                    likes_challenge=0.0,
                    competetiveness=0.0,
                    stress_resistance=1.0,
                    fatigue_resistance=1.0),  # profile 0
    'unmotivated': dict(social_influence=0.0,
                        likes_challenge=0.0,
                        competetiveness=0.0,
                        stress_resistance=0.0,
                        fatigue_resistance=0.0),  # profile 1
    'social': dict(social_influence=1.0,
                   likes_challenge=1.0,
                   competetiveness=1.0,
                   stress_resistance=0.0,
                   fatigue_resistance=0.0),  # profile 2
    'mixed': dict(social_influence=0.5,
                  likes_challenge=1.0,
                  competetiveness=0.5,
                  stress_resistance=0.5,
                  fatigue_resistance=0.0)  # profile 3
}


class ProfiledPatient(Patient):
    def __init__(self, *args, profile=None, practice_in_group=True, **kwargs):
        if profile is None:
            raise ValueError("Profile must be provided as a keyword argument 'profile'")
        if profile not in PROFILES:
            raise ValueError(f"Profile must be one of {list(PROFILES.keys())}")

        self.profile = PROFILES[profile]
        self.in_group = practice_in_group
        self.activity_scores = [0]
        # if rescale_weights:
        #     self.__rescale_weights()

        super().__init__(*args, **kwargs)

    # def __rescale_weights(self):
    #     """ Rescales the profile weights to be between 0 and 1"""
    #     self.profile = {key: (value + 1) / 2 for key, value in self.profile.items()}


    def _get_activity_score(self):
        return np.random.normal(loc=sum(self.activity_performed)+self.last_activity_score, scale=1.5)

    def _get_group_score(self):
        return np.random.normal(loc=sum(self.activity_performed), scale=1.0)

    @override
    def reset(self, seed=None):
        state, info = super().reset(seed=seed)
        return state, info

    @override
    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        behaviour = reward > 0
        if behaviour:
            self.activity_scores.append(self._get_activity_score())
        else:
            self.activity_scores.append(0)

        return state, reward, terminated, truncated, info

    def _social_influence_weight(self):
        # If I see that I perform health-related activities better
        # than my peers, I feel motivated

        # motivation based on group performance compared to individual
        if not self.in_group:
            return 0
        better_score = self.activity_scores[-1] > self._get_group_score()
        return self.profile['social_influence'] * (1 if better_score else 0)

    def _likes_challenge_weight(self):
        # I am keen to perform a simple activity for a longer time
        # when I want to break my own record

        if len(self.activity_scores) < 2:
            return 0
        is_tired = self.is_tired_of_repeating_the_activity()
        if is_tired == -1:
            return 0
        better_score = self.activity_scores[-1] > self.activity_scores[-2]
        return self.profile['likes_challenge'] * better_score

    def _competetiveness_weight(self):
        # I am keen to perform a simple activity for a longer time
        # when I compete with others

        if not self.in_group:
            return 0
        is_tired = self.is_tired_of_repeating_the_activity()
        if is_tired == -1:
            return 0
        return self.profile['competetiveness']

    def get_motivation_weight(self):
        return max(self._social_influence_weight() +
                   self._likes_challenge_weight() +
                   self._competetiveness_weight(), 0)

    @override
    def get_motivation(self):
        return self.get_motivation_weight() + super().get_motivation()

    def is_stressed(self):
        # When I am stressed, I am unlikely to respond to any
        # reminders for health-related activities 

        # stress is only present when arousal is high and valence is low
        return (1-self.profile['stress_resistance']) * (self.arousal == 2 and self.valence == 0)
    
    def is_fatigued(self):
        # When I am tired, I am unlikely to respond to any 
        # reminders for health-related activities 
        number_of_hours_slept = self.awake_list[-24:].count('sleeping')
        sleep_deprivation = number_of_hours_slept < 7
        is_tired_of_repeating_the_activity = self.is_tired_of_repeating_the_activity() == 1
        return (1-self.profile['fatigue_resistance']) * (is_tired_of_repeating_the_activity or sleep_deprivation)

    def get_trigger_weight(self):
        return -self.is_stressed() - self.is_fatigued()

    @override
    def get_trigger(self):
        if self.should_prompt() == 0:
            return 0
    
        return max(self.get_trigger_weight() + super().get_trigger(), 0)


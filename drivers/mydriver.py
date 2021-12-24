import numpy as np
import scipy.interpolate
from collections import defaultdict
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, Union, Any, List

from resources.coordinatesystem import *
from resources.actions import *
from resources.states import *
from resources.rng import driver_rng
from drivers.driver import Driver

# TODO: Find out why Driver doesn't stop at the edges of the maze, seems like a condition wasn't caught

class MyDriver(Driver):
    def __init__(self, name, print_info=False, random_action_probability=0.9, random_action_decay=0.99,
                 min_random_action_probability=0.0, *args, **kwargs):
        self._name = name
        self.print_info = print_info
        self.random_action_probability = random_action_probability
        self.random_action_decay = random_action_decay
        self.min_random_action_probability = min_random_action_probability
        
        # avoiding safety car penalties are important (find speed not incurring penalties)
        self.safety_car_running = False
        self.safety_car_speed = 150         # initialise it to be at a medium speed
        self.n_safety_cars = 0
        self.n_safety_car_penalties = 0
        self.min_unsafe_safety_car_speed = self.safety_car_speed
        
        self.correct_turns = {}
        
        self.sl_data = {action: [] for action in Action.get_sl_actions()}  # straight line actions
        self.drs_data = {action: [] for action in [Action.LightThrottle, Action.FullThrottle, Action.Continue]}  # drs actions
        self.corner_speed = 350 / 2  # initialise it to be half top speed (renamed from end_of_straight_speed)
        self.max_safe_corner_speed = np.nan
        self.min_unsafe_corner_speed = np.nan
        self.uturn_speed = 350 / 2  #  initialise it to be half top speed
        self.max_safe_uturn_speed = np.nan
        self.min_unsafe_uturn_speed = np.nan
        self.target_speeds = 350 * np.ones(50)
        self.target_speeds[0] = self.corner_speed
        self.drs_was_active = False
        self.in_maze_branch = False
        
        self.sl_data = {**self.sl_data, **{
            Action.LightBrake: [[0,0]], # pretty sure braking at 0 will be 0 speed always
            Action.HeavyBrake: [[0,0]],
            Action.Continue: [[0,0], [500,500]] # intially assuming continue to be linear
        }}
        
        
    def prepare_for_race(self):
        pass

    
    def choose_tyres(self, track_info):         # for level 4 - Pro Driver
        return None

    
    def choose_aero(self, track_info):          # for level 4 - Pro Driver
        return None

    
    def make_a_move(self, car_state: CarState, track_state: TrackState) -> Action:
        if track_state.distance_ahead == 0 and not (track_state.distance_left == 0 and track_state.distance_right == 0
                                                    and car_state.speed > 0):
            return self._choose_turn_direction(track_state)
        
        # Get the target speed
        target_speed = self._get_target_speed(track_state.distance_ahead, track_state.safety_car_active)
        
        # Choose action that gets us closest to target, or choose randomly
        prevent_random_action = track_state.distance_ahead < 2
        if driver_rng().rand() > self.random_action_probability or prevent_random_action:
            action = self._choose_move_from_models(car_state.speed, target_speed, car_state.drs_active)
        else:
            action = self._choose_randomly(Action.get_sl_actions())
            
        # If DRS is available then need to decide whether to open DRS or not.
        if track_state.drs_available and not car_state.drs_active:
            # Simulate the straight with and without DRS and check which we think will be faster
            time_no_drs, targets_broken_no_drs, _ = self.simulate_straight(car_state.speed,
                                                                        track_state.distance_ahead,
                                                                        drs_active=False,
                                                                        safety_car_active=track_state.safety_car_active)
            time_drs, targets_broken_drs, _ = self.simulate_straight(car_state.speed, track_state.distance_ahead - 1,
                                                                     drs_active=True,
                                                                     safety_car_active=track_state.safety_car_active)
            time_drs = (1 / (car_state.speed + 1)) + time_drs

            if (time_drs < time_no_drs or driver_rng().rand() < self.random_action_probability
                or any(len(data) < 10 for data in self.drs_data.values())) and not targets_broken_drs:
                action = Action.OpenDRS
                self.drs_was_active = True
                if self.print_info:
                    print('Opening DRS')
            elif self.print_info:
                print('Chose not to open DRS')
                
        # Decay random_action_probability
        self.random_action_probability = max(self.random_action_probability * self.random_action_decay,
                                             self.min_random_action_probability)

        return action

        
    def _choose_turn_direction(self, track_state: TrackState):
        # Check if we need to make a decision about which way to turn
        self.in_maze_branch = False
        if track_state.distance_left > 0 and track_state.distance_right > 0:  # both options available, need to decide
            self.in_maze_branch = True
            if len(self.correct_turns) > 0:
                # TODO: probably want to put some limit on distance.  It is also doing a 1 nearest neighbor (can optimize).
                
                # Find the closest turn we have seen previously and turn in the same direction
                distances = np.array([track_state.position.distance_to(turn_position)
                                      for turn_position in self.correct_turns])
                i_closest = np.argmin(distances)
                return list(self.correct_turns.values())[i_closest]

            else:  # First race, no data yet so choose randomly
                return driver_rng().choice([Action.TurnLeft, Action.TurnRight])

        elif track_state.distance_left > 0:  # only left turn
            return Action.TurnLeft
        else:
            return Action.TurnRight  # only right or dead-end
        
        
    def _get_target_speed(self, distance_ahead, safety_car_active, target_speeds=None):
        if target_speeds is None:
            target_speeds = self.target_speeds

        if distance_ahead == 0:
            target_speed = 0  # dead end - need to stop!!
        else:
            target_speed = target_speeds[distance_ahead - 1]  # target for next step

        if safety_car_active:
            target_speed = min(target_speed, self.safety_car_speed)

        return target_speed
    
    
    def _choose_move_from_models(self, speed: float, target_speed: float, drs_active: bool, **kwargs):
        # Test each action to see which will get us closest to our target speed
        actions = Action.get_sl_actions()
        if 0 == speed:  # yes this is technically cheating but you can get stuck here with low grip so bending the rules
            actions = [Action.LightThrottle, Action.FullThrottle]
        next_speeds = np.array([self.estimate_next_speed(action, speed, drs_active, **kwargs) for action in actions]).astype(float)
        errors = next_speeds - target_speed    # difference between predicted next speed and target, +ve => above target

        # The target speed is the maximum safe speed so we want to be under the target if possible. This means we don't
        # necessarily want the action with the smallest error
        if np.any(errors <= 0):            # under or equal to the target speed
            errors[errors > 0] = np.inf    # at least one action gets us under the speed so ignore others even if close

        # Now we can choose the action with the smallest error score. At the start there will be multiple actions with
        # with the same score, so we will choose randomly from these
        # TODO: This optimization is maybe something to check out
        min_error = np.min(errors ** 2)
        available_actions = [action for action, error in zip(actions, errors)
                             if np.abs(error ** 2 - min_error) < 1e-3]
        
        return self._choose_randomly(available_actions)

    
    def estimate_next_speed(self, action: Action, speed, drs_active: bool, **kwargs) -> float:
        data = np.array(self.get_data(action, drs_active))
        if data.shape[0] < 2:
            return speed
        interp = scipy.interpolate.interp1d(data[:, 0], data[:, 1], fill_value='extrapolate', assume_sorted=False)
        return interp(speed)
    
    
    def estimate_previous_speed(self, test_input_speeds: np.ndarray, test_output_speeds: np.ndarray, speed) -> float:
        # TODO: double check new target speed estimate by squared min error
        errors = (test_output_speeds - speed)**2
        speeds_min_error = test_input_speeds[errors == np.min(errors)]
        return np.max(speeds_min_error)
    
    
    def simulate_straight(self, speed, distance_ahead, drs_active, safety_car_active):
        speeds = np.zeros(distance_ahead)
        break_target_speed = False
        for d in range(distance_ahead):
            target_speed = self._get_target_speed(distance_ahead - d, safety_car_active)
            action = self._choose_move_from_models(speed, target_speed, drs_active)
            speeds[d] = self.estimate_next_speed(action, speed, drs_active)
            speed = speeds[d]
            break_target_speed |= speed > target_speed
        time = np.sum(1 / (speeds + 1))
        return time, break_target_speed, speeds
    
    
    def get_data(self, action, drs_active=False):
        if drs_active and action in self.drs_data:
            return self.drs_data[action]
        else:
            return self.sl_data[action]
    
    
    # signature copied from RookieDriver
    def update_with_action_results(self, previous_car_state: CarState, previous_track_state: TrackState,
                                   action: Action, new_car_state: CarState, new_track_state: TrackState,
                                   result: ActionResult) -> None:
        
        if previous_track_state.safety_car_active:
            self.safety_car_running = True
            self._update_safety_car(new_car_state, result)
        else:
            # reset safety car
            if self.safety_car_running:
                self.n_safety_cars += 1
                self.safety_car_running = False
            self.safety_car_speed = min(150, self.min_unsafe_safety_car_speed - 10)
            

        if previous_track_state.distance_ahead == 0:

            if result.crashed or result.spun:
                if self.print_info:
                    print(f'\tCrashed! We targeted {self.corner_speed: .0f} speed and were going '
                          f'{previous_car_state.speed: .0f}')
                if previous_track_state.distance_left > 0 ^ previous_track_state.distance_right > 0:
                    self.min_unsafe_corner_speed = np.nanmin([self.min_unsafe_corner_speed, previous_car_state.speed])
                if previous_track_state.distance_left == 0 and previous_track_state.distance_right == 0:
                    self.min_unsafe_uturn_speed = np.nanmin([self.min_unsafe_uturn_speed, previous_car_state.speed])
            else:
                if previous_track_state.distance_left == 0 and previous_track_state.distance_right == 0:
                    self.max_safe_uturn_speed = np.nanmax([self.max_safe_uturn_speed, previous_car_state.speed])
                else:
                    self.max_safe_corner_speed = np.nanmax([self.max_safe_corner_speed, previous_car_state.speed])
            
            self.corner_speed = self.update_speed_adjust([self.max_safe_corner_speed, self.max_safe_uturn_speed],
                                                                  [self.min_unsafe_corner_speed],
                                                                  self.corner_speed)
            
            prev_uturn = self.uturn_speed
            self.uturn_speed = self.update_speed_adjust([self.max_safe_uturn_speed], 
                                                             [self.min_unsafe_uturn_speed, self.min_unsafe_corner_speed], 
                                                             self.uturn_speed)
            
            # Update our target speeds
            self.update_target_speeds()
            self.drs_was_active = False

        elif action in self.sl_data:       # record the change in speed resulting from the action we took
            # Record the point if it is not on top of another point (interpolation doesn't like points too close
            # together in x, plus it is also a bit unnecessary) and we are below 200 points (just to keep code
            # performance up)
            current_data = self.get_data(action, previous_car_state.drs_active)
            if 0 == len(current_data):
                closest_distance = 1000
            else:
                closest_distance = np.min(np.abs(np.array(current_data)[:, 0] - previous_car_state.speed))
            if closest_distance > 2 and len(current_data) < 200:
                new_data = [previous_car_state.speed, new_car_state.speed]
                current_data.append(new_data)

    
    @staticmethod
    def update_speed_adjust(safe_speeds: List[float], unsafe_speeds: List[float], current_speed: float, accept_threshold: float = 10.) -> float:        
        if (~np.isnan(safe_speeds)).sum() == 0 or (~np.isnan(unsafe_speeds)).sum() == 0:
            return current_speed
        max_safe, min_unsafe = np.nanmax(safe_speeds), np.nanmin(unsafe_speeds)
        if min_unsafe - max_safe < accept_threshold:
            return max_safe
        return np.mean([max_safe, min_unsafe])
                

    def _update_safety_car(self, previous_car_state: CarState, result: ActionResult) -> None:
        prev_safety_car_speed = self.safety_car_speed
        if result.safety_car_speed_exceeded:  # we ended up going too fast so safe speed must be below current speed
            self.n_safety_car_penalties += 1

            # try to go just a bit slower than min_unsafe_safety_car_speed
            self.min_unsafe_safety_car_speed = min(self.min_unsafe_safety_car_speed, previous_car_state.speed)
            self.safety_car_speed = min(self.min_unsafe_safety_car_speed - 10, self.safety_car_speed)
        elif self.safety_car_running:
            self.safety_car_speed = max(self.safety_car_speed, previous_car_state.speed)

        if prev_safety_car_speed > self.safety_car_speed:
            if self.print_info:
                print(f'\tDecreasing estimate of safety car speed from {self.safety_car_speed: .1f} to {self.safety_car_speed}')
        elif prev_safety_car_speed < self.safety_car_speed:
            if self.print_info:
                print(f'\tIncreasing estimate of safety car speed from {self.safety_car_speed: .1f} to {self.safety_car_speed}')
        
        
    def update_target_speeds(self):
        previous_targets = np.copy(self.target_speeds)
        speed = self.corner_speed if self.in_maze_branch == False else self.uturn_speed

        test_input_speeds = np.linspace(0, 350, 351)
        test_output_speeds = {action: self.estimate_next_speed(action, test_input_speeds, False)
                              for action in self.sl_data}

        for i in range(len(self.target_speeds)):
            self.target_speeds[i] = speed
            speed = np.nanmax([self.estimate_previous_speed(test_input_speeds, test_output_speeds[action], speed)
                               for action in self.sl_data])
        if self.print_info and not np.array_equal(previous_targets, self.target_speeds):
            print(f'New target speeds: mid-straight->{np.array2string(self.target_speeds[5::-1], precision=0)}<-end')
       
    
    def update_after_race(self, correct_turns: Dict[Position, Action]):
        # Called after the race by RaceControl
        self.correct_turns.update(correct_turns)            # dictionary mapping Position -> TurnLeft or TurnRight
        self.n_safety_car_penalties = 0
        
        
    def get_safety_car_speed_estimate(self):
        return self.safety_car_speed
    
    
    def _choose_randomly(self, available_actions):
        return driver_rng().choice(available_actions)  # randomly choose an action uniformly over all available actions

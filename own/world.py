from __future__ import absolute_import, division, print_function
from copy import deepcopy
from re import findall, escape, compile
from os.path import join
from os import walk
from imutils import opencv2matplotlib as flip, translate
from collections import OrderedDict
from itertools import groupby
import numpy as np
import cv2
import imutils


def multiple_replace(text='', replacement=None):
    if text == '':
        raise IOError
    if replacement is None:
        replacement = {
            ".": "",
            "[": "",
            "]": "",
            " ": "",
            ",": "x",
            "'": "_",
            "{": "",
            "}": "",
            ":": "",
        }
    # Create a regular expression  from the dictionary keys
    regex = compile("(%s)" % "|".join(map(escape, replacement.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacement[mo.string[mo.start():mo.end()]], text)


class Environment(object):
    def __init__(self, master_path):
        self.master_path = master_path
        self.ordered_scenes = self.get_scenes_ordered(master_path)

        self.season_idx = 0
        self.episode_idx = 0
        self.scn_idx = 0
        self._num_of_seasons = len(self.ordered_scenes)
        self.current_season, self.current_season_episodes, = self.ordered_scenes.items()[self.season_idx]
        self.current_episode, self.current_episodes_scns = self.current_season_episodes.items()[self.episode_idx]
        self.current_scn, self.current_path = self.current_episodes_scns[self.scn_idx]
        states, num_of_frames = self.get_scene_from_path(self.current_path)
        self.current_states = deepcopy(states[:-1])
        self.next_states = deepcopy(states[1:])

    def reset(self):
        self.season_idx = 0
        self.episode_idx = 0
        self.scn_idx = 0
        self._num_of_seasons = len(self.ordered_scenes)
        self.current_season, self.current_season_episodes, = self.ordered_scenes.items()[self.season_idx]
        self.current_episode, self.current_episodes_scns = self.current_season_episodes.items()[self.episode_idx]
        self.current_scn, self.current_path = self.current_episodes_scns[self.scn_idx]
        states, num_of_frames = self.get_scene_from_path(self.current_path)
        self.current_states = deepcopy(states[:-1])
        self.next_states = deepcopy(states[1:])
        return self.current_states

    def __iter__(self):
        self.idx = 0
        return self

    def next(self):
        self.idx += 1
        idx = self.idx
        if idx > self._total_num_of_scenes:
            raise StopIteration
        new_season = False
        new_episode = False
        validation = False
        test = False
        season, episode, scn_num, path = self.ordered_scenes[idx]
        if episode != self.current_episode:
            new_episode = True
            self.current_episode = episode
        if season != self.current_season:
            new_season = True
            self.current_season = season

        if (season, episode) in self.validation_episodes:
            validation = True
        if (season, episode) in self.test_episodes:
            test = True

        scene = self.get_scene_from_path(path)

        return scene, season, episode, scn_num, new_season, new_episode, validation, test

    def get_next_scene(self):
        self.idx += 1
        idx = self.idx
        if idx > self._total_num_of_scenes:
            raise
        new_season = False
        new_episode = False
        validation = False
        test = False
        season, episode, scn_num, path = self.ordered_scenes[idx]
        if episode != self.current_episode:
            new_episode = True
            self.current_episode = episode
        if season != self.current_season:
            new_season = True
            self.current_season = season

        if (season, episode) in self.validation_episodes:
            validation = True
        if (season, episode) in self.test_episodes:
            test = True

        scene = self.get_scene_from_path(path)

        return scene, season, episode, scn_num, new_season, new_episode, validation, test

    def next_season(self):
        pass

    @staticmethod
    def get_scenes_ordered(master_path):
        # receives path to master scenes dir and returns a list of ordered tuples (season, eps, scn, path_to_scene) sorted by season then episode then scene number
        a = [[path[-5:-3], path[-2:], filename[-6:-4], join(path, filename)] for path, _, files in walk(master_path) for filename in files]
        b = [int(''.join(map(str, findall('\d+', path_to_scene)))) for season, eps, scn, path_to_scene in a]

        ordered_scenes = [x for (y, x) in sorted(zip(b, a))]

        seasons = groupby(ordered_scenes, key=lambda x: x[0])
        ordered_all = OrderedDict()
        for k, v in seasons:
            v = list(v)
            for sublist in v:
                del sublist[0]

            episodes = groupby(v, key=lambda x: x[0])
            ordered_s = OrderedDict()
            if k not in ordered_all:
                for j, t in episodes:
                    t = list(t)
                    for sublist in t:
                        del sublist[0]
                    ordered_s[j] = t
                ordered_all[k] = ordered_s
            else:
                for j, t in episodes:
                    t = list(t)
                    for sublist in t:
                        del sublist[0]
                    t = [[l[1][-7:-4], l[1]] for l in t]
                    ordered_all[k][j] += t
        return ordered_all

    @staticmethod
    def get_scene_from_path(path):
        # max length not needed because all scenes are max 32 long
        # last scene not needed because all scenes are known

        # noinspection PyArgumentList
        cap = cv2.VideoCapture(path)  # load scene to cap
        num_of_frames = int(cap.get(7))
        ret, frame = cap.read()
        scene = [flip(frame) / 255]
        ret, next_frame = cap.read()

        while ret:
            # cv2.imshow('frame', next_frame)
            # cv2.waitKey(2)
            scene.append(flip(next_frame) / 255)
            ret, next_frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()

        return np.asarray(scene, np.float32), num_of_frames


class World(object):

    """
    Wrapper for OpenAI Gym Atari environments that returns the last 4 processed frames.
    Input frames are converted to grayscale and downsampled from 120x160 to 84x112 and cropped to 84x84 around the game's main area.
    The final size of the states is 84x84x4.

    If debug=True, an OpenCV window is opened and the last processed frame is displayed. This is particularly useful when adapting the wrapper to novel environments.
    """

    def __init__(self, env_name, master_path, validation_episodes, test_episodes, debug=False):


        self.validation = False
        self.test = False

        self.validation_episodes = validation_episodes
        self.test_episodes = test_episodes

        self.env_name = env_name
        self.debug = debug
        # self.env = gym.make(env_name)

        # self.action_space = self.env.action_space
        self.action_space_size = 5
        # self.observation_space = Box(low=0, high=255, shape=(84, 84, 4))  #self.env.observation_space
        self.observation_space_shape = self.current_states.shape

        # self.frame_num = 0

        # self.monitor = self.env.monitor

        # self.frames = np.zeros((None, 320, 180, 3), dtype=np.float32)

        if self.debug:
            cv2.startWindowThread()
            cv2.namedWindow("Game")

    def step(self, actions_hat, actions_step_size, fill_value=0):
        next_states = self.next_states
        # image_t1_stay = deepcopy(self.next_states)

        image_t1_up = np.empty_like(next_states)
        image_t1_down = np.empty_like(next_states)
        image_t1_left = np.empty_like(next_states)
        image_t1_right = np.empty_like(next_states)

        if len(next_states.shape) == 4:

            image_t1_down[:, actions_step_size:, :, :] = next_states[:, :-actions_step_size, :, :]
            image_t1_down[:, :actions_step_size, :, :] = fill_value

            image_t1_right[:, :, actions_step_size:, :] = next_states[:, :, :-actions_step_size, :]
            image_t1_right[:, :, :actions_step_size, :] = fill_value

            image_t1_left[:, :, :-actions_step_size, :] = next_states[:, :, actions_step_size:, :]
            image_t1_left[:, :, -actions_step_size:, :] = fill_value

            image_t1_up[:, :-actions_step_size, :, :] = next_states[:, actions_step_size:, :, :]
            image_t1_up[:, -actions_step_size:, :, :] = fill_value

        else:

            image_t1_down[actions_step_size:, :, :] = next_states[:-actions_step_size, :, :]
            image_t1_down[:actions_step_size, :, :] = fill_value

            image_t1_right[:, actions_step_size:, :] = next_states[:, :-actions_step_size, :]
            image_t1_right[:, :actions_step_size, :] = fill_value

            image_t1_left[:, :-actions_step_size, :] = next_states[:, actions_step_size:, :]
            image_t1_left[:, -actions_step_size:, :] = fill_value

            image_t1_up[:-actions_step_size, :, :] = next_states[actions_step_size:, :, :]
            image_t1_up[-actions_step_size:, :, :] = fill_value

        np.putmask(next_states, actions_hat == 1, image_t1_up)
        np.putmask(next_states, actions_hat == 2, image_t1_down)
        np.putmask(next_states, actions_hat == 3, image_t1_left)
        np.putmask(next_states, actions_hat == 4, image_t1_right)

        if self.debug:
            cv2.imshow('Game', next_states)

        done_with_eps = False
        done_with_season = False
        if self.current_episode != self.next_episode():
            done_with_eps = True
        if self.current_season != self.next_season():
            done_with_season = True

        return next_states.copy(), done_with_season, done_with_eps

    @staticmethod
    def shift_image(states, actions_step_size, fill_value=0):

        up = np.empty_like(states)
        down = np.empty_like(states)
        left = np.empty_like(states)
        right = np.empty_like(states)
        if len(states.shape) == 4:

            right[:, :actions_step_size, :, :] = fill_value
            right[:, actions_step_size:, :, :] = states[:, :-actions_step_size, :, :]

            left[:, actions_step_size:, :, :] = fill_value
            left[:, :actions_step_size, :, :] = states[:, -actions_step_size:, :, :]

            down[:, :, :actions_step_size, :] = fill_value
            down[:, :, actions_step_size:, :] = states[:, :, :-actions_step_size, :]

            up[:, :, actions_step_size:, :] = fill_value
            up[:, :, :actions_step_size, :] = states[:, :, -actions_step_size:, :]

        else:

            right[:actions_step_size, :, :] = fill_value
            right[actions_step_size:, :, :] = states[:-actions_step_size, :, :]

            left[actions_step_size:, :, :] = fill_value
            left[:actions_step_size, :, :] = states[-actions_step_size:, :, :]

            down[:, :actions_step_size, :] = fill_value
            down[:, actions_step_size:, :] = states[:, :-actions_step_size, :]

            up[:, actions_step_size:, :] = fill_value
            up[:, :actions_step_size, :] = states[:, -actions_step_size:, :]

        return up, down, left, right

    def translate_actions(self, actions_hat, actions_step_size, fill_value=0):
        states = self.current_states
        image_t1_stay = states

        image_t1_up = np.empty_like(states)
        image_t1_down = np.empty_like(states)
        image_t1_left = np.empty_like(states)
        image_t1_right = np.empty_like(states)

        if len(states.shape) == 4:

            image_t1_down[:, actions_step_size:, :, :] = states[:, :-actions_step_size, :, :]
            image_t1_down[:, :actions_step_size, :, :] = fill_value

            image_t1_right[:, :, actions_step_size:, :] = states[:, :, :-actions_step_size, :]
            image_t1_right[:, :, :actions_step_size, :] = fill_value

            image_t1_left[:, :, :-actions_step_size, :] = states[:, :, actions_step_size:, :]
            image_t1_left[:, :, -actions_step_size:, :] = fill_value

            image_t1_up[:, :-actions_step_size, :, :] = states[:, actions_step_size:, :, :]
            image_t1_up[:, -actions_step_size:, :, :] = fill_value

        else:

            image_t1_down[actions_step_size:, :, :] = states[:-actions_step_size, :, :]
            image_t1_down[:actions_step_size, :, :] = fill_value

            image_t1_right[:, actions_step_size:, :] = states[:, :-actions_step_size, :]
            image_t1_right[:, :actions_step_size, :] = fill_value

            image_t1_left[:, :-actions_step_size, :] = states[:, actions_step_size:, :]
            image_t1_left[:, -actions_step_size:, :] = fill_value

            image_t1_up[:-actions_step_size, :, :] = states[actions_step_size:, :, :]
            image_t1_up[-actions_step_size:, :, :] = fill_value

        np.putmask(image_t1_stay, actions_hat == 1, image_t1_up)
        np.putmask(image_t1_stay, actions_hat == 2, image_t1_down)
        np.putmask(image_t1_stay, actions_hat == 3, image_t1_left)
        np.putmask(image_t1_stay, actions_hat == 4, image_t1_right)

        return image_t1_stay

    def get_scenes_for_batch_experience(self, experience):
        replay0 = []
        replay1 = []
        for season, eps, scn_num, path_to_scene in experience:
            scene = self.get_scene_from_path(path_to_scene)[0]  # get the whole scene as np array
            scenes_images = [image for image in scene]
            replay0 += scenes_images[:-1]
            replay1 += scenes_images[1:]

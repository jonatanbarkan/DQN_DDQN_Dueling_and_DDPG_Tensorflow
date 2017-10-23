# import gym
from modules.dqn import *
# from modules.env_utils import *
from own.world import World
from own.learner import Learner
import time
import logging
import shutil
import os
import sys

import tensorflow as tf
import gc
gc.enable()


# Save video rendering every Xth game played
def multiples_video_schedule(episode_id):
    return episode_id % 100 == 0  # and episode_id>0
    # return episode_id % 200 == 0


print()
print("Usage:")
print("  ", sys.argv[0], " [optional: path_to_ckpt_file] [optional: True/False test mode]")
print()
print()

# outdir = "delme"


TOTAL_FRAMES = 20000000  # TRAIN
MAX_TRAINING_STEPS = 20*60*60/3  # MAX STEPS BEFORE RESETTING THE ENVIRONMENT
TESTING_GAMES = 30  # no. of games to average on during testing
MAX_TESTING_STEPS = 5*60*60/3  # 5 minutes  '/3' because gym repeating the last action 3-4 times already!
TRAIN_AFTER_FRAMES = 50000
epoch_size = 50000  # every how many frames to test

MAX_NOOP_START = 30


outdir = "curiosity_results"
ENV_NAME = 'bigbang'
LOG_DIR = outdir+'/'+ENV_NAME+'/logs/'
if os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR)
journalist = tf.train.SummaryWriter(LOG_DIR)

# Build environment
# env = AtariEnvWrapper(ENV_NAME)
validation_episodes = [('10', str(eps)) for eps in range(20, 25)]
test_episodes = [('06', '12'), ('10', '04')]
master_path = '/home/goren/PycharmProjects/jonatan/movies/all/mean_countedges_min_thresh_0.7_min_num_frames20/5fps/'
env = World(ENV_NAME, master_path, validation_episodes, test_episodes)


# Initialize Tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)


# Create DQN agent
UseDoubleDQN = True

agent = DQN(state_size=env.observation_space_shape,
            action_size=env.action_space_size,
            session=session,
            summary_writer=journalist,
            exploration_period=1000000,
            minibatch_size=None,
            discount_factor=0.9,
            experience_replay_buffer=500,
            target_qnet_update_frequency=50,  # 30000 if UseDoubleDQN else 10000, ## Tuned DDQN
            initial_exploration_epsilon=1.0,
            final_exploration_epsilon=0.1,
            reward_clipping=1.0,
            DoubleDQN=UseDoubleDQN)


session.run(tf.initialize_all_variables())
journalist.add_graph(session.graph)

saver = tf.train.Saver(tf.all_variables())

# env.monitor.start(outdir+'/'+ENV_NAME, force=True, video_callable=multiples_video_schedule)
# env.monitor.start(outdir+'/'+ENV_NAME, force=True, video_callable=multiples_video_schedule)
logger = logging.getLogger()
logging.disable(logging.INFO)

# If an argument is supplied, load the specific checkpoint.
test_mode = False
if len(sys.argv) >= 2:
    saver.restore(session, sys.argv[1])
if len(sys.argv) == 3:
    test_mode = sys.argv[2] == 'True'

num_frames = 0
num_games = 0

current_game_frames = 0
init_no_ops = np.random.randint(MAX_NOOP_START+1)

last_time = time.time()
last_frame_count = 0.0


state = env.reset()
actions_step_size = agent.action_size
# while num_frames <= TOTAL_FRAMES+1:
while True:
    # if test_mode:
    #     env.render()

    # num_frames += 1
    # current_game_frames += 1

    # Pick action given current state
    if not test_mode:
        action = agent.action(state, training=True)
    else:
        action = agent.action(state, training=False)

    # if current_game_frames < init_no_ops:
    #     action = 0

    # Perform the selected action on the environment
    # next_state, reward, done, _ = env.step(action)
    next_state, done_with_season, done_with_episode = env.step(action, actions_step_size)
    reward = agent.compute_reward(state, next_state)

    # Store experience
    if current_game_frames >= init_no_ops:
        agent.store(state, action, reward, next_state, done_with_season, done_with_episode)
    state = next_state

    # Train agent
    if num_frames >= TRAIN_AFTER_FRAMES:
        agent.train()

    if done or current_game_frames > MAX_TRAINING_STEPS:
        state = env.reset()
        current_game_frames = 0
        num_games += 1
        init_no_ops = np.random.randint(MAX_NOOP_START+1)

    # Print an update
    if num_frames % 10000 == 0:
        new_time = time.time()
        diff = new_time - last_time
        last_time = new_time

        elapsed_frames = num_frames - last_frame_count
        last_frame_count = num_frames

        print("frames: ", num_frames, "    games: ", num_games, "    speed: ", (elapsed_frames/diff), " frames/second")

    # Save the network's parameters after every epoch
    if num_frames % epoch_size == 0 and num_frames > TRAIN_AFTER_FRAMES:
        saver.save(session, outdir+"/"+ENV_NAME+"/model_"+str(num_frames/1000)+"k.ckpt")

        print()
        print("epoch:  frames=", num_frames, "   games=", num_games)

    # Testing -- it's kind of slow, so we're only going to test every 2 epochs
    if num_frames % (2*epoch_size) == 0  and num_frames > TRAIN_AFTER_FRAMES:
        total_reward = 0
        avg_steps = 0
        for i in xrange(TESTING_GAMES):
            state = env.reset()
            init_no_ops = np.random.randint(MAX_NOOP_START+1)
            frm = 0
            while frm < MAX_TESTING_STEPS:
                frm += 1
                # env.render()
                action = agent.action(state, training = False) # direct action for test

                if current_game_frames < init_no_ops:
                    action = 0

                state, reward, done, _ = env.step(action)

                total_reward += reward
                if done:
                    break

            avg_steps += frm
        avg_reward = float(total_reward)/TESTING_GAMES

        str_ = session.run( tf.scalar_summary('test reward ('+str(epoch_size/1000)+'k)', avg_reward) )
        journalist.add_summary(str_, num_frames) #np.round(num_frames/epoch_size)) # in no. of epochs, as in Mnih

        print '  --> Evaluation Average Reward: ',avg_reward,'   avg steps: ',(avg_steps/TESTING_GAMES)

        state = env.reset()

env.monitor.close()
journalist.close()

## Save the final network
saver.save(session, outdir+"/"+ENV_NAME+"/final.ckpt")




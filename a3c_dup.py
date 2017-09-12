import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
from vizdoom import *
import sys
from PIL import Image

from random import choice
from time import sleep
from time import time

IMPOSSIBLE = -1.0
FP_UPDATE_LENGTH = 30
AC_UPDATE_LENGTH = 30

eStart = 1.0
eEnd = 0.1
annealingStep = 1000
stepDrop = (eStart - eEnd) / annealingStep


#Define the function to update weights between master and workers
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    #Define the update operations which will be executed later
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
#clip and resize input image
def process_frame(frame):
    s = frame[10 : -10, 30 : -30] # Discard the first a few of dimensions
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0 #Flatten it
    return s
    

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def normalize_vector(v):
    minv = np.min(v)
    maxv = np.max(v)
    w = v
    v = v - minv
    if sum(v) == 0:
        if sum(w) == 0:
            print 'w', w
            sys.exit("Error message")
        v = w / sum(w)
    else:
        if sum(v) == 0:
            print 'v', v
            sys.exit("Error message")
        v = v / sum(v)
    return v


class FramePrediction_Network(object):
    def __init__(self, h_size, scope, retro_step, trainer):
        with tf.variable_scope(scope):
            #self.inputs = tf.placeholder(tf.float32, [None, (retro_step + 1) * 32])
            self.inputs = tf.placeholder(tf.float32)
            self.imageIn = tf.reshape(self.inputs, [-1, 84 * (retro_step + 1), 84, 1])


            #Conv layers
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=128, kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv2, num_outputs=32, kernel_size=[2, 2], stride=[1, 1], padding='VALID')
            #One FC layer
            hidden1 = slim.fully_connected(slim.flatten(self.conv3), h_size, activation_fn=tf.nn.elu)
            
            hidden2 = slim.fully_connected(hidden1, h_size / 2, activation_fn=tf.nn.elu)
            hidden3 = slim.fully_connected(hidden2, h_size / 4, activation_fn=tf.nn.elu)
            hidden4 = slim.fully_connected(hidden3, h_size / 8, activation_fn=tf.nn.elu)

            self.predicted_observation = slim.fully_connected(hidden4, 7056, activation_fn=tf.nn.relu)
            self.predicted_reward = slim.fully_connected(hidden4, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(0.1), biases_initializer=None)

            self.predicted_done = slim.fully_connected(hidden4, 1, activation_fn=tf.nn.sigmoid, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
            

            if scope != 'global_FPN':
                #self.true_observation = tf.placeholder(shape=[None, 32], dtype=tf.float32)
                self.true_observation = tf.placeholder(dtype=tf.float32)
                self.true_reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
                self.true_done = tf.placeholder(shape=[None, 1], dtype=tf.float32)

                self.observation_loss = tf.reduce_sum(tf.square(self.true_observation - self.predicted_observation))
                self.reward_loss = tf.reduce_sum(tf.square(self.true_reward - self.predicted_reward))
                self.done_loss = tf.reduce_sum(-tf.log(tf.multiply(self.predicted_done, self.true_done) + tf.multiply(1 - self.predicted_done, 1 - self.true_done)))
                self.loss = tf.reduce_mean(self.observation_loss + self.reward_loss + 10 * self.done_loss)
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                
                #Apply local updates to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_FPN')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
                           
class AC_Network(object):
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32) #Original Input image
            self.imageIn = tf.reshape(self.inputs, [-1, 84, 84, 1]) #Reshape input
            #Conv layers
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.imageIn, num_outputs=16, kernel_size=[8, 8], stride=[4, 4], padding='VALID')           
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu, inputs=self.conv1, num_outputs=32, kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            #One FC layer
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            #RNN
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1] #Number of images fed in?
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size, time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :]) #Why the first?
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            

            #Policy and Value
            self.policy = slim.fully_connected(rnn_out, a_size, activation_fn=tf.nn.softmax, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
            
            self.value = slim.fully_connected(rnn_out, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)

            #if this is a worker
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
                
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                
                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss =-tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Calculate local updates
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                #Apply local updates to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
                
class Worker(object):
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes, retro_step):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.retro_buffer = []
        self.retro_step = retro_step
        self.a_size = a_size
        self.e = eStart
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))


        #Create local copy of the network and the tensorflow op to copy global parameters to local network, for the policy network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)


        #Create local copy of the network and the tensorflow op to copy global parameters to local network, for the prediction network
        self.name_FPN = self.name + '_FPN'
        self.local_FP = FramePrediction_Network(256, self.name_FPN, retro_step, trainer)
        self.update_local_ops_FPN = update_target_graph('global_FPN', self.name_FPN)

        #Set up the specific game environment
        game.set_doom_scenario_path("basic.wad")
        game.set_doom_map("map01")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        game.add_available_button(Button.MOVE_LEFT) 
        game.add_available_button(Button.MOVE_RIGHT) 
        game.add_available_button(Button.ATTACK) 
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.set_episode_timeout(300)
        game.set_episode_start_time(10)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        game.init()
        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        self.env = game


    #Obtain losses for local workers
    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]
        
        #Generate advantage, using Generalized Advantage Estimation
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     self.local_AC.inputs:np.vstack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages,
                     self.local_AC.state_in[0]:self.batch_rnn_state[0],
                     self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)        

        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n


    def train_FPN(self, rollout, sess, retro_step):
        rollout = np.array(rollout)
        observations = rollout[: , 0]
        actions = rollout[:, 1]
        #rewards = rollout[:, 2]
        #dones = rollout[:, 3]
        #stack data by retro_step
        past_history = []
        for i in range(retro_step, len(rollout)):
            one_past = np.vstack(observations[(i - retro_step) : i])
            one_action = actions[i - 1 : i]            
            #broadcast the action to a shape that is similar to one image
            one_action = np.full(one_past[0].shape, one_action)
            one_moment = np.vstack((one_past, one_action))
            past_history.append(one_moment)
        next_observations = np.vstack(observations[retro_step:])
        rewards = np.vstack(rollout[retro_step:, 2])
        dones = np.vstack(rollout[retro_step:, 3])
        past_history = np.array(past_history)
        feed_dict = {self.local_FP.inputs:past_history, self.local_FP.true_observation:next_observations, self.local_FP.true_reward:rewards, self.local_FP.true_done:dones}
        o_l, r_l, d_l, loss, _ = sess.run([self.local_FP.observation_loss, self.local_FP.reward_loss, self.local_FP.done_loss, self.local_FP.loss, self.local_FP.apply_grads], feed_dict=feed_dict)
        
        return loss / (len(rollout) - retro_step), o_l / (len(rollout) - retro_step), r_l / (len(rollout) - retro_step), d_l / (len(rollout) - retro_step)



        
    def predict_frame(self, action, play_time):
        one_past = np.array(self.retro_buffer)
        one_action = np.full(one_past[0].shape, action)
        one_moment = np.vstack((one_past, one_action))
        done = False
        p_ob, p_r, p_d = sess.run([self.local_FP.predicted_observation, self.local_FP.predicted_reward, self.local_FP.predicted_done], feed_dict={self.local_FP.inputs:one_moment})
        if p_d[0][0] > 0.1 or play_time >= 300:
            done = True
        p_r = p_r[0][0] if done == True else IMPOSSIBLE
        return p_ob, p_r, done

            
            


    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        e = self.e
        with sess.as_default(), sess.graph.as_default():           
        #The worker starts to play
            while not coord.should_stop():
                sess.run(self.update_local_ops) # Fetch network weights from the master network
                episode_buffer = []
                experience_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                rnn_state_pre = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                #The worker starts to play for one episode
                while self.env.is_episode_finished() == False:
                    feed_dict = {self.local_AC.inputs:[s],
                                 self.local_AC.state_in[0]:rnn_state[0],
                                 self.local_AC.state_in[1]:rnn_state[1]}
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out], feed_dict=feed_dict)


                    #a = np.random.choice(a_dist[0], p=a_dist[0])
                    #a = np.argmax(a_dist == a)
                    seed = np.random.random()
                    e = -1000
                    if seed < e:
                        a = np.random.randint(0, a_size)
                    else:
                        if len(self.retro_buffer) < self.retro_step:
                            #Sample an action from the learnt policy, given the input image
 #                           a = np.random.choice(a_dist[0], p=a_dist[0])
 #                           a = np.argmax(a_dist == a)
                            a = np.random.randint(0, a_size)
                        else:
                            #Instead of actually executing the action, we make several steps of predictions to find the best action move combo, now we just use one step
                            reward_holder = []
                            for ac in range(a_size):
                                next_frame, rew , d = self.predict_frame(ac, len(experience_buffer))                                                               
#                                for idx, image in enumerate(self.retro_buffer):
#                                    image = np.array(image)
#                                    image = np.reshape(image, [84, 84])
#                                    im = Image.fromarray(image)
#                                    im = im.convert('RGB')
#                                    im.save('frames/image' + str(total_steps) + str(idx) + '.png')
 #                               im = Image.fromarray(np.reshape(next_frame, [84, 84]))
 #                               im = im.convert('RGB')
 #                               im.save('frames/image' + str(total_steps) + 'pred.png')
                                feed_dict = {self.local_AC.inputs:next_frame, 
                                             self.local_AC.state_in[0]:rnn_state_pre[0],
                                             self.local_AC.state_in[1]:rnn_state_pre[1]}
                                v, rnn_state_pre = sess.run([self.local_AC.value, self.local_AC.state_out], feed_dict=feed_dict)
                                reward_holder.append(rew + v[0, 0])                            
                            reward_holder = np.array(reward_holder)                              
                            if sum(reward_holder == IMPOSSIBLE) == a_size:
                                a = np.random.randint(0, a_size)
                            else:
                                p_dist = normalize_vector(reward_holder)
                                print p_dist
                                a = np.random.choice(p_dist, p=p_dist)
                                a = np.argmax(p_dist == a)

                    e -= stepDrop
#                    if e > eEnd:
#                        e -= stepDrop                    

                    r = self.env.make_action(self.actions[a]) / 100.0
                    d = self.env.is_episode_finished()
                    if d == False: #Still playing 
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                        #im = Image.fromarray(np.reshape(s1, [84, 84]))
                        #im = im.convert('RGB')
                        #im.save('frames/image' + str(total_steps) + '.png')
                    else: #game over
                        s1 = s
                    
                    #Stack the stats for this move into the buffer
                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    experience_buffer.append([s, a, r, d])
                    episode_values.append(v[0, 0])
                    #Update the retro_buffer
                    if len(self.retro_buffer) < self.retro_step:
                        self.retro_buffer.append(s)
                    else:
                        self.retro_buffer.pop(0)
                        self.retro_buffer.append(s)

               
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    
                    #If the episode hasn't ended but the buffer is full, update the global as well the local networks
                    if len(episode_buffer) == AC_UPDATE_LENGTH and d != True and episode_step_count != max_episode_length - 1:
                        feed_dict = {self.local_AC.inputs:[s], 
                                     self.local_AC.state_in[0]:rnn_state[0],
                                     self.local_AC.state_in[1]:rnn_state[1]}
                        v1 = sess.run(self.local_AC.value, feed_dict=feed_dict)[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        #After you have made an update, clear the buffer
                        episode_buffer = []
                        #Fetch the latest network weights from master
                        sess.run(self.update_local_ops)


                    #If the episode hasn't ended but the buffer is full, update the global as well the local networks, This tome for the FPN
                    if len(experience_buffer) == FP_UPDATE_LENGTH and d != True and episode_step_count != max_episode_length - 1:
                        l, o_l, r_l, d_l  = self.train_FPN(experience_buffer, sess, self.retro_step)
                        #After you have made an update, clear the buffer
                        experience_buffer = []
                        #Fetch the latest network weights from master
                        sess.run(self.update_local_ops_FPN)




                    #If the episode is over
                    if d == True:
                        break
                #Stack episode stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                #Since one episode is over, make an update
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)
                if len(experience_buffer) > retro_step:
                    l, o_l, r_l, d_l = self.train_FPN(experience_buffer, sess, self.retro_step)
                    

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif', duration=len(images) * time_per_step, true_image=True, salience=False)
                            
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print ('Saved Model')

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                        
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary.value.add(tag='Losses/Prediction Loss', simple_value=float(l))
                    summary.value.add(tag='Losses/Observation Loss', simple_value=float(o_l))
                    summary.value.add(tag='Losses/Reward Loss', simple_value=float(r_l))
                    summary.value.add(tag='Losses/Done Loss', simple_value=float(d_l))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()                        
                    #increase the global episode number
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


#Configure some parameters
max_episode_length = 300
gamma = .99
s_size = 7056
a_size = 3
h_size = 256
load_model = False
retro_step = 3
model_path = './model'

tf.reset_default_graph()
if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
    #The master network that accepts gradients from workers
    master_network = AC_Network(s_size, a_size, 'global', None)
    master_network_FPN = FramePrediction_Network(h_size, 'global_FPN', 3, None)
    num_workers = 3
    workers = []
    #Define 'num_workers' workers
    for i in range(num_workers):
        workers.append(Worker(DoomGame(), i, s_size, a_size, trainer, model_path, global_episodes, retro_step))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    #If load model from checkpoint
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_heckpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else: #Start a new model
        sess.run(tf.global_variables_initializer())

    worker_threads = []
    #Start the worker threads
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)













                        




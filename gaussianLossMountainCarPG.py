import gym
import numpy as np
#working

env = gym.make('MountainCarContinuous-v0')

N_EPOCHS = 1000
N_EPS_PER_BATCH = 4
n_inputs = 2
n_HL1 = 40
n_HL2 = 1
REWARD_THRESH = 70
ACTION_STD_DEV = 0.5
EPSILON = 1e-13

GAMMA = 0.9999
ALPHA = 1e-3

W1 = np.random.rand(n_HL1, n_inputs) / np.sqrt(n_inputs) #He Initializer
W2 = np.random.rand(n_HL2, n_HL1) / np.sqrt(n_HL1)

def activate(z, activation = 'sigmoid'):
	
	if activation == 'sigmoid':
		a = 1. / (1. + np.exp(-z))
	
	elif activation == 'relu':
		a = x.copy()
		a[z<0] = 0
	elif activation == 'tanh':
		t1 = np.exp(z)
		t2 = 1. / t1
		a = (t1 - t2) / (t1 + t2)

	return a

def forward_prop(inputs):

	z_1 = np.matmul(W1, inputs)
	a_1 = activate(z_1, 'sigmoid')
	z_2 = np.matmul(W2, a_1)
	pred = activate(z_2, 'tanh')
	return pred, a_1


def backward_prop(states, predictions, actions, a_1s, advantages):
	global W1, W2

	# print states.shape
	# print predictions.shape
	# print actions.shape
	# print a_1s.shape
	# print advantages.shape
	# print '+'*40

	d_a_2 = actions - predictions
	d_z_2 = d_a_2 * (1 - predictions ** 2)
	d_W2 = np.matmul(d_z_2.T , a_1s)
	d_a_1 = np.matmul(d_z_2, W2)
	d_z_1 = a_1s * (1 - a_1s) * d_a_1
	d_W1 = np.matmul(d_z_1.T, states)

	# print d_z_2.shape
	# print d_W2.shape
	# print d_a_1.shape
	# print d_z_1.shape
	# print d_W1.shape

	#weight update::

	# print W1

	W1 += ALPHA * d_W1
	W2 += ALPHA * d_W2

	# print W1
	# exit()

def sample_action(mean, sd):

	return np.random.normal(mean, sd)

epoch_sd = ACTION_STD_DEV

for i in range(N_EPOCHS):
	ep_states = []
	ep_actions = []
	ep_a_1 = []
	ep_preds = []
	ep_advantages = []
	epoch_reward = 0
	epoch_sd = epoch_sd * np.exp(-i * 1./400000)
	for j in range(N_EPS_PER_BATCH):
		
		state = env.reset()
		temp_adv_list = []
		episode_time = 0
		episode_reward = 0

		while True:

			episode_time += 1

			#run inference to predict a mean::
			action_mean, a_1 = forward_prop(state)
			action = sample_action(action_mean, epoch_sd)

			ep_states.append(state)
			ep_actions.append(action)
			ep_a_1.append(a_1)
			ep_preds.append(action_mean)

			#take the sampled action on env::
			# print action_mean, epoch_sd,action
			next_state, reward, done, info = env.step(action)

			state = next_state
			episode_reward += reward

			if episode_reward >= REWARD_THRESH:
				advantage = 1
			else:
				advantage = -1

			if i > N_EPOCHS - 100:
				env.render()

			if done or episode_reward >= REWARD_THRESH:
				break

		#prepare the advantage list for the episode::
		epoch_reward += episode_reward

		factor = advantage
		for k in range(episode_time):
			temp_adv_list.append(factor)
			factor *= GAMMA
		temp_adv_list.reverse()
		
		#append to the list of advantages
		ep_advantages = ep_advantages + temp_adv_list
		print 'EPISODE NUMBER : {}\tTIME : {}\t REWARD : {}'.format(j, episode_time, episode_reward)

	# if i % 20 == 0:
	print '===========  EPOCH NUMBER : {}\t AVERAGE REWARD : {}  ==========='.format(i, epoch_reward/N_EPS_PER_BATCH)
	#convert all to numpy arrays::
	ep_advantages = np.array(ep_advantages)
	ep_advantages = np.reshape(ep_advantages,(ep_advantages.shape[0],1))
	ep_states = np.array(ep_states)
	ep_actions = np.array(ep_actions)
	ep_preds = np.array(ep_preds)
	ep_a_1 = np.array(ep_a_1)

	#backprop to update weights ::
	backward_prop(ep_states, ep_preds, ep_actions, ep_a_1, ep_advantages)
import os
import h5py
import urllib.request
import numpy as np
import pickle
import gzip

# TODO : @dhruvramani - add non-mujoco based tasks
_ENV_URL_DICT = {
	'hopper-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
	'halfcheetah-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
	'walker2d-medium-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
	'hopper-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
	'halfcheetah-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
	'walker2d-expert-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
	'hopper-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
	'halfcheetah-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
	'walker2d-random-v0' : 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5', 
	'hopper-mixed-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
	'walker2d-mixed-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker_mixed.hdf5',
	'halfcheetah-mixed-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
	'walker2d-medium-expert-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
	'halfcheetah-medium-expert-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
	'hopper-medium-expert-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
	'ant-medium-expert-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium_expert.hdf5',
	'ant-mixed-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_mixed.hdf5',
	'ant-medium-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_medium.hdf5',
	'ant-random-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_random.hdf5',
	'ant-expert-v0' :'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/ant_expert.hdf5',
}

def get_keys(h5file):
	keys = []
	def visitor(name, item):
		if isinstance(item, h5py.Dataset):
			keys.append(name)
	h5file.visititems(visitor)
	return keys

def get_dataset(env_name, h5path=None):
	global _ENV_URL_DICT
	dataset_url = _ENV_URL_DICT[env_name.lower()]
	if h5path is None:
		h5path = "./{}".format(dataset_url.split("/")[-1])
		if dataset_url is None:
			raise ValueError("Offline env not configured with a dataset URL.")

		if not os.path.exists(h5path):
			print('Downloading dataset:', dataset_url, 'to', h5path)
			urllib.request.urlretrieve(dataset_url, h5path)

		if not os.path.exists(h5path):
			raise IOError("Failed to download dataset from %s" % dataset_url)
        
	dataset_file = h5py.File(h5path, 'r')
	data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
	dataset_file.close()

	# Run a few quick sanity checks
	for key in ['observations', 'actions', 'rewards', 'terminals']:
		assert key in data_dict, 'Dataset is missing key %s' % key
	return data_dict

class ReplayBuffer(object):
	def __init__(self, state_dim=10, action_dim=4):
		self.storage = dict()
		self.storage['observations'] = np.zeros((1000000, state_dim), np.float32)
		self.storage['next_observations'] = np.zeros((1000000, state_dim), np.float32)
		self.storage['actions'] = np.zeros((1000000, action_dim), np.float32)
		self.storage['rewards'] = np.zeros((1000000, 1), np.float32)
		self.storage['terminals'] = np.zeros((1000000, 1), np.float32)
		self.storage['bootstrap_mask'] = np.zeros((10000000, 4), np.float32)
		self.buffer_size = 1000000
		self.ctr = 0

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage['observations'][self.ctr] = data[0]
		self.storage['next_observations'][self.ctr] = data[1]
		self.storage['actions'][self.ctr] = data[2]
		self.storage['rewards'][self.ctr] = data[3]
		self.storage['terminals'][self.ctr] = data[4]
		self.ctr += 1
		self.ctr = self.ctr % self.buffer_size

	def sample(self, batch_size, with_data_policy=False):
		ind = np.random.randint(0, self.storage['observations'].shape[0], size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		s = self.storage['observations'][ind]
		a = self.storage['actions'][ind]
		r = self.storage['rewards'][ind]
		s2 = self.storage['next_observations'][ind]
		d = self.storage['terminals'][ind]
		mask = self.storage['bootstrap_mask'][ind]

		if with_data_policy:
				data_mean = self.storage['data_policy_mean'][ind]
				data_cov = self.storage['data_policy_logvar'][ind]

				return (np.array(s), 
						np.array(s2), 
						np.array(a), 
						np.array(r).reshape(-1, 1), 
						np.array(d).reshape(-1, 1),
						np.array(mask),
						np.array(data_mean),
						np.array(data_cov))

		return (np.array(s), 
				np.array(s2), 
				np.array(a), 
				np.array(r).reshape(-1, 1), 
				np.array(d).reshape(-1, 1),
				np.array(mask))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage)

	def load(self, filename, bootstrap_dim=None):
		"""Deprecated, use load_hdf5 in main.py with the D4RL environments""" 
		with gzip.open(filename, 'rb') as f:
				self.storage = pickle.load(f)
		
		sum_returns = self.storage['rewards'].sum()
		num_traj = self.storage['terminals'].sum()
		if num_traj == 0:
				num_traj = 1000
		average_per_traj_return = sum_returns/num_traj
		print ("Average Return: ", average_per_traj_return)
		# import ipdb; ipdb.set_trace()
		
		num_samples = self.storage['observations'].shape[0]
		if bootstrap_dim is not None:
				self.bootstrap_dim = bootstrap_dim
				bootstrap_mask = np.random.binomial(n=1, size=(1, num_samples, bootstrap_dim,), p=0.8)
				bootstrap_mask = np.squeeze(bootstrap_mask, axis=0)
				self.storage['bootstrap_mask'] = bootstrap_mask[:num_samples]

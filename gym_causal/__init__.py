from gym.envs.registration import register

register(
	id='causal-v0',
	entry_point='gym_causal.envs:CausalEnv',
)
register(
	id='causal-extrahard-v0',
	entry_point='gym_causal.envs:CausalExtraHardEnv',
)

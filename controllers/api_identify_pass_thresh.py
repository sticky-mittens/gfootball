import gfootball.env as football_env

env = football_env.create_environment(
    env_name='1_vs_1_easy',#'11_vs_11_stochastic',
    representation='raw',#'simple115v2',
    rewards='checkpoints,scoring',
    number_of_left_players_agent_controls=1,
    render=True)
env.reset()

done = False
i = 0
while not done:
    #action = env.action_space.sample()
    #print(action)
    #exit(0)
    action = [10]
    observation, reward, done, info = env.step(action)
    print(i, observation[0]['ball'], observation[0]['ball_owned_team'], '\n')
    i += 1
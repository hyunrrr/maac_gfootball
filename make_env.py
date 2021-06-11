def make_soccer_env(num_agents):

    import gfootball.env as football_env

    env = football_env.create_environment(
            env_name='11_vs_11_easy_stochastic', stacked=False,
            logdir='/tmp/maddpg_soccer',
            dump_frequency=0,
            representation='simple115v2',
            number_of_left_players_agent_controls=num_agents,
            #channel_dimensions=(43, 43),
            render=False
            )

    return env


def make_soccer_env_rollout(num_agents):

    import gfootball.env as football_env

    env = football_env.create_environment(
            env_name='11_vs_11_easy_stochastic', stacked=False,
            logdir='/tmp/maddpg_soccer',
            dump_frequency=0,
            representation='simple115v2',
            number_of_left_players_agent_controls=num_agents,
            #channel_dimensions=(43, 43),
            render=True
            )

    return env
from dql.environment.cartpole_gym_env import CartpoleEnvDQL


def main():
    env = CartpoleEnvDQL(plot=True)
    agt = DQLAgent(env, load_network_path='latest')

    agt.learn(100000)


if __name__ == '__main__':
    main()

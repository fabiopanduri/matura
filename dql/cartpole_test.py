from dql.agent.agent import DQLAgent
from dql.environment.cartpole_gym_env import CartpoleEnvDQL


def main():
    env = CartpoleEnvDQL(plot=True)
    agt = DQLAgent(env)

    agt.learn(100000)


if __name__ == '__main__':
    main()

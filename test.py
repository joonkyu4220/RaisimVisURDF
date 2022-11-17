if __name__ == '__main__':
    import os
    import time
    from ruamel.yaml import YAML, dump, RoundTripDumper
    from raisimGymTorch.env.bin import RaisimVisURDF
    from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
    
    import torch
    import torch.utils.data

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
    control_dt = float(cfg["environment"]["control_dt"])

    # create environment from the configuration file
    env = RaisimVisURDF.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnv(RaisimVisURDF.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
    print("env_created")
    env.setTask()
    env.reset()
    
    max_length = 10000000
    for i in range(max_length):
        env.step(torch.zeros(1))
        time.sleep(control_dt)
        
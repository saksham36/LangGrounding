if __name__ == "__main__": 
    import argparse
    import time
    import datetime
    import torch_ac
    import tensorboardX
    import sys

    import utils
    from utils import device
    from model import ACModel, DQNModel, BootDQNModel
    from scripts import custom_algo

    from tqdm import tqdm
    import math
    import numpy as np
    import random


    # Parse arguments

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--num_episodes", type=int, default=10**7,
                        help="number of episodes of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--num_ensemble", type=int, default=2, help="number of ensembles for bootstrapped DQN")
    parser.add_argument("--mask_prob", type=float, default=0.9, help="mask probabilities for bootstrapped DQN")

    args = parser.parse_args()
    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    print(f'Model_name: {model_name}')
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environment
    env = utils.make_env(args.env, args.seed + 0)
   
    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_episodes": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    if args.algo=="BootDQN":
         model = BootDQNModel(obs_space, env.action_space, args.num_ensemble, args.mask_prob, args.mem, args.text)

    if "model_state" in status:
        model.load_state_dict(status["model_state"])
    model.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(model))
    # Load algo
    if args.algo == "BootDQN":
        algo = custom_algo.BootDQNAlgo([env], model, device, args.frames_per_proc, args.discount, args.lr, args.max_grad_norm, args.recurrence,
                                preprocess_obss, 
                                num_ensemble=args.num_ensemble, mask_prob=args.mask_prob)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model
    num_frames = status["num_episodes"]
    update = status["update"]
    start_time = time.time()
    print(50*"*")
    timestep = 0
    update = 0
    for eps_idx in tqdm(range(num_frames,args.num_episodes)):
        # Update model parameters
        env = utils.make_env(args.env, args.seed + 10000 * eps_idx)
        obs = env.reset()
        Q_idx = random.randint(0,args.num_ensemble-1) # active head
        episode_reward = 0
        done = False
        discount = 1
        log_episode_return = 0
        while not done: # Collect experience using a given Q_idx
            # Generate an action from the agent's policy.
            action = algo.select_action(obs, Q_idx)
            
            # Step the environment.
            obs_, reward, done, _ = env.step(action)

            # Tell the agent about what just happened.
            timestep += 1
            # Sample minibatch and update
            logs_temp = algo.update(obs, action, reward, obs_, timestep)
            if logs_temp is not None and len(logs_temp) > 0: #Only store the one last updated
                logs = logs_temp


            # Book keeping
            obs = obs_
            episode_reward += reward

        # Log keeping
        log_episode_return = episode_reward
        num_frames+=1
        # Print logs

        if eps_idx % args.log_interval == 0:
            update += 1
            duration = int(time.time() - start_time)
            return_per_episode = episode_reward

            header = ["update", "Num_Episodes", "duration"]
            data = [update, eps_idx+1, duration]
            header += ["return"]
            data += [return_per_episode]
            
            header += ["loss_" + str(key) for key in logs]
            data += [np.mean([logs[key]["loss"].data for key in logs])]
            txt_logger.info(
            "U {} | F {:06} | D {} | rR:μσmM {:.2f} | Mean Loss {:.3f}"
            .format(*data))

            if eps_idx == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and eps_idx % args.save_interval == 0:
            status = {"num_episodes": eps_idx+1, "update": update,
                    "model_state": model.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

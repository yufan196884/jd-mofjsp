from typing import Union
import numpy as np
from torch.distributions import Categorical
import torch
import time
import copy

from src.agents.heuristic.heuristic_agent import HeuristicSelectionAgent
from src.utils.evaluations import EvaluationHandler

from tensordict.tensordict import TensorDict

from torchrl.modules import MaskedCategorical

def run_episode_rl(env, model, lambda_weights, handler: EvaluationHandler, greedy: bool = False, episode_iter: int = 1, max_steps = 1000, data_index=None, device='cpu') -> None:
    """
    Executes one testing episode with an RL model, supporting both greedy and non-greedy action selection
    
    :param env: TorchRL environment
    :param model: RL model object
    :param handler: EvaluationHandler object
    :param greedy: Whether to use greedy action selection (highest logit) or sampling
    :param episode_iter: Number of episodes to run for non-greedy sampling
    :param device: Device to run the model on (e.g., 'cpu' or 'cuda')
    
    :return: Environment with best performance
    """
    # For non-greedy sampling, we'll run multiple episodes and keep track of the best one
    if not greedy:
        num_iterations = episode_iter
        min_makespan = np.infty
        env_min = None
    else:
        num_iterations = 1

    if hasattr(model, "model"):
        model.model.to(device)

    env.time = time.time()
    envs_return = []

    total_steps = 0
    total_reward = 0

    for iteration in range(num_iterations):
        # Reset environment to start new episode
        td = env.reset(lambda_weights=lambda_weights)
        obs = td
        done = False
        steps = 0
        episode_reward = 0
        
        while not done:
            steps += 1
            total_steps += 1
            obs = obs.to(device)
            
            # Get action from RL model
            with torch.no_grad():
                # Create TensorDict for model input
                model_input = obs

                # Select action based on policy (greedy or sampling)
                if hasattr(model, "policy_actor"):
                    actor_output = model.policy_actor(model_input.clone())
                    if greedy:
                        logits = actor_output["logits"]
                        if ("mask" in model_input and
                                not model.config.get("heuristic_action_space", False)):
                            logits = logits.masked_fill(~model_input["mask"], float("-inf"))
                        action = logits.argmax(dim=-1).unsqueeze(-1)
                    else:
                        action = actor_output["action"]

                elif hasattr(model, "select_epsilon_greedy_dqn"):
                    mask = model_input.get("dynamic_pair_mask_tensor", None)
                    num_jobs = env.num_jobs
                    num_machines = env.num_machines

                    action = model.select_greedy_dqn(model_input.clone(), mask=mask)
                    action = model.dqn_action_to_env_action(env.max_num_machines, num_machines, action)

                elif hasattr(model, "model"):
                    # Using a wrapped model (like from Model.load)
                    
                    # network_inputs = {key: model_input[key] for key in model_input.keys()}
                    
                    # Forward pass through the model
                    
                    if env.config.get("model_type", "MLP").lower() == "mlp":
                        logits = model.model(model_input["input"])
                    else:
                        logits, _ = model.model(**model_input)

                    if greedy:
                        
                        # Get action mask if available
                        mask = model_input.get("mask", None)
                        if mask is not None and not env.config.get("heuristic_action_space", False):
                            # Apply mask to logits
                            masked_logits = logits.clone()
                            masked_logits = masked_logits.masked_fill(~mask, float("-inf"))
                            action = masked_logits.argmax(dim=-1).unsqueeze(-1)
                        else:
                            action = logits.argmax(dim=-1).unsqueeze(-1)

                    else:
                        # Sample from the distribution
                        mask = model_input.get("mask", None)
                        if mask is not None and not env.config.get("heuristic_action_space", False):
                            logits = logits.clone()
                            dist = MaskedCategorical(logits=logits, mask=mask)
                            action = dist.sample().unsqueeze(-1)
                        else:
                            dist = Categorical(logits=logits)
                            action = dist.sample().unsqueeze(-1)

                else:
                    # Generic predict interface
                    action = model.predict(obs, deterministic=greedy)
                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, device=env.device)
                    if action.dim() == 0:
                        action = action.unsqueeze(0).unsqueeze(0)

            # Update the observation TensorDict with the action
            action = action.detach().cpu()
            obs = obs.set("action", action)
            
            # Take step in environment
            next_td = env.step(obs)
            
            # Extract reward, done flag, and check for constraint breaks
            reward = next_td.get(("next", "reward"))
            done = next_td.get(("next", "done")).item()
            
            # Detect constraint breakage (typically indicated by a large negative reward)
            if isinstance(reward, TensorDict):
                reward_value = reward.get("reward").item()
            else:
                reward_value = reward.mean().item()

            
            # Update for next step
            obs = next_td.get("next")
            episode_reward += reward_value

        
        # After episode completion
        envs_return.append((copy.deepcopy(env), episode_reward))

        if iteration != (num_iterations - 1):
            # Reset for next iteration if not the final one
            obs = env.reset()

    # Calculate mean reward
    mean_reward = total_reward / total_steps if total_steps > 0 else 0

    return envs_return, mean_reward


def run_episode(env, 
                model, 
                heuristic_id: Union[str, None], 
                handler: EvaluationHandler, 
                greedy: bool = False, 
                episode_iter: int = 100, 
                max_steps = 1000, 
                data_index = None,
                weights = [0.0,1.0]) -> None:
    """
    This function executes one testing episode with a TorchRL environment
    
    :param env: TorchRL environment
    :param model: Model object. E.g. PPO object
    :param heuristic_id: Heuristic identifier. Can be None
    :param handler: EvaluationHandler object
    :param greedy: Whether to use greedy action selection
    :param episode_iter: Number of episodes to run for random sampling
    :param max_steps: Maximum number of steps before terminating the episode
    :param data_index: Index for tracking the data instance
    
    :return: Environment with best performance and success flag
    """
    
    heuristic_agent = HeuristicSelectionAgent() if heuristic_id else None
    
    if heuristic_id == 'rand':
        num_iterations = episode_iter
    else:
        num_iterations = 1


    envs_return = []

    total_steps = 0
    total_reward = 0

    for iteration in range(num_iterations):
        # Reset environment to start new episode
        td = env.reset()
        obs = td
        done = False
        steps = 0
        episode_reward = 0
        constraint_broken = False
        
        while not done:
            steps += 1
            total_steps += 1
            
            # Get action from model or heuristic
            if heuristic_id:
                action = heuristic_agent(env, heuristic_id)
                Use_Heuristic = True
                # Create TensorDict with action
                action_td = TensorDict({"action": torch.tensor(action, device=env.device).unsqueeze(0)}, 
                                      batch_size=[1])
            else:
                Use_Heuristic = False
                # Get action from RL model
                with torch.no_grad():
                    # Extract observation components
                    mask = obs.get("mask")
                    
                    # Use model to predict action (adjusted for your model output format)
                    if hasattr(model, "get_action_and_value"):
                        # PPO-style API
                        action, _, _ = model.get_action_and_value(obs, deterministic=greedy)
                    else:
                        # Generic prediction API
                        action = model.predict(obs, deterministic=greedy)
                    
                    # Create TensorDict with action
                    action_td = TensorDict({"action": action.unsqueeze(0)}, batch_size=[1])
            
            # Take step in environment
            result_td = env.step(action_td, heuristic=Use_Heuristic)
            
            # Extract next observation, reward, and done signal
            next_obs = result_td
            reward = result_td.get(("next", "reward"))
            done = result_td.get(("next", "done"))
            
            # For scalar rewards in TensorDict
            if isinstance(reward, TensorDict):
                reward = reward.get("reward")
            
            # Convert to scalar for constraint checking
            reward_value = reward.mean().item()
            
            
            # Update tracking variables
            episode_reward += reward_value
            total_reward += reward_value
            obs = next_obs
            
            # Check if max steps reached
            if steps >= max_steps:
                done = True
                print(f"Maximum steps ({max_steps}) reached - terminating episode")

        # After episode completion
        envs_return.append((copy.deepcopy(env), episode_reward))

        if iteration != (num_iterations - 1):
            # Reset for next iteration if not the final one
            obs = env.reset()

    # Calculate mean reward
    mean_reward = total_reward / total_steps if total_steps > 0 else 0

    return envs_return, mean_reward
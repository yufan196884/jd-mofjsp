import torch.nn as nn
import torch
from torch.optim import Adam
import numpy as np
import random
from typing import List
from pathlib import Path
from tqdm import tqdm

from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator, MaskedCategorical
from torch.distributions import Categorical

from src.agents.FiLMMLP import MLPNetwork

from src.utils.logger import Logger
from src.agents.reinforcement_learning.Utils import explained_variance, MakespanLRScheduler, generate_distractor_preferences

class Model:
    """Simple wrapper class for loaded models"""
    pass

class MultiCriticHeadedMLP(nn.Module):

    def __init__(self, mlp_network, out_head_idx=0, *args, **kwargs):
        super(MultiCriticHeadedMLP, self).__init__(*args, **kwargs)
        self.mlp_network = mlp_network
        self.out_head_idx = out_head_idx

    def forward(self, input):
        # Forward pass through the shared layers
        shared_output = self.mlp_network(input)
        return shared_output[:, self.out_head_idx:self.out_head_idx+1]  # Return only the output of the specified head
        


class MaskedPPO_MLP:
    def __init__(self, env, config: dict, logger: Logger = None):
        """
        PPO implementation for DANIEL network with the Flex4Green environment using TorchRL
        
        Args:
            env: The environment to train on
            config: Configuration dictionary
            logger: Logger for tracking metrics
        """
        print("    Initializing MLP agent")
        self.env = env
        self.config = config
        self.network_keys = ['input']  # Example input keys for the MLP network
        
        # PPO hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.n_epochs = config.get('n_epochs', 10)
        self.rollout_steps = config.get('rollout_steps', 2048)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.batch_size = config.get('batch_size', 256)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Diversity loss hyperparameters
        self.diversity_alpha = config.get('diversity_alpha', 1.0)
        self.diversity_scale = config.get('diversity_scale', 5.0)

        # Training tracking
        self.num_timesteps = 0
        self.n_updates = 0
        self.intermediate_test_interval = config.get('intermediate_test_interval', 1)
        
        # Device setup
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device_train = torch.device(self.config.get('device', 'cuda'))
        else:
            self.device_train = torch.device("cpu")
        print(f"Training device: {self.device_train}")
        
        # Logger
        self.logger = logger if logger else Logger(config=config)
        
        # Set random seed
        self.seed = config.get('seed', None)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        input_dim = env.observation_spec['input'].shape[-1]
        self.mlp_network = MLPNetwork(input_dim=input_dim, output_dim=env.action_spec.high[0], config=config)

        # Two separate critic networks (makespan and energy)
        self.mlp_network_critic_makespan = MLPNetwork(
            input_dim=input_dim, output_dim=1, config=config
        )
        self.mlp_network_critic_energy = MLPNetwork(
            input_dim=input_dim, output_dim=1, config=config
        )

        self.value_operator_makespan = ValueOperator(
            in_keys=['input'],
            out_keys=["state_value"],
            module=self.mlp_network_critic_makespan
        )

        self.value_operator_energy = ValueOperator(
            in_keys=['input'],
            out_keys=["state_value"],
            module=self.mlp_network_critic_energy
        )
        self.critic_module_makespan = TensorDictModule(
            module=self.mlp_network_critic_makespan,
            in_keys=['input'],
            out_keys=['state_value']
        )
        self.critic_module_energy = TensorDictModule(
            module=self.mlp_network_critic_energy,
            in_keys=['input'],
            out_keys=['state_value']
        )


        if self.config.get("heuristic_action_space", False):
            in_keys_actor = ['logits']
        else:
            in_keys_actor = ['logits', 'mask']
        
        # Create actor and critic modules using TensorDict wrappers
        self.policy_module = TensorDictModule(
            module=self.mlp_network,
            in_keys=['input'],
            out_keys=["logits"]
        )



        # Create the probabilistic actor that handles action selection
        self.policy_actor = ProbabilisticActor(
            module=self.policy_module,
            spec=env.action_spec,
            in_keys=in_keys_actor,
            out_keys=["action"],
            distribution_class=Categorical if self.config.get("heuristic_action_space", False) else MaskedCategorical, # Use Categorical for multi-discrete action space
            return_log_prob=True
        )
        
        # Create the data collector
        _ = self.env.reset()
        self.collector = SyncDataCollector(
            env,
            self.policy_actor,
            frames_per_batch=self.rollout_steps,
            total_frames=self.rollout_steps * config.get('number_rollouts', 10),
            split_trajs=False,
            device=self.device
        )

        # Create advantage estimation module with our custom value network
        self.advantage_module_makespan = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=None,#value_module,
            average_gae=True,
            vectorized=True,
            device=self.device_train,
        )
        self.advantage_module_makespan.set_keys(value="state_value_makespan")
        self.advantage_module_makespan.set_keys(reward="Reward_Makespan")
        self.advantage_module_makespan.set_keys(advantage="advantage_makespan")
        self.advantage_module_makespan.set_keys(value_target="value_target_makespan")
        
        self.advantage_module_energy = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=None,#value_module,
            average_gae=True,
            vectorized=True,
            device=self.device_train,
        )
        self.advantage_module_energy.set_keys(value="state_value_energy")
        self.advantage_module_energy.set_keys(reward="Reward_Energy")
        self.advantage_module_energy.set_keys(advantage="advantage_energy")
        self.advantage_module_energy.set_keys(value_target="value_target_energy")

        # Create PPO loss module
        self.loss_makespan = ClipPPOLoss(
            actor_network=self.policy_actor,
            critic_network=self.critic_module_makespan,
            clip_epsilon=self.clip_range,
            entropy_coef=self.ent_coef,
            critic_coef=self.value_coef,
            loss_critic_type="smooth_l1",
            reduction="none"  # Keep individual losses for makespan and energy to combine them manually
        )
        self.loss_makespan.set_keys(advantage="advantage_makespan")
        self.loss_makespan.set_keys(value_target="value_target_makespan")

        self.loss_energy = ClipPPOLoss(
            actor_network=self.policy_actor,
            critic_network=self.critic_module_energy,
            clip_epsilon=self.clip_range,
            entropy_coef=self.ent_coef,
            critic_coef=self.value_coef,
            loss_critic_type="smooth_l1",
            reduction="none"  # Keep individual losses for makespan and energy to combine them manually
        )
        self.loss_energy.set_keys(advantage="advantage_energy")
        self.loss_energy.set_keys(value_target="value_target_energy")

        # Create optimizer
        self.optimizer = Adam(list(self.policy_actor.parameters()) +
                        list(self.critic_module_makespan.parameters()) +
                        list(self.critic_module_energy.parameters()),
                        lr=self.learning_rate)
    
        # Learning rate scheduler
        self.scheduler = MakespanLRScheduler(
            optimizer=self.optimizer,
            factor=config.get('lr_factor', 1.0),
            patience=config.get('lr_patience', 10),
            min_lr=float(config.get('min_lr', 1e-6)),
            verbose=True
        )
        
        # Initialize replay buffer for rollouts
        self.rollout_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.rollout_steps),
            sampler=SamplerWithoutReplacement()
        )

    def _sync_policy_to_collector(self) -> None:
        """Sync latest policy weights to the collector-side policy (CPU collection path)."""
        if not hasattr(self, "collector"):
            return

        try:
            collector_policy = getattr(self.collector, "policy", None)
            if collector_policy is not None and collector_policy is not self.policy_actor:
                state_dict_cpu = {k: v.detach().cpu() for k, v in self.policy_actor.state_dict().items()}
                collector_policy.load_state_dict(state_dict_cpu, strict=True)

            if hasattr(self.collector, "update_policy_weights_"):
                self.collector.update_policy_weights_()
        except Exception as e:
            print(f"[PolicySync] Warning: failed to sync policy to collector: {e}")


    @staticmethod
    def get_MLP_sizes(config: dict) -> List[int]:
        from src.agents.train_test_utility_functions import load_config


        data_config = load_config(config["data_generation_yaml"], {})
        max_num_jobs = data_config.get('jobs_config', {}).get('max', 50)
        max_num_machines = data_config.get('machines_config', {}).get('max', 25)
        max_num_operations = max_num_jobs * data_config.get('task_config', {}).get('max', 50)
        horizon =  data_config.get('Energy_info:', {}).get('forecast_horizon', 500)
        input_dim = max_num_jobs + \
                    max_num_jobs * max_num_machines + \
                    max_num_jobs * max_num_machines + \
                    max_num_jobs * max_num_machines + \
                    max_num_machines + \
                    max_num_operations + \
                    max_num_operations + \
                    (horizon * 2) + 2
        output_dim = 16 if config.get("heuristic_action_space", False) else max_num_jobs * max_num_machines

        return [input_dim, output_dim]


    @classmethod
    def load(cls, file: Path, config: dict, logger: Logger = None):
        """
        Load a pre-trained model. The output dimension is read from the saved
        checkpoint so that train-time and test-time action-space sizes cannot
        silently diverge.
        """
        model = Model()

        data = torch.load(f"{file}.pkl", map_location=torch.device('cpu'))
        state = data["model_params"]

        MLP_config = MaskedPPO_MLP.get_MLP_sizes(config)
        input_dim = MLP_config[0]

        # Derive output_dim from the last linear layer of the checkpoint
        linear_keys = sorted(
            [k for k in state.keys() if k.endswith("linear.weight")],
            key=lambda s: int(s.split(".")[1])
        )
        output_dim = state[linear_keys[-1]].shape[0]

        mlp_network = MLPNetwork(input_dim=input_dim, output_dim=output_dim, config=config)
        mlp_network.load_state_dict(state)

        model.model = mlp_network
        return model
    
    def save(self, file: Path) -> None:
        """
        Save the current model
        
        Args:
            file: Path to save the model to
        """
        data = {
            "model_params": self.mlp_network.state_dict(),
            "config": self.config
        }
        
        torch.save(data, f"{file}.pkl")

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Main training loop using TorchRL
        
        Args:
            total_instances: Number of rollouts to collect
            total_timesteps: Total timesteps to train for
            intermediate_test: Intermediate test callback
        """
        # Determine total number of rollouts based on timesteps or instances
        total_rollouts = min(total_instances, total_timesteps // self.rollout_steps)
        
        # Main training loop
        for i, data in tqdm(enumerate(self.collector)):

            if i >= total_rollouts:
                break
            
            print(f"\n--- Training iteration {i+1}/{total_rollouts} ---")

            # Flatten batch dimensions
            data = data.reshape(-1)
            
            # Track episode statistics, using completed episodes only
            if "next" in data and "makespan" in data["next"]:
                if "done" in data["next"]:
                    done_indices = torch.nonzero(data["next", "done"], as_tuple=True)[0]

                    if len(done_indices) > 0:
                        mean_makespan = data["next", "makespan"].float()[done_indices].mean().item()
                        mean_energy_costs = data["next", "Energy_Consumed"].float()[done_indices].mean().item()
                        mean_total_costs = data["next", "Total_Costs"].float()[done_indices].mean().item()

                    else:
                        mean_makespan = torch.nan
                        mean_energy_costs = torch.nan
                        mean_total_costs = torch.nan

            
            # Calculate mean reward
            mean_training_reward = data["next", "reward"].float().mean().item()

            
            # Move data to training device
            data = data.to(self.device_train)
            
            # Pre-compute value estimates in batches for makespan and energy
            self.mlp_network_critic_makespan.to(self.device_train)
            self.mlp_network_critic_makespan.eval()
            self.mlp_network_critic_energy.to(self.device_train)
            self.mlp_network_critic_energy.eval()

            num_frames = data.shape[0]
            state_values_makespan = torch.zeros(num_frames, 1, device=self.device_train)
            state_values_energy = torch.zeros(num_frames, 1, device=self.device_train)
            next_state_values_makespan = torch.zeros(num_frames, 1, device=self.device_train)
            next_state_values_energy = torch.zeros(num_frames, 1, device=self.device_train)

            with torch.no_grad():
                for j in range(0, num_frames, self.batch_size):
                    batch_slice = slice(j, min(j + self.batch_size, num_frames))
                    batch_data = data[batch_slice]

                    # Compute V(s_t) for the current batch
                    current_values_td_makespan = self.mlp_network_critic_makespan(
                        input=batch_data["input"]
                    )
                    state_values_makespan[batch_slice] = current_values_td_makespan

                    current_values_td_energy = self.mlp_network_critic_energy(
                        input=batch_data["input"]
                    )
                    state_values_energy[batch_slice] = current_values_td_energy


                    # Compute V(s_{t+1}) for the current batch
                    if "next" in batch_data.keys(include_nested=True):
                        next_obs_td = batch_data["next"]
                        input_next = next_obs_td.get("input", None)

                        computed_next_values_td_makespan = self.value_operator_makespan(input=input_next)
                        computed_next_values_makespan = computed_next_values_td_makespan                     
                        computed_next_values_td_energy = self.value_operator_energy(input=input_next)
                        computed_next_values_energy = computed_next_values_td_energy


                        # Set V(s_{t+1}) = 0 for terminal states
                        if "done" in batch_data["next"].keys():
                            terminals = batch_data["next", "done"].bool().squeeze(-1)
                            computed_next_values_makespan[terminals] = 0.0
                            computed_next_values_energy[terminals] = 0.0
                        next_state_values_makespan[batch_slice] = computed_next_values_makespan
                        next_state_values_energy[batch_slice] = computed_next_values_energy
                        
                    else:
                         print(f"Warning: 'next' key group missing in batch slice {batch_slice}. Setting V(s_t+1)=0.")
                         next_state_values_makespan[batch_slice] = 0.0
                         next_state_values_energy[batch_slice] = 0.0

            self.mlp_network_critic_makespan.train()
            self.mlp_network_critic_energy.train()

            # GAE uses keys "state_value" and ("next", "state_value") by default
            data["state_value_makespan"] = state_values_makespan
            data["state_value_energy"] = state_values_energy
            data[("next", "state_value_makespan")] = next_state_values_makespan
            data[("next", "state_value_energy")] = next_state_values_energy

            # GAE finds the pre-computed 'state_value' keys and skips its internal value computation
            self.advantage_module_makespan.to(self.device_train)
            sample = self.advantage_module_makespan(data)

            self.advantage_module_energy.to(self.device_train)
            sample_energy_tmp = self.advantage_module_energy(data)
            sample["advantage_energy"] = sample_energy_tmp["advantage_energy"]
            del sample_energy_tmp

            # Extend replay buffer with new data
            self.rollout_buffer.extend(sample)
            
            # Tracking loss values
            Clip_losses, Diversity_losses, value_losses_makespan, value_losses_energy, entropy_losses, total_losses = [], [], [], [], [], []
            
            self.policy_actor.to(self.device_train)
            
            # PPO training loop
            for epoch in range(self.n_epochs):
                for _ in range(self.rollout_steps // self.batch_size):
                    # Sample a batch from the replay buffer
                    subdata = self.rollout_buffer.sample(self.batch_size).to(self.device_train)
                    

                    loss_vals_makespan = self.loss_makespan(subdata)
                    loss_vals_energy = self.loss_energy(subdata)
                    
                    Loss_critic_makespan = torch.mean(loss_vals_makespan["loss_critic"])
                    Loss_ent_makespan = torch.mean(loss_vals_makespan["loss_entropy"])
                    Loss_critic_energy = torch.mean(loss_vals_energy["loss_critic"])

                    Preferences = subdata["Preference"]
                    Loss_Clip = torch.mean((loss_vals_makespan["loss_objective"] * Preferences[:, 0]) + (loss_vals_energy["loss_objective"] * Preferences[:, 1]))

                    distractor_preferences = generate_distractor_preferences(Preferences, mean=0, std=0.1)
                    input_distractor = subdata["input"].clone()
                    logits_data = self.policy_module(input=subdata["input"])
                    input_distractor[:, -2:] = distractor_preferences  # Assuming the last two features are the preference vector
                    with torch.no_grad():
                        logits_distractor = self.mlp_network(input_distractor)
                    Loss_Diversity = self.calculate_diversity_loss_depr(Preferences, 
                                                                    logits_data, 
                                                                    distractor_preferences, 
                                                                    logits_distractor, 
                                                                    subdata["mask"] if subdata["mask"].shape[1] == logits_distractor.shape[1] else None, 
                                                                    alpha=1.0)

                    loss_value = (
                        Loss_Clip 
                        + self.diversity_scale * Loss_Diversity
                        + Loss_critic_makespan
                        + Loss_ent_makespan # entropy is shared between makespan and energy since the actor network is shared
                        + Loss_critic_energy
                    )

                    # Track losses
                    Clip_losses.append(Loss_Clip.item())
                    Diversity_losses.append(Loss_Diversity.item())
                    value_losses_makespan.append(Loss_critic_makespan.item())
                    value_losses_energy.append(Loss_critic_energy.item())
                    entropy_losses.append(Loss_ent_makespan.item())
                    
                    # Total loss
                    total_losses.append(loss_value.item())
                    
                    # Gradient update
                    loss_value.backward()
                    
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                        list(self.policy_actor.parameters()) +
                        list(self.critic_module_makespan.parameters()) +
                        list(self.critic_module_energy.parameters()),
                        self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Free memory
                    del loss_vals_makespan, loss_vals_energy, loss_value, subdata

            # Sync policy to collector
            if self.device_train.type == "cuda" and self.device.type == "cpu":
                self._sync_policy_to_collector()

            # Clear replay buffer
            self.rollout_buffer.empty()
            
            # Move networks back to evaluation device
            self.policy_actor.to(self.device)
            self.mlp_network_critic_energy.to(self.device)
            self.mlp_network_critic_makespan.to(self.device)
            
            # Update step counter
            self.num_timesteps += self.rollout_steps
            
            # Calculate explained variance
            explained_var_makespan = explained_variance(
                data['value_target_makespan'].squeeze(1),  # Target values
                data['state_value_makespan'].squeeze(1)    # Critic's predictions
            )
            explained_var_energy = explained_variance(
                data['value_target_energy'].squeeze(1),  # Target values
                data['state_value_energy'].squeeze(1)    # Critic's predictions
            )
            
            mean_state_value_makespan = data['state_value_makespan'].mean().item()
            mean_state_value_energy   = data['state_value_energy'].mean().item()
            mean_reward_makespan = data["next", "Reward_Makespan"].float().mean().item()
            mean_reward_energy   = data["next", "Reward_Energy"].float().mean().item()

            
            # Free memory
            del data
            
            # Log training results
            self.logger.record({
                'results_on_train_dataset/mean_reward': mean_training_reward,
                'results_on_train_dataset/mean_makespan': mean_makespan,
                'results_on_train_dataset/mean_energy_costs': mean_energy_costs,
                'results_on_train_dataset/mean_total_costs': mean_total_costs,
                'agent_training/n_updates': self.num_timesteps,
                'agent_training/loss': torch.mean(torch.tensor(total_losses)),
                'agent_training/value_loss_makespan': torch.mean(torch.tensor(value_losses_makespan)),
                'agent_training/value_loss_energy': torch.mean(torch.tensor(value_losses_energy)),
                'agent_training/clip_loss': torch.mean(torch.tensor(Clip_losses)),
                'agent_training/diversity_loss': torch.mean(torch.tensor(Diversity_losses)),
                'agent_training/entropy_loss': torch.mean(torch.tensor(entropy_losses)),
                'agent_training/explained_variance_makespan': explained_var_makespan,
                'agent_training/explained_variance_energy': explained_var_energy,
                'agent_training/mean_reward_makespan': mean_reward_makespan,
                'agent_training/mean_reward_energy':   mean_reward_energy,
                'agent_training/mean_advantage_makespan': sample["advantage_makespan"].float().mean().item(),
                'agent_training/mean_advantage_energy':   sample["advantage_energy"].float().mean().item(),
                'agent_training/mean_state_value_makespan': mean_state_value_makespan,
                'agent_training/mean_state_value_energy':    mean_state_value_energy,
            })
            self.logger.dump()
            
            # Run intermediate testing if available
            if intermediate_test and (i % self.intermediate_test_interval == 0 and i != 0):
                self.policy_actor.to(self.device)
                val_reward = intermediate_test.on_step(self.rollout_steps*i, self)

                # Negate the reward since higher reward is better but the scheduler minimizes
                self.scheduler.step(-val_reward)
                
                # Log current learning rate
                self.logger.record({
                    'agent_training/learning_rate': self.scheduler.get_lr()
                })
                self.logger.dump()


    def calculate_diversity_loss_depr(self, preferences, logits, distractor_preferences, distractor_logits, mask, alpha=1.0):      
        """
        Calculate a diversity loss to encourage the model to learn from both objectives
        
        Args:
            preferences: Original preference vector
            logits: Original action logits
            distractor_preferences: Perturbed preference vector for the distractor
            distractor_logits: Action logits for the distractor input
            mask: Action mask for the original input
            alpha: Weighting factor for the preference difference in the diversity loss 
        """

        # Calculate KL divergence between original and distractor logits
        if mask is None:
            dist = Categorical(logits=logits)  # Original distribution without mask
            distractor_dist = Categorical(logits=distractor_logits)  # Distractor distribution
        else:
            dist = MaskedCategorical(logits=logits, mask=mask)  # Original distribution with mask
            distractor_dist = MaskedCategorical(logits=distractor_logits, mask=mask)  # Distractor distribution with same mask
        
        # Replace 0.00 with very small value to avoid log(0) issues in KL divergence calculation
        dist_clamped = torch.clamp(dist.probs, min=1e-8)
        distractor_dist_clamped = torch.clamp(distractor_dist.probs, min=1e-8)
        dist = Categorical(probs=dist_clamped)
        distractor_dist = Categorical(probs=distractor_dist_clamped)
        actual_kl_divergence = torch.distributions.kl.kl_divergence(dist, distractor_dist).mean()
        
        # Calculate mean absolute difference in preferences
        with torch.no_grad():
            preference_diff = torch.abs(preferences - distractor_preferences).sum(dim=1).mean()
            target_kl_divergence = alpha * preference_diff

        diversity_loss = (target_kl_divergence - actual_kl_divergence).pow(2)

        if diversity_loss == torch.inf:
            print("Warning: Diversity loss is infinite. Check for numerical issues in KL divergence calculation.")

        return diversity_loss
    
    def calculate_diversity_loss(self, preferences, logits, distractor_preferences, distractor_logits, mask, alpha=1.0):
        eps = 1e-8
        logit_clip = 20.0
        kl_cap = 1e3

        # sanitize logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=logit_clip, neginf=-logit_clip).clamp(-logit_clip, logit_clip)
        distractor_logits = torch.nan_to_num(
            distractor_logits, nan=0.0, posinf=logit_clip, neginf=-logit_clip
        ).clamp(-logit_clip, logit_clip)

        if mask is not None:
            mask = mask.bool()
            # ensure at least one valid action per row
            valid_rows = mask.any(dim=-1, keepdim=True)
            if not valid_rows.all():
                # fallback: if row is fully invalid, make all actions valid for that row
                mask = torch.where(valid_rows, mask, torch.ones_like(mask, dtype=torch.bool))

            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, neg_inf)
            distractor_logits = distractor_logits.masked_fill(~mask, neg_inf)

        p = torch.nn.functional.softmax(logits, dim=-1)
        q = torch.nn.functional.softmax(distractor_logits, dim=-1)

        if mask is not None:
            p = p.masked_fill(~mask, 0.0)
            q = q.masked_fill(~mask, 0.0)

        # renormalize + clamp
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)

        kl_per_sample = (p * (p.log() - q.log())).sum(dim=-1)
        kl_per_sample = torch.nan_to_num(kl_per_sample, nan=0.0, posinf=kl_cap, neginf=0.0)
        actual_kl_divergence = kl_per_sample.mean().clamp_max(kl_cap)

        with torch.no_grad():
            preference_diff = torch.abs(preferences - distractor_preferences).sum(dim=1).mean()
            target_kl_divergence = alpha * preference_diff

        diversity_loss = (target_kl_divergence - actual_kl_divergence).pow(2)
        diversity_loss = torch.nan_to_num(diversity_loss, nan=0.0, posinf=kl_cap, neginf=0.0)
        return diversity_loss
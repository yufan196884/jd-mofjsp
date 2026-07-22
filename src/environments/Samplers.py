import numpy as np
from typing import Optional

from src.data_generator.sp_factory import SPFactory
from scipy.stats import gaussian_kde

class PreferenceSampler:
    def __init__(self, history_limit=1000):
        # We only store w1 because w2 is always (1 - w1)
        self.pref_hist = [] 
        self.history_limit = history_limit

    def add_to_history(self, weight_vector):
        """Adds a used [w1, w2] vector to the history."""
        self.pref_hist.append(weight_vector[0])
        if len(self.pref_hist) > self.history_limit:
            self.pref_hist.pop(0)

    def sample_uniform_preference(self, candidate_size=50):
        """
        Generates a new preference vector by picking the candidate 
        with the lowest current density in the history.
        """
        # 1. Handle cold start: If history is empty, pick purely at random
        if len(self.pref_hist) < 5:
            w1 = np.random.rand()
            return np.array([w1, 1.0 - w1])

        # 2. Fit KDE to the existing history of w1 values
        # We use a small silverman bandwidth to keep the density sensitive
        kde = gaussian_kde(self.pref_hist)

        # 3. Generate a batch of random candidates [0, 1]
        candidates = np.random.rand(candidate_size)

        # 4. Evaluate density at each candidate point
        densities = kde.evaluate(candidates)

        # 5. Pick the candidate with the MINIMUM density (the most undersampled area)
        best_w1 = candidates[np.argmin(densities)]
        
        return np.array([best_w1, 1.0 - best_w1])


class IndexDrivenSampler:
    """
    Sampler that returns a specific instance based on the provided index.
    Can be shared across multiple environments.
    """
    def __init__(self, data, 
                 config: dict = {}, 
                 mode: str = "train", 
                 seed: Optional[int] = None,
                 On_fly_data_generation: bool = False):
        """
        Args:
            instances: List of instances to sample from
            mode: "train" or "test" - determines sampling strategy
            seed: Random seed for reproducibility
        """
        self.num_instances = len(data)
        self.mode = mode
        self.seed = seed
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        self.On_fly_data_generation = On_fly_data_generation
        if self.On_fly_data_generation:
            del data

        # For training mode, prepare shuffled indices
        if mode == "train":
            self.all_indices = np.arange(self.num_instances)
            self.rng.shuffle(self.all_indices)
            self.available_indices = set(self.all_indices)
        # For test mode, use fixed sequential order
        else:
            self.all_indices = np.arange(self.num_instances)
            self.next_index = 0
    
    def get_instance(self, index: Optional[int] = None) -> int:
        """
        Get an instance index based on the specified strategy.
        
        Args:
            index: If provided, returns this specific instance.
                  If None, samples based on the mode (train/test)
        
        Returns:
            Index of the instance to use
        """
        if index is not None:
            # Ensure index is within bounds
            return index % self.num_instances
            
        # If no specific index provided, sample based on mode
        if self.mode == "train":
            # Training mode: sample without replacement
            if not self.available_indices:
                # If all indices have been used, reset
                self.available_indices = set(self.all_indices)
                # Optional: re-shuffle for the next epoch
                self.rng.shuffle(self.all_indices)
                self.available_indices = set(self.all_indices)
            
            # Sample a random index from available indices
            index = self.rng.choice(list(self.available_indices))
            self.available_indices.remove(index)
            return index
        else:
            # Test mode: sequential sampling
            index = self.next_index
            self.next_index = (self.next_index + 1) % self.num_instances

            return index
            
    def reset(self):
        """Reset the sampler state"""
        if self.mode == "train":
            self.rng.shuffle(self.all_indices)
            self.available_indices = set(self.all_indices)
        else:
            self.next_index = 0
            
    def set_mode(self, mode: str):
        """Change the sampling mode"""
        if mode not in ["train", "test"]:
            raise ValueError(f"Mode must be 'train' or 'test', got {mode}")
        self.mode = mode
        self.reset()
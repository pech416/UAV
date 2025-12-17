"""
Vectorized operations for the UAV project.
Provides vectorized implementations to replace for loops for better performance.
"""

import numpy as np
import math
#from configs.config import Config

class VectorizedOps:
    """
    Vectorized operations for UAV task processing.
    
    This class provides vectorized implementations of common operations
    to improve computational efficiency.
    """
    
    @staticmethod
    def calculate_distances_batch(uav_positions, user_positions, cloud_position):
        """
        Calculate distances between all UAVs and users in a vectorized manner.
        
        Args:
            uav_positions (np.ndarray): UAV positions array of shape (n_uavs, 3) [x, y, z]
            user_positions (np.ndarray): User positions array of shape (n_users, 2) [x, y]
            cloud_position (list): Cloud position [x, y]
            
        Returns:
            tuple: (distances_uav_user, distances_uav_cloud)
                - distances_uav_user: shape (n_uavs, n_users)
                - distances_uav_cloud: shape (n_uavs,)
        """
        n_uavs = uav_positions.shape[0]
        n_users = user_positions.shape[0]
        
        # Calculate UAV to user distances
        # Expand dimensions for broadcasting
        uav_xy = uav_positions[:, :2]  # (n_uavs, 2)
        uav_z = uav_positions[:, 2]    # (n_uavs,)
        
        # Calculate horizontal distances: (n_uavs, n_users)
        dx = uav_xy[:, 0:1] - user_positions[:, 0]  # (n_uavs, n_users)
        dy = uav_xy[:, 1:2] - user_positions[:, 1]  # (n_uavs, n_users)
        d_level = np.sqrt(dx**2 + dy**2)
        
        # Calculate 3D distances
        distances_uav_user = np.sqrt(d_level**2 + uav_z[:, np.newaxis]**2)
        
        # Calculate UAV to cloud distances
        cloud_xy = np.array(cloud_position)
        dx_cloud = uav_xy[:, 0] - cloud_xy[0]
        dy_cloud = uav_xy[:, 1] - cloud_xy[1]
        distances_uav_cloud = np.sqrt(dx_cloud**2 + dy_cloud**2)
        
        return distances_uav_user, distances_uav_cloud
    
    @staticmethod
    def calculate_transmission_rates_batch(distances_uav_user, distances_uav_cloud, uav_params):
        """
        Calculate transmission rates for all UAV-user and UAV-cloud pairs.
        
        Args:
            distances_uav_user (np.ndarray): Distances between UAVs and users (n_uavs, n_users)
            distances_uav_cloud (np.ndarray): Distances between UAVs and cloud (n_uavs,)
            uav_params (dict): UAV parameters (P_u, P_v_w, r, sigma2, N, B, B_cloud)
            
        Returns:
            tuple: (user_to_uav_rates, uav_to_cloud_rates)
                - user_to_uav_rates: shape (n_uavs, n_users)
                - uav_to_cloud_rates: shape (n_uavs,)
        """
        # UAV to cloud transmission rate using Shannon formula
        uav_to_cloud_rates = (
            10 * uav_params['B_cloud'] * 
            np.log2(1 + 10000 * uav_params['P_u'] * 
                   np.power(distances_uav_cloud / 100, -uav_params['r']) / 
                   (uav_params['sigma2'] + uav_params['N']))
        )
        
        # User to UAV transmission rate using Shannon formula
        # Expand cloud rates for broadcasting
        user_to_uav_rates = (
            100 * uav_params['B'] * 
            np.log2(1 + 5000 * (uav_params['P_v_w'] * 
                   np.power(distances_uav_user / 10, -uav_params['r']) / 
                   (uav_params['sigma2'] + uav_params['N'])))
        )
        
        return user_to_uav_rates, uav_to_cloud_rates
    
    @staticmethod
    def find_best_uav_for_tasks(tasks, uavs, users, cloud_position):
        """
        Find the best UAV for each task in a vectorized manner.
        
        Args:
            tasks (list): List of task lists for each user
            uavs (list): List of UAV objects
            users (list): List of User objects
            cloud_position (list): Cloud position [x, y]
            
        Returns:
            dict: Dictionary mapping (user_idx, task_idx) to (best_uav_idx, min_time)
        """
        # Extract UAV positions
        uav_positions = np.array([uav.position for uav in uavs])
        user_positions = np.array([user.position for user in users])
        
        # Calculate distances in batch
        distances_uav_user, distances_uav_cloud = VectorizedOps.calculate_distances_batch(
            uav_positions, user_positions, cloud_position)
        
        # Get UAV parameters from first UAV (assuming all UAVs have same parameters)
        uav = uavs[0]
        uav_params = {
            'P_u': uav.P_u,
            'P_v_w': uav.P_v_w,
            'r': uav.r,
            'sigma2': uav.sigma2,
            'N': uav.N,
            'B': uav.B,
            'B_cloud': uav.B_cloud
        }
        
        # Calculate transmission rates in batch
        user_to_uav_rates, uav_to_cloud_rates = VectorizedOps.calculate_transmission_rates_batch(
            distances_uav_user, distances_uav_cloud, uav_params)
        
        # Process each task
        task_assignments = {}
        for user_idx, user in enumerate(users):
            for task_idx, task in enumerate(user.tasks):
                if task[-1]:  # Skip completed tasks
                    continue
                
                # Calculate processing time for each UAV
                processing_times = np.zeros(len(uavs))
                for uav_idx, uav in enumerate(uavs):
                    processing_times[uav_idx] = VectorizedOps._calculate_single_task_time(
                        task, uav, 
                        user_to_uav_rates[uav_idx, user_idx],
                        uav_to_cloud_rates[uav_idx]
                    )
                
                # Find best UAV
                best_uav_idx = np.argmin(processing_times)
                min_time = processing_times[best_uav_idx]
                task_assignments[(user_idx, task_idx)] = (best_uav_idx, min_time)
        
        return task_assignments
    
    @staticmethod
    def _calculate_single_task_time(task, uav, uav_user_rate, uav_cloud_rate):
        """Calculate processing time for a single task on a UAV."""
        total_time = 0
        
        if task[0] in uav.cached_content_type:
            total_time = 0
        elif task[0] in uav.cached_service_type:
            computing_capacity = np.random.randint(3, 8) * uav.F_cpu / 30.0
            total_time = task[1] / computing_capacity
        else:
            uav_cloud_time = task[1] / uav_cloud_rate
            total_time = 2 * uav_cloud_time
        
        return (
            total_time + 
            task[1] / uav_user_rate +
            (task[1] * 0.6) / uav.U_V_transmission_rate
        )
    
    @staticmethod
    def calculate_profits_batch(tasks_info, uavs_info, task_times):
        """
        Calculate profits for tasks in a vectorized manner.
        
        Args:
            tasks_info (np.ndarray): Task information array
            uavs_info (np.ndarray): UAV information array
            task_times (np.ndarray): Task processing times
            
        Returns:
            np.ndarray: Profits for each task
        """
        # This is a simplified version - actual implementation would depend on
        # the specific profit calculation formula
        pass
    
    @staticmethod
    def move_uavs_batch(uav_positions, actions, move_distance=10):
        """
        Move all UAVs in a vectorized manner.
        
        Args:
            uav_positions (np.ndarray): UAV positions array of shape (n_uavs, 3)
            actions (np.ndarray): Actions for each UAV of shape (n_uavs,)
            move_distance (float): Distance to move
            
        Returns:
            np.ndarray: Updated UAV positions
        """
        # Map actions from [-1,1] to [0, 2π] radians
        move_directions = np.pi * actions + np.pi
        
        # Calculate position changes
        dx = move_distance * np.cos(move_directions)
        dy = move_distance * np.sin(move_directions)
        
        # Update positions
        new_positions = uav_positions.copy()
        new_positions[:, 0] += dx
        new_positions[:, 1] += dy
        
        return new_positions
    
    @staticmethod
    def enforce_min_distance_batch(uav_positions, min_distance=400, max_range=20000):
        """
        Enforce minimum distance constraint between UAVs in a vectorized manner.
        
        Args:
            uav_positions (np.ndarray): UAV positions array of shape (n_uavs, 3)
            min_distance (float): Minimum distance between UAVs
            max_range (float): Maximum flight range
            
        Returns:
            np.ndarray: Adjusted UAV positions
        """
        n_uavs = uav_positions.shape[0]
        positions = uav_positions.copy()
        
        # Calculate pairwise distances
        for i in range(n_uavs):
            for j in range(i + 1, n_uavs):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < min_distance:
                    # Adjust positions to maintain minimum distance
                    overlap = min_distance - distance
                    positions[i, 0] += overlap * dx / 2
                    positions[i, 1] += overlap * dy / 2
                    positions[j, 0] -= overlap * dx / 2
                    positions[j, 1] -= overlap * dy / 2
        
        # Ensure UAVs stay within flight range
        positions[:, 0] = np.clip(positions[:, 0], -max_range, max_range)
        positions[:, 1] = np.clip(positions[:, 1], -max_range, max_range)
        
        return positions
    
    @staticmethod
    def calculate_uav_rewards_batch(uav_profits, uav_base_costs, uav_tasked_numbers, task_num_sills):
        """
        Calculate rewards for all UAVs in a vectorized manner.
        
        Args:
            uav_profits (np.ndarray): Profits for each UAV (n_uavs,)
            uav_base_costs (np.ndarray): Base costs for each UAV (n_uavs,)
            uav_tasked_numbers (np.ndarray): Number of tasks processed by each UAV (n_uavs,)
            task_num_sills (np.ndarray): Task number thresholds for each UAV (n_uavs,)
            
        Returns:
            tuple: (rewards, updated_base_costs, updated_sills)
        """
        # Calculate basic rewards
        rewards = uav_profits - uav_base_costs
        
        # Calculate task excess
        task_excess = uav_tasked_numbers - task_num_sills
        task_excess = np.maximum(task_excess, 0)
        
        # Update base costs
        updated_base_costs = uav_base_costs + task_excess * 3
        updated_sills = task_num_sills + task_excess
        
        return rewards, updated_base_costs, updated_sills

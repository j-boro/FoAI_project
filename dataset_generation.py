import numpy as np
import multiprocessing
import os
import time
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def generate_chunk(args):
    """
    Worker function to generate a subset of data.
    """
    num_samples, grid_size, obstacle_density, min_dist, seed_offset = args
    
    np.random.seed(int(time.time()) + os.getpid() + seed_offset)
    
    inputs = np.zeros((num_samples, grid_size, grid_size, 3), dtype='float32')
    targets = np.zeros((num_samples, grid_size, grid_size, 1), dtype='float32')
    
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    
    count = 0
    while count < num_samples:
        map_obstacles = (np.random.rand(grid_size, grid_size) < obstacle_density).astype(int)
        
        walkable_matrix = 1 - map_obstacles
        grid_obj = Grid(matrix=walkable_matrix)
        
        free_cells = np.argwhere(walkable_matrix == 1)
        if len(free_cells) < 2: continue 
            
        found_pair = False
        for _ in range(10): 
            idx_start = np.random.randint(len(free_cells))
            idx_end = np.random.randint(len(free_cells))
            
            start_y, start_x = free_cells[idx_start]
            end_y, end_x = free_cells[idx_end]
            
            dist = abs(start_x - end_x) + abs(start_y - end_y)
            if dist >= min_dist:
                found_pair = True
                break
        
        if not found_pair: continue
        
        start_node = grid_obj.node(start_x, start_y)
        end_node = grid_obj.node(end_x, end_y)
        
        path, runs = finder.find_path(start_node, end_node, grid_obj)
        
        if len(path) > 0:
            inputs[count, :, :, 0] = map_obstacles
            inputs[count, start_y, start_x, 1] = 1
            inputs[count, end_y, end_x, 2] = 1
            
            for step in path:
                targets[count, step.y, step.x, 0] = 1
                
            count += 1

    return inputs, targets

def generate_data(num_samples, grid_size=32, obstacle_density=0.4, min_dist=15):
    """
    Multiprocessed data generator with robust seeding.
    """
    num_processes = multiprocessing.cpu_count()
    samples_per_process = num_samples // num_processes
    remainder = num_samples % num_processes
    
    tasks = []
    for i in range(num_processes):
        count = samples_per_process + (1 if i < remainder else 0)
        if count > 0:
            tasks.append((count, grid_size, obstacle_density, min_dist, i * 1000))
            
    print(f"Generating {num_samples} samples using {num_processes} threads (Density: {obstacle_density})...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_chunk, tasks)
        
    all_inputs = np.concatenate([res[0] for res in results], axis=0)
    all_targets = np.concatenate([res[1] for res in results], axis=0)
    
    return all_inputs, all_targets
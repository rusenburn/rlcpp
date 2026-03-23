import numpy as np

def generate_symmetries():
    # Create a 8x8 grid of indices [0, 1, ..., 63]
    base_grid = np.arange(64).reshape(8, 8)
    
    # Define the 7 transformations (excluding identity)
    # Note: 'rot90' k=1 is 90 deg, k=2 is 180 deg, k=3 is 270 deg
    transforms = {
        "ROT90": lambda g: np.rot90(g, k=1),
        "ROT180": lambda g: np.rot90(g, k=2),
        "ROT270": lambda g: np.rot90(g, k=3),
        "FLIP_LR": lambda g: np.fliplr(g),  # Vertical axis flip
        "FLIP_UD": lambda g: np.flipud(g),  # Horizontal axis flip
        "TRANSPOSE": lambda g: g.T,         # Main diagonal
        "ANTI_TRANSPOSE": lambda g: np.transpose(np.flip(g))
    }

    def get_full_obs_mapping(grid_transform_func):
        # Apply transformation to the 8x8 index grid
        transformed_grid = grid_transform_func(base_grid)
        flat_grid = transformed_grid.flatten()
        
        # Tile for 4 channels, adding 64 for each subsequent channel
        full_obs = []
        for c in range(4):
            full_obs.extend((flat_grid + (c * 64)).tolist())
        return flat_grid.tolist(), full_obs

    for name, func in transforms.items():
        actions_sym, obs_sym = get_full_obs_mapping(func)
        
        print(f"// --- {name} ---")
        print(f"constexpr std::array<int, 64> {name}_ACTIONS_SYM = {{ {{")
        # print(f"constexpr std::array<int, 64> {name}_ACTIONS_SYM = {{{{")
        print(', '.join(map(str, actions_sym)))
        print("} };\n")
        
        print(f"constexpr std::array<int, 256> {name}_OBS_SYM = {{ {{")
        # Print in chunks of 8 for readability
        for i in range(0, len(obs_sym), 8):
            print('  ' + ', '.join(map(str, obs_sym[i:i+8])) + (',' if i+8 < 256 else ''))
        print("} };\n")

if __name__ == "__main__":
    generate_symmetries()
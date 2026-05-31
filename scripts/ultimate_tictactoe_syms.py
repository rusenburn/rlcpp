import numpy as np

def generate_symmetries():
    BOARDS = 9
    ROWS = 3
    COLS = 3
    N_ACTIONS = BOARDS * ROWS * COLS  # 81
    CHANNELS = 30
    
    # Overall 9x9 grid position transforms
    def overall_rot90(r, c):
        return c, 8 - r
    
    def overall_rot180(r, c):
        return 8 - r, 8 - c
    
    def overall_rot270(r, c):
        return 8 - c, r
    
    def overall_flip_lr(r, c):
        return r, 8 - c
    
    def overall_flip_ud(r, c):
        return 8 - r, c
    
    def overall_transpose(r, c):
        return c, r
    
    def overall_anti_transpose(r, c):
        return 8 - c, 8 - r

    # 3x3 grid transforms (meta-level only) - same pattern as 9x9 but size 3
    def meta_rot90(r, c):
        return c, 2 - r
    
    def meta_rot180(r, c):
        return 2 - r, 2 - c
    
    def meta_rot270(r, c):
        return 2 - c, r
    
    def meta_flip_lr(r, c):
        return r, 2 - c
    
    def meta_flip_ud(r, c):
        return 2 - r, c
    
    def meta_transpose(r, c):
        return c, r
    
    def meta_anti_transpose(r, c):
        return 2 - c, 2 - r

    overall_funcs = {
        "ROT90": overall_rot90,
        "ROT180": overall_rot180,
        "ROT270": overall_rot270,
        "FLIP_LR": overall_flip_lr,
        "FLIP_UD": overall_flip_ud,
        "TRANSPOSE": overall_transpose,
        "ANTI_TRANSPOSE": overall_anti_transpose,
    }
    
    meta_funcs = {
        "ROT90": meta_rot90,
        "ROT180": meta_rot180,
        "ROT270": meta_rot270,
        "FLIP_LR": meta_flip_lr,
        "FLIP_UD": meta_flip_ud,
        "TRANSPOSE": meta_transpose,
        "ANTI_TRANSPOSE": meta_anti_transpose,
    }
    
    for name in overall_funcs:
        overall_f = overall_funcs[name]
        meta_f = meta_funcs[name]
        
        # --- ACTION MAPPING (81 elements) ---
        # Action index = board * 9 + row * 3 + col
        # Overall position: overall_r = (board//3)*3 + row, overall_c = (board%3)*3 + col
        # After transform: (new_overall_r, new_overall_c) = transform(overall_r, overall_c)
        # new_board = (new_overall_r//3)*3 + (new_overall_c//3)
        # new_cell = (new_overall_r%3, new_overall_c%3)
        actions_sym = np.zeros(N_ACTIONS, dtype=int)
        for b in range(BOARDS):
            meta_r = b // 3
            meta_c = b % 3
            for cell_r in range(ROWS):
                for cell_c in range(COLS):
                    action_idx = b * 9 + cell_r * 3 + cell_c
                    overall_r = meta_r * 3 + cell_r
                    overall_c = meta_c * 3 + cell_c
                    new_overall_r, new_overall_c = overall_f(overall_r, overall_c)
                    new_meta_r = new_overall_r // 3
                    new_meta_c = new_overall_c // 3
                    new_cell_r = new_overall_r % 3
                    new_cell_c = new_overall_c % 3
                    new_b = new_meta_r * 3 + new_meta_c
                    new_action_idx = new_b * 9 + new_cell_r * 3 + new_cell_c
                    actions_sym[action_idx] = new_action_idx
        
        # --- OBSERVATION MAPPING (270 elements) ---
        # Observation index = channel * 9 + row * 3 + col
        obs_sym = np.zeros(CHANNELS * ROWS * COLS, dtype=int)
        
        for ch in range(CHANNELS):
            for r in range(ROWS):
                for c in range(COLS):
                    obs_idx = ch * 9 + r * 3 + c
                    
                    if ch < 9:
                        # Channels 0-8: current player's small boards
                        # board_no = ch, so (r,c) are cell coords within that board
                        # board meta-position = (ch//3, ch%3)
                        # overall position: overall_r = (ch//3)*3 + r, overall_c = (ch%3)*3 + c
                        meta_r = ch // 3
                        meta_c = ch % 3
                        overall_r = meta_r * 3 + r
                        overall_c = meta_c * 3 + c
                        new_overall_r, new_overall_c = overall_f(overall_r, overall_c)
                        new_meta_r = new_overall_r // 3
                        new_meta_c = new_overall_c // 3
                        new_cell_r = new_overall_r % 3
                        new_cell_c = new_overall_c % 3
                        new_ch = new_meta_r * 3 + new_meta_c
                        
                    elif ch == 9:
                        # Channel 9: current player's ultimate board
                        # (r,c) is meta-position of a small board on the ultimate board
                        # A cell at (r,c) in the ultimate board represents the status of small board at (r,c)
                        # After the overall transform, the board at (r,c) moves to meta_f(r,c)
                        # So the cell (r,c) should map to meta_f(r,c) in channel 9
                        new_cell_r, new_cell_c = meta_f(r, c)
                        new_ch = 9
                        
                    elif ch < 19:
                        # Channels 10-18: opponent's small boards
                        board_no = ch - 10
                        meta_r = board_no // 3
                        meta_c = board_no % 3
                        overall_r = meta_r * 3 + r
                        overall_c = meta_c * 3 + c
                        new_overall_r, new_overall_c = overall_f(overall_r, overall_c)
                        new_meta_r = new_overall_r // 3
                        new_meta_c = new_overall_c // 3
                        new_cell_r = new_overall_r % 3
                        new_cell_c = new_overall_c % 3
                        new_ch = new_meta_r * 3 + new_meta_c + 10
                        
                    elif ch == 19:
                        # Channel 19: opponent's ultimate board
                        new_cell_r, new_cell_c = meta_f(r, c)
                        new_ch = 19
                        
                    elif ch < 29:
                        # Channels 20-28: legal action masks
                        board_no = ch - 20
                        meta_r = board_no // 3
                        meta_c = board_no % 3
                        overall_r = meta_r * 3 + r
                        overall_c = meta_c * 3 + c
                        new_overall_r, new_overall_c = overall_f(overall_r, overall_c)
                        new_meta_r = new_overall_r // 3
                        new_meta_c = new_overall_c // 3
                        new_cell_r = new_overall_r % 3
                        new_cell_c = new_overall_c % 3
                        new_ch = new_meta_r * 3 + new_meta_c + 20
                        
                    else:
                        # Channel 29: all ones - identity mapping
                        new_cell_r = r
                        new_cell_c = c
                        new_ch = 29
                    
                    new_obs_idx = new_ch * 9 + new_cell_r * 3 + new_cell_c
                    obs_sym[obs_idx] = new_obs_idx
        
        # Validate: check that the mapping is a permutation (bijection)
        assert len(set(actions_sym)) == N_ACTIONS, f"{name} actions_sym is not a permutation"
        assert len(set(obs_sym)) == CHANNELS * ROWS * COLS, f"{name} obs_sym is not a permutation"
        
        print(f"// --- {name} ---")
        print(f"constexpr std::array<int, 81> {name}_ACTIONS_SYM = {{ {{")
        for i in range(0, len(actions_sym), 9):
            chunk = actions_sym[i:i+9]
            line = ', '.join(str(x) for x in chunk)
            if i + 9 < len(actions_sym):
                print(f"  {line},")
            else:
                print(f"  {line}")
        print("} };\n")

        print(f"constexpr std::array<int, 270> {name}_OBS_SYM = {{ {{")
        for i in range(0, len(obs_sym), 9):
            chunk = obs_sym[i:i+9]
            line = ', '.join(str(x) for x in chunk)
            if i + 9 < len(obs_sym):
                print(f"  {line},")
            else:
                print(f"  {line}")
        print("} };\n")

if __name__ == "__main__":
    generate_symmetries()
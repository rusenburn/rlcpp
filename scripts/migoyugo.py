import re

def get_action_id(coord):
    col_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    column = col_map[coord[0].upper()]
    row = 8 - int(coord[1])
    return str((row * 8) + column)

print("Paste your move text below. Type 'DONE' on a new line or press Enter twice to process:")

lines = []
while True:
    line = input()
    if line.upper() == "DONE" or (not line and lines and lines[-1] == ""):
        break
    lines.append(line)

move_text = "\n".join(lines)

# Extract coordinates (e.g., d3, e3, c8)
raw_coords = re.findall(r'[a-h][1-8]', move_text.lower())
action_ids = [get_action_id(c) for c in raw_coords]

print("\n--- Action IDs ---")
print(" ".join(action_ids) + " -1") # Added -1 to trigger the C++ finish condition
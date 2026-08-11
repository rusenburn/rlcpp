def action_id_to_coord(action_id):
    col_map = ["A", "B", "C", "D", "E", "F", "G", "H"]
    row = 8 - (action_id // 8)
    column = col_map[action_id % 8]
    return f"{column}{row}"


print(
    "Paste your action IDs below (space or newline separated). Type 'DONE' or end with -1:"
)

input_text = ""
while True:
    try:
        line = input()
        if line.strip().upper() == "DONE":
            break
        input_text += " " + line
        if "-1" in line.split():
            break
    except EOFError:
        break

# Extract all valid integers up to -1
raw_tokens = input_text.split()
action_ids = []
for token in raw_tokens:
    if token == "-1":
        break
    if token.isdigit():
        val = int(token)
        if 0 <= val <= 63:
            action_ids.append(val)

# Convert IDs to coordinates
coords = [action_id_to_coord(aid) for aid in action_ids]

# Output formatted move list
print("\n--- Moves List ---")
for i in range(0, len(coords), 2):
    move_num = (i // 2) + 1
    move1 = coords[i]
    move2 = coords[i + 1] if i + 1 < len(coords) else ""

    # Formats with aligned spacing
    print(f"{move_num:2d}. {move1:<8}{move2}")

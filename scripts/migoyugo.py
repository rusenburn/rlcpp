import re

# Your input data
move_text = """
1. c3 f6
2. c4 e4
3. c2 c5
4. c1 (1 yugo) e5
5. d4 e6
6. d2 d5
7. d3 d6
8. d1 (1 yugo) c6 (1 yugo)
9. e1 b1
10. f1 (1 yugo) e1
11. d3 f3 (1 yugo)
12. c4 d5
13. e4 d6
14. d2 d4
15. e2 (1 yugo) c4
16. d3 c3 (1 yugo)
17. c4 (1 yugo) d3 (1 yugo)
18. e3 d4
19. b2 f6 (1 yugo)
20. c2 (1 yugo) d4
21. a4 e5 (1 yugo)
22. d4 f5
23. b3 (1 yugo) a4
24. d6 c5
25. d2 g6
26. b2 (1 yugo) d2
27. d5 h6
28. e6 (1 yugo) d5 (1 yugo)
29. f5 b5
30. c5 g7
31. h8 g5
32. g4 d8
33. e7 a8
34. b7 b8
35. c8 g8 (1 yugo)
36. g2 f8
37. e8 g5
38. g3 g1
39. f7 g7
40. c7 a2
41. h5 h2
42. g6 (1 yugo) e8 (1 yugo)
43. d8 f8
44. h4 f7
45. h1 h3
46. h5 a6
47. a1 a3
48. a7 h7
49. a5 b6
50. b4 (2 yugos) e4 (1 yugo)
"""

def get_action_id(coord):
    """Converts coordinate (e.g., 'A8') to Action ID (0-63)."""
    col_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    
    column = col_map[coord[0].upper()]
    # Calculation: (8 - Rank) * 8 + Column
    row = 8 - int(coord[1])
    
    return str((row * 8) + column)

# 1. Extract only the board coordinates using regex
raw_coords = re.findall(r'[A-H][1-8]', move_text.upper())

# 2. Convert coordinates to Action IDs (as strings for joining)
action_ids = [get_action_id(c) for c in raw_coords]

# 3. Join them with a single space
output_string = " ".join(action_ids)

print(output_string)
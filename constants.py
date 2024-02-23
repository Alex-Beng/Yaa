# 储存常量

DT = 0.05

# state name -> dim
STATE_DIM = 19

SN = {
    'W': 0,
    'A': 1,
    'S': 2,
    'D': 3,
    'Q': 4,
    'E': 5,
    'X': 6,
    ' ': 7,
    'LS': 8,
    'T': 9,
    'Z': 10,
    '1': 11,
    '2': 12,
    '3': 13,
    '4': 14,
    'ML': 15,
    'MRo': 16, 
    'Mdx': 17,
    'Mdy': 18
}

SN_idx2key = [
    'W', 
    'A', 
    'S', 
    'D', 
    'Q', 
    'E', 
    'X', 
    ' ', 
    'LS',
    'T', 
    'Z', 
    '1', 
    '2', 
    '3', 
    '4', 
    'ML', 
    'MRo', 
    'Mdx', 
    'Mdy'
]

# SCANCODE of KOI(key of interest)
SC_idx2sc = [
    0x11, # W
    0x1E, # A
    0x1F, # S
    0x20, # D
    0x10, # Q
    0x12, # E
    0x2D, # X
    0x39, # Space
    0x1D, # LShift
    0x14, # T
    0x2C, # Z
    0x02, # 1
    0x03, # 2
    0x04, # 3
    0x05, # 4
    # for mouse, there is no scancode
]

SC_sc2idx = {
    0x11: 0, # W
    0x1E: 1, # A
    0x1F: 2, # S
    0x20: 3, # D
    0x10: 4, # Q
    0x12: 5, # E
    0x2D: 6, # X
    0x39: 7, # Space
    0x1D: 8, # LShift
    0x14: 9, # T
    0x2C: 10, # Z
    0x02: 11, # 1
    0x03: 12, # 2
    0x04: 13, # 3
    0x05: 14, # 4
    # for mouse, there is no scancode
}


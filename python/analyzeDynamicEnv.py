import numpy as np

filename = 'assignment.txt'

def parse_assignment(filename):
    with open(filename, 'r') as f:
        assignment = f.readlines()

    parsed_assignments = []
    for outcome in assignment:
        parsed_outcome = outcome.rstrip().split('|')[:-1]
        parsed_assignments.append(parsed_outcome)
    parsed_assignments = [[group.split(' ')[:-1] for group in outcome] for outcome in parsed_assignments]
    return parsed_assignments


parsed_assignments = parse_assignment(filename)

# number of players
n = sum([len(room_group) for room_group in parsed_assignments[0]])
# number of rooms
m = len(parsed_assignments[0])

print('n', n, 'm', m)

count = [[0] * m for _ in range(n)]
# freq that agent i in any room j
for outcome in parsed_assignments:
    for room_num, room in enumerate(outcome):
        for player in room:
            count[int(player)][room_num] += 1


print('count:', count)

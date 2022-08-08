import pickle
from LK3_classification import Line

def read_object(filename):
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        lines = pickle.load(inp)
    return lines

lines = read_object('./line_segments.pkl')
print(len(lines))

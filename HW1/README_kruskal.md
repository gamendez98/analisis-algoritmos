# Kruskal's Algorithm Script

This Python script `kruskal.py` implements Kruskal's algorithm to find the Minimum Spanning Tree (MST) of a graph. This corresponds to point 5 of the homework. The graph is provided as input in a `.in` file, and the resulting MST is written to a `.out` file.

## Input File Format
The input file (e.g., `graph.in`) should contain one edge per line in the format:
```
<node1> <node2> <weight>
```
For example:
```
A B 4
A D 1
A E 5
B G 1
D B 2
D E 3
D F 8
E F 4
G F 2
```

## Output File Format
The output file (e.g., `mst.out`) contains the edges of the MST, one edge per line in the format:
```
<node1> <node2> <weight>
```
For example:
```
A D 1
B G 1
D B 2
G F 2
D E 3
```

## How to Run
1. Create an input file named `graph.in` in the same directory as the script with the graph edges.
2. Open a terminal or command prompt and navigate to the script's directory.
3. Run the script using the following command:
   ```
   python kruskal.py graph.in output_name.out
   ```
   You are free to choose an output name.
4. The MST will be written to `output_name.out` in the same directory.

## Example
### Input (`graph.in`):
```
A B 4
A D 1
A E 5
B G 1
D B 2
D E 3
D F 8
E F 4
G F 2
```

### Command:
```
python kruskal.py graph.in mst.out
```

### Output (`mst.out`):
```
A D 1
B G 1
D B 2
G F 2
D E 3
```

## Requirements
- Python 3.6 or later

## Notes
- Ensure the input file follows the format strictly.
- Negative weights are not allowed.
- Disconnected graphs will raise a Value Error, since no MST is possible.
- The output file will overwrite any existing file in the same directory.
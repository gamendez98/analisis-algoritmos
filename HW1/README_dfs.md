# DFS Algorithm Script

This Python script `dfs.py` implements Kruskal's algorithm to find the cycles of a graph. This corresponds to point 2 of the homework. The graph is provided as input in a `.in` file, and the resulting list of cycles is written to a `.out` file.

## Input File Format
The input file (e.g., `graph_dfs.in`) should contain one vertex (bank) per line in the format:
```
<node1>,<node2>,<weight>
```
For example:
```
A,B,4
B,D,5
D,E,7
E,A,1
A,D,1
A,E,5
B,G,1
D,B,2
D,E,3
D,F,8
E,F,4
G,F,2
A,A,99
```

## Output File Format
The output file (e.g., `dfs.out`) contains the cycles in the graph, in the format:
```
<node1> -> <node2> -> (...) -> <node_n>
```
For example:
```
A
B -> D
A -> B -> D -> E
```
If no self-loans (cycles) were found, then the  output file has the following content:
```
No self-loans were detected
```

## How to Run
1. Create an input file named `graph_dfs.in` in the same directory as the script with the graph edges.
2. Open a terminal or command prompt and navigate to the script's directory.
3. Run the script using the following command:
   ```
   python dfs.py graph.in output_name.out
   ```
   You are free to choose an output name.
4. The cycles will be written to `output_name.out` in the same directory.

## Example
### Input (`graph_dfs.in`):
```
A,B,4
B,D,5
D,E,7
E,A,1
A,D,1
A,E,5
B,G,1
D,B,2
D,E,3
D,F,8
E,F,4
G,F,2
A,A,99
```

### Command:
```
python dfs.py graph.in cycles.out
```

### Output (`cycles.out`):
```
A
B -> D
A -> B -> D -> E
```

## Requirements
- Python 3.6 or later

## Notes
- Ensure the input file follows the format strictly.
- Negative weights are not allowed, as there are no negative loans.
- Disconnected graphs will raise a Value Error, since no MST is possible. CHECK for DTS
- The output file will overwrite any existing file in the same directory.
# Maximum Flow Algorithms Script

This Python script `maximum_flow.py` implements two **maximum flow algorithms**:
- **Edmonds-Karp Algorithm**: A breadth-first search (BFS) implementation of the Ford-Fulkerson method.
- **Push-Relabel Algorithm**: A more advanced algorithm that efficiently pushes excess flow to compute the maximum flow.

The graph is provided as input in a `.in` file, and the computed flow results are written to a `.out` file.

## Team Members

- **David Alejandro Fuquen Florez**
- **Isabella Martinez Martinez**
- **Gustavo Andrés Mendez Hernández**

## Input File Format
The input file (e.g., `network.in`) should contain the number of nodes in the first line, followed by one edge per line in the format:
```
<source_node> <target_node> <capacity>
```
For example:
```
4
0 1 10
0 2 5
1 2 15
1 3 10
2 3 10
```

## Output File Format
The output file (e.g., `max_flow.out`) contains the flow assigned to each edge, the total maximum flow, and execution times for both algorithms:
```
Algorithm Results:

Edmonds-Karp Algorithm
Edge Flows:
<source_node> <target_node> <flow>
...

Total Flow: <max_flow_value>
Execution Time: <time_in_seconds>

Push-Relabel Algorithm
Edge Flows:
<source_node> <target_node> <flow>
...

Maximum Flow: <max_flow_value>
Execution Time: <time_in_seconds>
```
For example:
```
Algorithm Results:

Edmonds-Karp Algorithm
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Total Flow: 15
Execution Time: 0.000123 seconds

Push-Relabel Algorithm
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Maximum Flow: 15
Execution Time: 0.000095 seconds
```

## How to Run
1. Create an input file named `network.in` with the graph definition.
2. Open a terminal or command prompt and navigate to the script's directory.
3. Run the script using the following command:
   ```
   python maximum_flow.py input_file.in output_file.out
   ```
   Replace `input_file.in` with the name of your input file and `output_file.out` with the desired output file name.
4. The computed maximum flow for both algorithms will be written to the specified output file.

## Example
### Input (`network.in`):
```
4
0 1 10
0 2 5
1 2 15
1 3 10
2 3 10
```

### Command:
```
python maximum_flow.py network.in max_flow.out
```

### Output (`max_flow.out`):
```
Algorithm Results:

Edmonds-Karp Algorithm
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Total Flow: 15
Execution Time: 0.000123 seconds

Push-Relabel Algorithm
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Maximum Flow: 15
Execution Time: 0.000095 seconds
```

## Requirements
- Python 3.6 or later

## Notes
- Ensure the input file follows the format strictly.
- Negative edge capacities are not allowed; the script will raise a `ValueError` if they are encountered.
- The graph must be directed. It is assumed that the first node (zero) is the source and the last (N-1) is the destination.
- The output file will overwrite any existing file in the same directory.
- The script runs **both algorithms** and provides a comparison of their execution times.
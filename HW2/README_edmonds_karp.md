# Edmonds-Karp Algorithm Script

This Python script `edmonds_karp.py` implements the **Edmonds-Karp algorithm**, an efficient implementation of the **Ford-Fulkerson method** to compute the **maximum flow** in a directed graph. The graph is provided as input in a `.in` file, and the resulting flow network is written to a `.out` file.

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

The first line must have the number of nodes in the network N. It is assumed that the first node (zero) is the source and the last (N-1) is the destination.

## Output File Format
The output file (e.g., `max_flow.out`) contains the flow assigned to each edge, followed by the total maximum flow and execution time:
```
Edge Flows:
<source_node> <target_node> <flow>
...

Total Flow: <max_flow_value>
Execution Time: <time_in_seconds>
```
For example:
```
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Total Flow: 15
Execution Time: 0.000123 seconds
```

## How to Run
1. Create an input file named `network.in` with the graph definition.
2. Open a terminal or command prompt and navigate to the script's directory.
3. Run the script using the following command:
   ```
   python edmonds_karp.py input_file.in output_file.out
   ```
   Replace `input_file.in` with the name of your input file and `output_file.out` with the desired output file name.
4. The computed maximum flow and edge flows will be written to the specified output file.

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
python edmonds_karp.py network.in max_flow.out
```

### Output (`max_flow.out`):
```
Edge Flows:
0 1 10
0 2 5
1 2 5
1 3 10
2 3 10

Total Flow: 15
Execution Time: 0.000123 seconds
```

## Requirements
- Python 3.6 or later

## Notes
- Ensure the input file follows the format strictly.
- Negative edge capacities are not allowed; the script will raise a `ValueError` if they are encountered.
- The output file will overwrite any existing file in the same directory.
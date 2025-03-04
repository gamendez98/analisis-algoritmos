# Suffix Array and Query Search

This project implements a suffix array to efficiently search for substrings in a given text. It includes binary search functionality for fast query lookups and tools to generate test data.

## Team Members

- **David Alejandro Fuquen Florez**
- **Isabella Martinez Martinez**
- **Gustavo Andrés Mendez Hernández**

## Requirements

Ensure you have Python 3 installed

## Scripts

### 1. `suffixes_no_prints.py`

Builds a suffix array for a given text file and performs substring searches using binary search.

#### **Inputs**
- `filepath` (str, required): Path to the input text file.
- `query` (str, required): Path to the file containing query strings (one per line).
- `output` (str, required): Path to the output file.

#### **Outputs**
- A file containing the queries and their positions in the input text, separated by tabs.
- Execution time metrics for suffix array construction and query search.

#### **How to Run**
```bash
python suffixes_no_prints.py <filepath> <query_file> <output_file>
```
Example:
```bash
python suffixes_no_prints.py text.txt queries.txt results.txt
```

---

### 2. `query_generator.py`

Generates a specified number of random query strings from a given text file.

#### **Inputs**
- `filepath` (str, required): Path to the input text file.
- `amount` (int, required): Number of queries to generate.

#### **Outputs**
- A file containing randomly extracted query strings.

#### **How to Run**
```bash
python query_generator.py <filepath> <amount>
```
Example:
```bash
python query_generator.py text.txt 1000
```

---

### 3. `generate_sequence.py`

Generates a random sequence of characters of a specified length to use as input text.

#### **Inputs**
- `length` (int, required): Length of the sequence to generate.
- `output` (str, required): Path to the output file.

#### **Outputs**
- A text file containing a randomly generated sequence.

#### **How to Run**
```bash
python generate_sequence.py <length> <output_file>
```
Example:
```bash
python generate_sequence.py 1000000 text.txt
```

## Experimentation




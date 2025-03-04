import sys
from typing import List, Tuple
import argparse
import time

'''
def build_suffix_array(text: str) -> List[int]:
    """
    Build a suffix array for the given text.
    
    It doesn't explicitly store all suffixes.
    Instead, it sorts the indices based on the suffixes they represent.
    
    Args:
        text: The input text
        
    Returns:
        A sorted array of indices representing the suffix array
    """
    # Create a list of tuples (index, suffix)
    # Store the index and compute the suffix on-the-fly during comparison
    text = "".join([text, "$"])
    suffixes = [(i, text[i:]) for i in range(len(text))]

    # Sort based on lexicographical order of suffixes
    suffixes.sort(key=lambda x: x[1])
    
    # Extract just the indices to form the suffix array
    suffix_array = [index for index, _ in suffixes]
    
    return suffix_array
'''

def build_suffix_array(text: str) -> List[int]:
    """
    Build a suffix array for the given text using the Doubling Algorithm.
    
    Args:
        text: The input text.
        
    Returns:
        A sorted array of indices representing the suffix array.
    """
    n = len(text)
    suffix_array = list(range(n))
    
    # Initialize rank array
    rank = [ord(c) for c in text] + [-1]  # Adding -1 for the sentinel character
    tmp_rank = [0] * (n + 1)  # Temporary array to store ranks
    
    # Function to compare two suffixes based on the first 2^k characters
    def suffix_compare(i, j, k):
        if rank[i] != rank[j]:
            return rank[i] - rank[j]
        next_i = i + (1 << k)
        next_j = j + (1 << k)
        if next_i < n and next_j < n:
            return rank[next_i] - rank[next_j]
        return 0  # For suffixes that are of unequal lengths

    # Sorting by doubling the length of the prefix considered
    k = 0
    while (1 << k) < n:
        # Sort suffixes based on the first 2^k characters
        suffix_array.sort(key=lambda i: (rank[i], rank[i + (1 << k)] if i + (1 << k) < n else -1))
        
        # Compute new ranks based on the sorted suffixes
        tmp_rank[suffix_array[0]] = 0
        for i in range(1, n):
            tmp_rank[suffix_array[i]] = tmp_rank[suffix_array[i - 1]]
            if suffix_compare(suffix_array[i - 1], suffix_array[i], k) != 0:
                tmp_rank[suffix_array[i]] += 1
        
        rank = tmp_rank[:]
        k += 1
    
    return suffix_array

def binary_search(text: str, suffix_array: List[int], query: str) -> List[int]:
    """
    Find all occurrences of query in text using binary search on the suffix array.
    
    Args:
        text: The input text
        suffix_array: The suffix array for the text
        query: The string to search for
        
    Returns:
        A list of positions where query is found
    """
    n = len(text)
    query_len = len(query)
    positions = []
    
    ##print(f"\n--- DEBUGGING BINARY SEARCH FOR QUERY: '{query}' ---")
    ##print(f"Text: '{text}'")
    ##print(f"Suffix Array: {suffix_array}")
    ##print(f"Suffixes in sorted order:")
    #for i, pos in enumerate(suffix_array):
    #    suffix = text[pos:min(pos + 15, len(text))]  # Show first 15 chars of suffix
    #    ##print(f"  {i}: [{pos}] '{suffix}{'...' if pos + 15 < len(text) else ''}'")
    
    # Binary search to find the first occurrence
    left, right = 0, n - 1
    first_occurrence = -1
    
    
    
    iteration = 1
    while left <= right:
        mid = (left + right) // 2
        suffix_start = suffix_array[mid]
        suffix = text[suffix_start:min(suffix_start + query_len, len(text))]
        
        if suffix < query:
            left = mid + 1
            #print(f"  New range: left={left}, right={right}")
        elif suffix > query:
            #print(f"  Suffix '{suffix}' > query '{query}', moving left")
            right = mid - 1
            #print(f"  New range: left={left}, right={right}")
        else:
            #print(f"  Match found! Suffix '{suffix}' == query '{query}'")
            #print(f"  Setting first_occurrence={mid} and continuing leftward")
            first_occurrence = mid
            right = mid - 1
            #print(f"  New range: left={left}, right={right}")
        
        iteration += 1
    
    # If no occurrence found, return empty list
    if first_occurrence == -1:
        #print("\nNo occurrences found.")
        return []
    
    #print(f"\nFirst occurrence found at index {first_occurrence} in suffix array (position {suffix_array[first_occurrence]} in text)")
    
    #print("\n--- FINDING LAST OCCURRENCE ---")
    
    # Binary search to find the last occurrence
    left, right = first_occurrence, n - 1
    last_occurrence = first_occurrence
    
    #print(f"Initial search range: left={left}, right={right}")
    
    iteration = 1
    while left <= right:
        mid = (left + right) // 2
        suffix_start = suffix_array[mid]
        suffix = text[suffix_start:min(suffix_start + query_len, len(text))]
        
        #print(f"\nIteration {iteration}:")
        #print(f"  mid={mid}, suffix_start={suffix_start}")
        #print(f"  Comparing query '{query}' with suffix '{suffix}'")
        
        if suffix < query:
            #print(f"  Suffix '{suffix}' < query '{query}', moving right")
            left = mid + 1
            #print(f"  New range: left={left}, right={right}")
        elif suffix > query:
            #print(f"  Suffix '{suffix}' > query '{query}', moving left")
            right = mid - 1
            #print(f"  New range: left={left}, right={right}")
        else:
            #print(f"  Match found! Suffix '{suffix}' == query '{query}'")
            #print(f"  Setting last_occurrence={mid} and continuing rightward")
            last_occurrence = mid
            left = mid + 1
            #print(f"  New range: left={left}, right={right}")
        
        iteration += 1

    #print(f"\nLast occurrence found at index {last_occurrence} in suffix array (position {suffix_array[last_occurrence]} in text)")

    # Collect all positions
    #print("\n--- COLLECTING ALL POSITIONS ---")
    #print(suffix_array[first_occurrence:last_occurrence + 1])
    for i in range(first_occurrence, last_occurrence + 1):
        #print(f"Adding position {suffix_array[i]} from suffix array index {i}")
        positions.append(suffix_array[i])
    
    #print(f"\nFinal result: Query '{query}' found at positions {positions} in the text")
    #print("query")
    return positions


def main():
    """Main function to handle file input, suffix array construction, and binary search."""
    parser = argparse.ArgumentParser(description="Suffix Array and Binary Search")
    parser.add_argument("filepath", type=str, help="Path to the input text file")
    parser.add_argument("query", type=str, help="Path to the file with the queries strings to search for")
    parser.add_argument("output", type=str, help="Path to the output file")
    args = parser.parse_args()

    # Read the input text file
    with open(args.filepath, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', '')[:10000]
    start_build = time.time()    
    suffix_array = build_suffix_array(text)
    print("--- %s seconds to build array---" % (time.time() - start_build))
    results = []

    start_search = time.time()
    with open(args.query, 'r', encoding='utf-8') as file:
        for line in file:
            positions = binary_search(text, suffix_array, line.replace('\n', ''))
            results.append((line.replace('\n', ''), positions))
    print("--- %s seconds to search---" % (time.time() - start_search))

    with open(args.output, 'w', encoding='utf-8') as file_out:
        for r in results: 
                file_out.write(str(r) + '\n')



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
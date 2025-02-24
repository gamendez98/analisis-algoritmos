"""
The following scripts compresses a file by applyinh the Huffman coding algorithm.
"""

import math
import argparse
from collections import Counter
from queue import PriorityQueue

class Node:
    """Node class for Huffman tree."""
    def __init__(self, symbol, probability):
        self.symbol = symbol
        self.probability = probability
        self.left = None
        self.right = None
    def show_node(self):
        return f"Node({repr(self.symbol)}, {self.probability})"   

def huffman_encoding(probabilities):
    characters_queue = PriorityQueue()
    count = 0
    for symbol, probability in probabilities.items():
        characters_queue.put((probability, count, Node(symbol, probability)))
        count += 1
    while characters_queue.qsize() > 1:
        left = characters_queue.get()
        right = characters_queue.get()
        new_node = Node("Inner node", left[0] + right[0])
        new_node.left = left[2]
        new_node.right = right[2]
        characters_queue.put((new_node.probability, count, new_node))
        count += 1
    root = characters_queue.get()[2]
    return root

def show_tree(root):
    def print_tree(node, indent="", last=True):
        if node is not None:
            # Print the current node with a branch marker.
            print(indent, end="")
            if last:
                print("└── ", end="")
                indent += "    "
            else:
                print("├── ", end="")
                indent += "│   "
            print(node.show_node())
            # Collect children (left and right)
            children = []
            if node.left:
                children.append(node.left)
            if node.right:
                children.append(node.right)
            # Recursively print each child.
            for i, child in enumerate(children):
                print_tree(child, indent, i == len(children) - 1)
    print_tree(root)


def main():
    """Main function to handle file input, encoding, compression, and decompression validation."""
    parser = argparse.ArgumentParser(description="Huffman Codes Text Compression")
    parser.add_argument("filepath", type=str, help="Path to the input text file")
    args = parser.parse_args()

    # Read the input text file
    with open(args.filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Compute symbol frequencies and probabilities
    frequency = Counter(text)
    total_chars = sum(frequency.values())
    probabilities = {char: count / total_chars for char, count in frequency.items()}
    root = huffman_encoding(probabilities)
    show_tree(root)

    
if __name__ == "__main__":
    main()
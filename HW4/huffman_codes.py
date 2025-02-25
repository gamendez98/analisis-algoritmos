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
        self.code = ''
        self.checked = False
    def show_node(self):
        return f"Node({repr(self.symbol)}, {self.probability}, {self.checked})"   


class HuffmanCode:
    """ Class for Huffman coding."""
    def __init__(self):
        self.code = {}
        self.reverse_code = {}  # Reverse lookup table for decoding
    def huffman_encoding(self, probabilities):
        characters_queue = PriorityQueue()
        count = 0
        for symbol, probability in probabilities.items():
            characters_queue.put((probability, count, Node(symbol, probability)))
            count += 1
        while characters_queue.qsize() > 1:
            left = characters_queue.get()
            right = characters_queue.get()
            if left[2].checked == True:
                print("Abort at ", left.show_node())
            if right[2].checked == True:
                print("Abort at ", right.show_node())    
            left[2].checked = True
            right[2].checked = True
            new_node = Node(f"Inner node {count}", left[0] + right[0])
            new_node.left = left[2]
            new_node.right = right[2]
            characters_queue.put((new_node.probability, count, new_node))
            count += 1
        root = characters_queue.get()[2]
        return root    
    def recursive_traverse(self, root):
        if len(root.symbol) == 1:
            self.code[root.symbol] = root.code
            self.reverse_code[root.code] = root.symbol 
        if root.left != None:
            root.left.code = root.code + '0'
            self.recursive_traverse(root.left)
        if root.right != None: 
            root.right.code = root.code + '1'
            self.recursive_traverse(root.right)

def calculate_worst_case_entropy(vocab_size):
    """Calculate the worst-case entropy given the vocabulary size."""
    return math.log2(vocab_size)
    
def compress_text(text, codes):
    """Compress the input text using Huffman codes."""
    return ''.join(codes[char] for char in text if char in codes)

def decompress_text(compressed_text, reverse_codes):
    """Decompress a Huffman code-encoded text back to its original form."""
    decoded_text = ""
    buffer = ""
    
    for bit in compressed_text:
        buffer += bit
        if buffer in reverse_codes:
            decoded_text += reverse_codes[buffer]
            buffer = ""
    
    return decoded_text

def expected_bits(probabilities, codes):
    expected_bits = 0
    for character, prob in probabilities.items():
        expected_bits += prob*len(codes[character])
    return expected_bits




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
    huffman_code = HuffmanCode()
    root = huffman_code.huffman_encoding(probabilities)
    #show_tree(root)
    huffman_code.recursive_traverse(root)
    codes = huffman_code.code
    reverse_codes = huffman_code.reverse_code
    compressed_text = compress_text(text, codes)
    decompressed_text = decompress_text(compressed_text, reverse_codes)

    worst_entropy = calculate_worst_case_entropy(len(probabilities))
    total_bits = len(compressed_text)
    expected_bits_per_symbol = expected_bits(probabilities, codes)

    print("Huffman Codes:")
    for char, code in codes.items():
        print(f"{repr(char)}: {code}")
    print(f"\nWorst-case Entropy: {worst_entropy:.4f} bits")
    print(f"Total Bits Required: {total_bits}")
    print(f"Expected Bits per Symbol: {expected_bits_per_symbol:.4f}")

    assert text == decompressed_text, "Decompressed text does not match the original!"
    print("\nDecompression successful! The original text was correctly reconstructed.")
    


    
if __name__ == "__main__":
    main()
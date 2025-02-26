import json
import struct
from collections import Counter
from queue import PriorityQueue

from networkx.classes.filters import show_nodes

BYTE_SIZE = 8


class Node:
    """Node class for Huffman tree."""

    def __init__(self, symbol, probability):
        self.symbol = symbol
        self.probability = probability
        self.left = None
        self.right = None
        self.code = ''
        self.checked = False

    def __repr__(self):
        return self.show_node()

    def show_node(self):
        return f"Node({repr(self.symbol)}, {self.probability}, {self.checked})"


class HuffmanCode:
    """ Class for Huffman coding."""

    def __init__(self):
        self.code = {}
        self.reverse_code = {}  # Reverse lookup table for decoding
        self.text = text
        self.hydrate_codes_from_text(text)

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

    def hydrate_codes_from_text(self, text):
        counter = Counter(text)
        total_chars = sum(counter.values())
        probabilities = {char: char_count/total_chars for char, char_count in counter.items()}
        root =  self.huffman_encoding(probabilities)
        return self.recursive_traverse(root)

    def byte_stream(self):
        current_chunk = ''
        for char in self.text:
            current_chunk += self.code[char]
            if len(current_chunk) >= BYTE_SIZE:
                byte_int = int(current_chunk[:BYTE_SIZE], 2)
                yield byte_int#.to_bytes(1, byteorder='big')
                current_chunk = current_chunk[BYTE_SIZE:]
        padding_size = BYTE_SIZE - len(current_chunk)
        current_chunk = current_chunk + '0' * padding_size
        yield int(current_chunk, 2)#.to_bytes(1, byteorder='big')

    def byte_stream_to_01(self, stream):
        for byte in stream:
            s = bin(byte)[2:]
            padding_size = BYTE_SIZE - len(s)
            s = '0' * padding_size + s
            for bit in s:
                yield bit

    def stream_to_text(self, stream, text_length=None):
        tree = make_tree(self.code)
        current_node = tree
        char_count = 0
        for byte in self.byte_stream_to_01(stream):
            current_node = current_node[byte]
            if isinstance(current_node, str):
                char_count += 1
                yield current_node
                current_node = tree
            if text_length is not None and char_count >= text_length:
                break

    @classmethod
    def compress(cls, file_name, out_name):
        with open(file_name, 'r') as f:
            text = f.read()
        hoffman = cls()
        hoffman.hydrate_codes_from_text(text)
        header = {'length': len(text), 'codes': hoffman.code}
        json_bytes = json.dumps(header).encode('utf-8')
        binary_blob = b''.join([b.to_bytes(1, byteorder='big') for b in hoffman.byte_stream()])

        with open(out_name, 'wb') as o:
            o.write(struct.pack("I", len(json_bytes)))
            o.write(json_bytes)
            o.write(binary_blob)
            for byte in hoffman.byte_stream():
                o.write(byte.to_bytes(1, byteorder='big'))

    @classmethod
    def decompress(cls, file_name, out_name):
        with open(file_name, 'rb') as f:
            header_size = struct.unpack("I", f.read(4))[0]
            header = json.loads(f.read(header_size).decode('utf-8'))
            codes = header['codes']
            length = header['length']
            hoffman = cls()
            hoffman.code = codes
            binary_blob = f.read()
        with open(out_name, 'w') as o:
            o.write(''.join(hoffman.stream_to_text(binary_blob, text_length=length)))







#%%

def make_tree(codes):
    if len(codes)==1:
        c = list(codes)[0]
        return c

    one_codes = {c: code[1:] for c, code in codes.items() if code.startswith('1')}
    zero_codes = {c: code[1:] for c, code in codes.items() if code.startswith('0')}

    one_tree = make_tree(one_codes)
    zero_tree = make_tree(zero_codes)

    return {'0': zero_tree, '1': one_tree}


if __name__ == "__main__":
    with open('HW4/test1.txt', 'r') as file:
        text = file.read()
    HuffmanCode.compress('HW4/test1.txt', 'HW4/test1.huff')
    print(HuffmanCode.decompress('HW4/test1.huff', 'HW4/test1_decompressed.txt'))





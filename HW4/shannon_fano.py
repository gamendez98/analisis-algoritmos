import math
import argparse
from collections import Counter

def calculate_shannon_entropy(probabilities):
    """Calculate the shannon entropy given symbol probabilities."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def calculate_worst_case_entropy(vocab_size):
    """Calculate the worst-case entropy given the vocabulary size."""
    return math.log2(vocab_size)

def expected_bits(probabilities, code_lengths):
    """Compute the expected bits per symbol using probability and code lengths."""
    return sum(p * l for p, l in zip(probabilities, code_lengths))

def shannon_fano_encoding(symbols, probabilities):
    """Generate Shannon-Fano encoding for given symbols and their probabilities."""
    sorted_symbols = sorted(zip(symbols, probabilities), key=lambda x: x[1], reverse=True)
    code_lengths = [math.ceil(math.log2(1 / p)) for _, p in sorted_symbols]
    codes = {}
    previous_code = 0
    
    for i, ((symbol, _), l) in enumerate(zip(sorted_symbols, code_lengths)):
        if i == 0:
            code = 0 # First symbol gets 0 as base code
        else:
            d_i = l - code_lengths[i - 1] # Difference in length between symbols
            code = (previous_code + 1) << d_i # Shift left to generate the next code
        codes[symbol] = format(code, f'0{l}b')
        previous_code = code
    
    return codes, code_lengths

def compress_text(text, codes):
    """Compress the input text using Shannon-Fano encoding."""
    return ''.join(codes[char] for char in text if char in codes)

def decompress_text(compressed_text, codes):
    """Decompress a Shannon-Fano encoded text back to its original form."""
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_text = ""
    buffer = ""
    
    for bit in compressed_text:
        buffer += bit
        if buffer in reverse_codes:
            decoded_text += reverse_codes[buffer]
            buffer = ""
    
    return decoded_text

def main():
    """Main function to handle file input, encoding, compression, and decompression validation."""
    parser = argparse.ArgumentParser(description="Shannon-Fano Text Compression")
    parser.add_argument("filepath", type=str, help="Path to the input text file")
    args = parser.parse_args()

    # Read the input text file
    with open(args.filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Compute symbol frequencies and probabilities
    frequency = Counter(text)
    total_chars = sum(frequency.values())
    probabilities = {char: count / total_chars for char, count in frequency.items()}
    
    # Generate encoding
    symbols = list(probabilities.keys())
    probs = list(probabilities.values())
    codes, code_lengths = shannon_fano_encoding(symbols, probs)
    
    # Compress and decompress the text
    compressed_text = compress_text(text, codes)
    decompressed_text = decompress_text(compressed_text, codes)

    # Compute entropies and expected bits per symbol
    worst_entropy = calculate_worst_case_entropy(len(symbols))
    entropy = calculate_shannon_entropy(probs)
    expected_bits_per_symbol = expected_bits(probs, code_lengths)
    total_bits = len(compressed_text)

    # Display results
    print("Shannon-Fano Encoding:")
    for char, code in codes.items():
        print(f"{repr(char)}: {code}")
    print(f"\nWorst-case Entropy: {worst_entropy:.4f} bits")
    print(f"Shannon Entropy: {entropy:.4f} bits")
    print(f"Expected Bits per Symbol: {expected_bits_per_symbol:.4f}")
    print(f"Total Bits Required: {total_bits}")
    
    # Verify decompression integrity
    assert text == decompressed_text, "Decompressed text does not match the original!"
    print("\nDecompression successful! The original text was correctly reconstructed.")
    
if __name__ == "__main__":
    main()
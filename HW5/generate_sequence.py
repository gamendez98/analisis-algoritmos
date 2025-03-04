import random

# Generate a random DNA sequence of 100,000 characters
dna_bases = ['A', 'T', 'C', 'G']
dna_sequence = ''.join(random.choices(dna_bases, k=10_000_000))

# Save to a .txt file with lines of 1000 characters each
file_path = "dna_sequence_10M.txt"
with open(file_path, "w") as file:
    for i in range(0, len(dna_sequence), 1000):
        file.write(dna_sequence[i:i+1000] + "\n")

# Provide the file for download
file_path

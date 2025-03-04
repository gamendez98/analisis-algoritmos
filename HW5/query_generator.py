import argparse
from random import randint

def generate_query(text, query_len):
    """Generate a query and print the query and its suffix."""
    with open(f"queries{query_len}.txt", "w") as file:
        for i in range(query_len):
            print(i)
            initial_pos = randint(0, len(text) - 1)
            lenght = randint(1, len(text) - initial_pos)
            query = text[initial_pos:initial_pos + lenght]
            file.write(f"{query}\n")

def main():
    """Main function to handle file input, suffix array construction, and binary search."""
    parser = argparse.ArgumentParser(description="Suffix Array and Binary Search")
    parser.add_argument("filepath", type=str, help="Path to the input text file")
    parser.add_argument("amount", type=int, help="Amount of queries to generate")
    args = parser.parse_args()

    with open(args.filepath, "r") as file:
        text = file.read()
    generate_query(text, args.amount)

if __name__ == "__main__":
    main()  
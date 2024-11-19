import os

def collect_digits_from_sol_files(directory, output_file):
    # List all .sol files in the directory (without sorting)
    sol_files = [f for f in os.listdir(directory) if f.endswith('.sol')]
    
    # Open the output file for writing
    with open(output_file, 'w') as output:
        for sol_file in sol_files:
            # Construct the full file path
            file_path = os.path.join(directory, sol_file)
            
            # Read the content of the .sol file
            with open(file_path, 'r') as file:
                content = file.read().strip()
                # Write the content to the output file
                output.write(content + '\n')

# Usage
input_directory = "./pattern_feature/label_by_extractor/reentrancy/validation"  # Change to your directory
output_file_path = "collected_digits_val.txt"  # Desired output file path
collect_digits_from_sol_files(input_directory, output_file_path)
print(f"Digits have been collected and saved to {output_file_path}.")
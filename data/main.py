import csv
import argparse

def convert_dat_to_csv(dat_file_path, csv_file_path, column_names=None):
    # Open the .dat file and read its contents
    with open(dat_file_path, 'r', encoding='latin1') as dat_file:
        lines = dat_file.readlines()

    # Open the .csv file in write mode
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the column names to the first row of the CSV file
        writer.writerow(column_names)
        
        # Process each line from the .dat file
        for line in lines:
            # Split the line by '::' and remove any surrounding whitespace
            data = line.strip().split('::')
            # Write the data to the CSV file
            writer.writerow(data)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Convert a .dat file to a .csv file.')
    
    # Add arguments for input and output file paths
    parser.add_argument('dat_file_path', help='The input path for the .dat file')
    parser.add_argument('csv_file_path', help='The output path for the .csv file')
    parser.add_argument('--column_names', nargs='+', help='Column names for the CSV file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to perform the conversion
    convert_dat_to_csv(args.dat_file_path, args.csv_file_path, args.column_names)

    print(f"Conversion complete. Data saved to {args.csv_file_path}")

if __name__ == '__main__':
    main()

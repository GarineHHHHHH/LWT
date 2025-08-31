import os
import csv

def process_csv_files(input_directory, output_directory):
    """
    Processes CSV files in an input directory, keeping only specified columns
    and reordering them, then saves the processed files to an output directory.

    Args:
        input_directory (str): The directory containing the input CSV files.
        output_directory (str): The directory to save the processed CSV files.
    """

    # Define the desired columns and their new order
    desired_columns = [
        "time",
        "product_gps_sv_number",
        "wifi_signal",
        "product_gps_longitude",
        "product_gps_latitude",
        "altitude",
        "speed_vx",
        "speed_vy",
        "speed_vz",
        "angle_phi",
        "angle_theta",
        "angle_psi",
    ]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)  # Save to the new directory
            print(f"Processing: {filename}")

            # Read the CSV file
            rows = []
            with open(input_filepath, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Create a new row with only the desired columns in the specified order
                    new_row = {col: row[col] for col in desired_columns}
                    rows.append(new_row)

            # Write the processed data to the output CSV file
            with open(output_filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=desired_columns)
                writer.writeheader()  # Write the header row
                writer.writerows(rows)  # Write the data rows

            print(f"Processed and saved: {filename} to {output_directory}")

# Example usage:
input_directory_path = "data_sup\\csv"  # Replace with the actual input directory path
output_directory_path = "data_sup\\csv2"  # Replace with the desired output directory path
process_csv_files(input_directory_path, output_directory_path)
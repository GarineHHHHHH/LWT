import json
import csv
import os

def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Converts a JSON file to a CSV file.

    Args:
        json_file_path (str): The path to the JSON file.
        csv_file_path (str): The path to the output CSV file.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        headers = data.get('details_headers')
        data_rows = data.get('details_data')

        if not headers or not data_rows:
            print(f"Warning: Could not find 'details_headers' or 'details_data' in {json_file_path}. Skipping.")
            return

        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write the header row
            csv_writer.writerow(headers)

            # Write the data rows
            csv_writer.writerows(data_rows)

        print(f"Successfully converted {json_file_path} to {csv_file_path}")

    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Converts all JSON files in the 'json' directory to CSV files in the root directory.
    """
    json_directory = 'data_sup\\json'
    output_directory = 'data_sup\\csv'  # Root directory

    if not os.path.exists(json_directory):
        print(f"Error: Directory '{json_directory}' not found.")
        return

    for filename in os.listdir(json_directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(json_directory, filename)
            csv_file_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.csv')
            convert_json_to_csv(json_file_path, csv_file_path)


if __name__ == "__main__":
    main()
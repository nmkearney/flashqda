import os, csv

def initialize_project(directory):
    
    """
    Prepare the project directory and required subdirectories.
    """

    os.makedirs(directory, exist_ok=True)
    os.makedirs(os.path.join(directory, 'Data'), exist_ok=True)
    os.makedirs(os.path.join(directory, 'Results'), exist_ok=True)
    os.chdir(directory)

def save_to_csv(data_list, results_folder, save_name):
    """
    Save a list of dictionaries to a CSV file.

    """
    csv_file_path = os.path.join(results_folder, save_name)
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
        # Assuming all dictionaries in the list have the same keys
        fieldnames = list(data_list[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the CSV header
        writer.writeheader()

        # Write each dictionary to the CSV file
        for entry in data_list:
            writer.writerow(entry)

    print("CSV file saved:", csv_file_path)

def read_csv_file(file_name):
    """
    Import a csv file from the 'Results' folder as a list of dictionaries
    """
    results_folder = 'Results/' + file_name + '/'
    file_name = file_name + '_sentences.csv'
    file_path = os.path.join(results_folder, file_name)

    try:
        with open(file_path, "r") as file:
            csv_reader = csv.DictReader(file)
            data = list(csv_reader)
            print("CSV file read:", file_path)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist in the 'Results' folder.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {str(e)}")
        return []
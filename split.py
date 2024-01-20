import os

def split_file(file_path, max_size=50*1024*1024, output_folder='split_files'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    part_number = 1
    with open(file_path, 'rb') as f:
        chunk = f.read(max_size)
        while chunk:
            part_filename = f"{os.path.join(output_folder, os.path.basename(file_path))}.part{part_number}"
            with open(part_filename, 'wb') as part_file:
                part_file.write(chunk)
            part_number += 1
            chunk = f.read(max_size)

    print(f"File {file_path} split into {part_number - 1} parts.")

# Example usage
split_file('/Users/Rene/Desktop/Pers-nlich/Yolo Computer Vision Project/best-detection-xl.pt')
split_file('/Users/Rene/Desktop/Pers-nlich/Yolo Computer Vision Project/best-segmentation-m.pt')

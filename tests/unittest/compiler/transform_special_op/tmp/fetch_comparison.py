import os
import re
from bs4 import BeautifulSoup
from difflib import unified_diff

def extract_sequence_number(filename):
    # Extract the sequence number from the filename (e.g., 01 from 01-dedup_make_jagged_ops_graph_vis.html)
    return int(filename.split('-')[0])

def read_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def compare_html_files(file1, file2):
    html1 = read_html(file1)
    html2 = read_html(file2)
    
    # Use BeautifulSoup to parse the HTML and get the text content for comparison
    soup1 = BeautifulSoup(html1, 'html.parser')
    soup2 = BeautifulSoup(html2, 'html.parser')
    
    text1 = soup1.get_text()
    text2 = soup2.get_text()
    
    diff = unified_diff(text1.splitlines(), text2.splitlines(), fromfile=file1, tofile=file2, lineterm='')
    
    return '\n'.join(diff)

def main():
    type = "bmm_rcr_n1_10_8_16_1_8_float32"
    directory = f"/home/wewe5215/Desktop/AITemplate/tests/unittest/compiler/transform_special_op/tmp/{type}"
    
    # Compile the regular expression pattern to match filenames
    pattern = re.compile(r'^\d+.*_pseudo_code\.txt$')

    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter HTML files that match the pattern
    graph_vis_files = [f for f in files if pattern.match(f)]
    
    # Sort files by their sequence number
    graph_vis_files.sort(key=extract_sequence_number)
    
    # Compare each file with the next one
    for i in range(len(graph_vis_files) - 1):
        file1 = os.path.join(directory, graph_vis_files[i])
        file2 = os.path.join(directory, graph_vis_files[i + 1])
        
        print(f"Comparing {graph_vis_files[i]} with {graph_vis_files[i + 1]}:")
        diff = compare_html_files(file1, file2)
        print(diff)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

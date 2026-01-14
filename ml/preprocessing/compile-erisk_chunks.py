import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pathlib

def compile_erisk_xml_chunks(root_dir: str, output_file: str):
    """
    Finds all individual user XML files, verifies their root tag is <INDIVIDUAL>,
    and combines them into a single master XML file.
    """
    print(f"--- Compiling eRisk XML Chunks from: {root_dir} ---")
    
    new_root = ET.Element('ERISK_DATA')
    
    # Use the robust os.walk to find all .xml files
    xml_file_paths = []
    print(f"Recursively searching for all .xml files inside '{root_dir}'...")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.xml'):
                full_path = os.path.join(dirpath, filename)
                xml_file_paths.append(full_path)

    if not xml_file_paths:
        raise FileNotFoundError(f"CRITICAL ERROR: No .xml files found anywhere inside '{root_dir}'. Please double-check the path.")
        
    print(f"Found {len(xml_file_paths)} total user XML files to process.")
    
    total_individuals_found = 0
    
    for xml_file in tqdm(xml_file_paths, desc="Processing XML Files"):
        try:
            tree = ET.parse(xml_file)
            xml_root = tree.getroot()
            
            if xml_root.tag == 'INDIVIDUAL':
                # Append the entire root element (the <INDIVIDUAL> block) to our new master root
                new_root.append(xml_root)
                total_individuals_found += 1
            else:
                print(f"Warning: Expected root tag <INDIVIDUAL> but found <{xml_root.tag}> in file {xml_file}. Skipping.")
            # ---------------------------------------------

        except ET.ParseError as e:
            print(f"Warning: Could not parse {xml_file}. Error: {e}. Skipping.")
            continue

    new_tree = ET.ElementTree(new_root)
    ET.indent(new_tree, space="  ", level=0)
    
    try:
        new_tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"\n--- Compilation Complete ---")
        print(f"Found and combined {total_individuals_found} <INDIVIDUAL> blocks.")
        print(f"Combined data saved to: {output_file}")
    except Exception as e:
        print(f"\nError writing to output file: {e}")

if __name__ == '__main__':

    ERISK_ROOT_DIRECTORY = 'data/reddit_test_depression/' 
    COMBINED_OUTPUT_FILE = 'data/reddit_test_depression/erisk_test_data_combined.xml'
    

    compile_erisk_xml_chunks(ERISK_ROOT_DIRECTORY, COMBINED_OUTPUT_FILE)
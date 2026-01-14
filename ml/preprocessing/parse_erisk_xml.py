import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import pathlib
import re

def parse_erisk_data(xml_path: str, labels_path: str) -> pd.DataFrame:
    print("--- Parsing eRisk Data (Corrected Structure) ---")
    
    label_dict = {}
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2: label_dict[parts[0]] = int(parts[1])
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    all_posts = []
    
    # Find all INDIVIDUAL tags
    individuals = root.findall('.//INDIVIDUAL')
    if not individuals:
        # Fallback for the chunk files where <INDIVIDUAL> is the root
        if root.tag == 'INDIVIDUAL':
            individuals = [root]

    print(f"Found {len(individuals)} <INDIVIDUAL> blocks to process.")
    
    for individual in tqdm(individuals, desc="Processing Users from XML"):
        try:
            # --- THIS IS THE DEFINITIVE FIX ---
            # Find the <ID> child tag and get its text content
            id_tag = individual.find('ID')
            if id_tag is None:
                continue # Skip this block if it has no ID tag
            user_id = id_tag.text
            # ----------------------------------
            
            if user_id in label_dict:
                for writing in individual.findall('WRITING'):
                    title = (writing.find('TITLE').text or "").strip()
                    text = (writing.find('TEXT').text or "").strip()
                    full_text = (title + " " + text).strip()
                    if not full_text: continue
                    all_posts.append({
                        'user_id': user_id,
                        'created': pd.to_datetime(writing.find('DATE').text, errors='coerce'),
                        'text': full_text,
                        'label': label_dict[user_id]
                    })
        except Exception as e:
            print(f"Warning: Error processing an INDIVIDUAL block: {e}")
            continue

    if not all_posts:
        raise ValueError("No posts were parsed. This indicates a persistent mismatch, even with the corrected logic.")

    df = pd.DataFrame(all_posts)
    df = df.dropna(subset=['created']).sort_values(['user_id', 'created']).reset_index(drop=True)
    df['label_name'] = df['label'].apply(lambda x: 'control' if x == 0 else 'depression')
    
    print("--- XML Parsing Complete ---")
    return df

if __name__ == '__main__':
    XML_FILE_PATH = 'data/reddit_test_depression/erisk_test_data_combined.xml'
    LABELS_FILE_PATH = 'data/reddit_test_depression/risk-golden-truth-test.txt'
    OUTPUT_PARQUET_PATH = pathlib.Path("data/erisk_parsed_raw.parquet")
    
    erisk_df = parse_erisk_data(XML_FILE_PATH, LABELS_FILE_PATH)
    erisk_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"\nSaved clean, raw eRisk data to: {OUTPUT_PARQUET_PATH}")
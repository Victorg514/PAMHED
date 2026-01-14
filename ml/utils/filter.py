import json

def filter_users_by_tweet_count(input_file, output_file, min_tweets=50):
    """
    Filter users from JSON file based on minimum tweet count.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        min_tweets (int): Minimum number of tweets required (default: 50)
    """
    
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter users with at least min_tweets tweets
    filtered_data = {}
    removed_users = []
    
    for username, tweets in data.items():
        if len(tweets) >= min_tweets:
            filtered_data[username] = tweets
        else:
            removed_users.append(username)
    
    # Save filtered data to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"Original users: {len(data)}")
    print(f"Users with {min_tweets}+ tweets: {len(filtered_data)}")
    print(f"Users removed: {len(removed_users)}")
    
    if removed_users:
        print(f"\nRemoved users (with tweet counts):")
        for user in removed_users:
            print(f"  - {user}: {len(data[user])} tweets")
    
    return filtered_data

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "candidate_controls.json"  # Your input file
    output_file = "final_candidate_controls.json"  # Output file
    
    # Filter users with less than 50 tweets
    filtered_data = filter_users_by_tweet_count(input_file, output_file, min_tweets=50)
    
    print(f"\nFiltered data saved to: {output_file}")
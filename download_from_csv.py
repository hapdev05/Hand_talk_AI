import csv
import requests
import os
from tqdm import tqdm
import urllib3
from urllib.parse import urlparse

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
csv_file = "Dataset/Texts/label.csv"
videos_dir = "Dataset/Videos"
base_url = "https://qipedc.moet.gov.vn/videos"

# Create videos directory if not exists
os.makedirs(videos_dir, exist_ok=True)

def download_video_from_url(video_filename, video_id, label):
    """Download video using requests (equivalent to curl -k -L)"""
    
    # Create full URL with autoplay parameter
    video_url = f"{base_url}/{video_filename}?autoplay=true"
    output_path = os.path.join(videos_dir, f"{video_id}.mp4")
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Skip: {video_id}.mp4 (already exists)")
        return True
    
    try:
        print(f"Downloading: {video_id}.mp4 ({label})")
        
        # Make request with SSL verification disabled (curl -k) and follow redirects (curl -L)
        response = requests.get(video_url, stream=True, verify=False, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Progress {video_id}.mp4",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=80
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    bar.update(size)
        
        print(f"✅ Completed: {video_id}.mp4")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {video_id}.mp4: {e}")
        # Remove partial file if download failed
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def main():
    print("🚀 Starting video download from CSV...")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    # Read CSV file
    videos_to_download = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row
            print(f"📋 CSV Header: {header}")
            
            for row in reader:
                if len(row) >= 3:
                    video_id = row[0]
                    video_filename = row[1]  # e.g., "D0001.mp4"
                    label = row[2]
                    videos_to_download.append((video_filename, video_id, label))
                    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    print(f"📊 Found {len(videos_to_download)} videos to download")
    
    # Download videos
    success_count = 0
    failed_count = 0
    
    for video_filename, video_id, label in videos_to_download:
        if download_video_from_url(video_filename, video_id, label):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n🎉 Download Summary:")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Videos saved to: {videos_dir}")

if __name__ == "__main__":
    main()

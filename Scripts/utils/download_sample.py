import requests
import os

def download_sample_sleep_edf():
    base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    files_to_download = [
        "SC4001E0-PSG.edf",
        "SC4001EC-Hypnogram.edf"
    ]
    
    data_dir = os.path.join("data", "sleep-edf")
    os.makedirs(data_dir, exist_ok=True)
    
    for f in files_to_download:
        url = base_url + f
        out_path = os.path.join(data_dir, f)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 100000:
            print(f"File {f} already exists and seems complete.")
            continue
            
        print(f"Downloading {f}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Downloaded {f}")
            
if __name__ == "__main__":
    download_sample_sleep_edf()

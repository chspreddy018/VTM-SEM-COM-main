import requests

def download_file(url, file_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Example usage
url = "https://example.com/myfile.txt"
file_path = "path/to/save/file.txt"
download_file(url, file_path)

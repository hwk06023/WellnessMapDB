import requests
import urllib.parse

def get_roadview_image(api_key, location):
    base_url = "https://naveropenapi.apigw.ntruss.com/map-static/v2/roadview"
    
    # location number to string
    location_str = f"{location[1]},{location[0]}"
    
    # URL Encoding
    url = f"{base_url}?location={location_str}&size=640x480&fov=120&pitch=10&key={api_key}"
    
    # API Request & Response
    response = requests.get(url)
    if response.status_code == 200:
        # Download Image
        with open("roadview_image.jpg", "wb") as f:
            f.write(response.content)
        print("Download succeeded.")
    else:
        print(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    api_key = "YOUR_CLIENT_ID" # Client ID
    location = (37.5665, 126.9780) 
    get_roadview_image(api_key, location)

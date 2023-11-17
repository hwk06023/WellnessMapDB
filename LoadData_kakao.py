import requests

def get_kakao_roadview_image(api_key, location):
    base_url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    
    # location number to string
    location_str = f"{location[1]},{location[0]}"
    
    # API header
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }
    
    # API Request & Response
    response = requests.get(base_url, params={"x": location[1], "y": location[0]}, headers=headers)
    
    if response.status_code == 200:
        # get address name
        address = response.json()["documents"][0]["address"]["address_name"]
        print(f"Address: {address}")
        roadview_url = f"https://dapi.kakao.com/v2/local/geo/transcoord.json?x={location[1]}&y={location[0]}&input_coord=WGS84&output_coord=TM"
        roadview_response = requests.get(roadview_url, headers=headers)
        
        if roadview_response.status_code == 200:
            trans_coord = roadview_response.json()["documents"][0]["y"], roadview_response.json()["documents"][0]["x"]
            roadview_image_url = f"http://kapi.kakao.com/v1/panorama?panoid={trans_coord[1]},{trans_coord[0]}"
            
            roadview_image_response = requests.get(roadview_image_url, headers=headers)
            
            if roadview_image_response.status_code == 200:
                with open("kakao_roadview_image.jpg", "wb") as f:
                    f.write(roadview_image_response.content)
                print("Download succeeded.")
            else:
                print(f"Download Error: {roadview_image_response.status_code}, {roadview_image_response.text}")
        else:
            print(f"Transform Error: {roadview_response.status_code}, {roadview_response.text}")
    else:
        print(f"Another response Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    api_key = "YOUR_KAKAO_API_KEY" # Client ID
    location = (37.5665, 126.9780) 
    get_kakao_roadview_image(api_key, location)

import requests

def get_kakao_roadview_image(api_key, location, heading=0, pitch=0, size=(640, 400)):
    base_url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
    
    # API header
    headers = {
        "Authorization": f"KakaoAK {api_key}"
    }

    roadview_url = f"https://dapi.kakao.com/v2/local/geo/transcoord.json?x={location[1]}&y={location[0]}&input_coord=WTM&output_coord=WGS84"
    roadview_response = requests.get(roadview_url, headers=headers)
    location = roadview_response.json()["documents"][0]["y"], roadview_response.json()["documents"][0]["x"]

    # API Request & Response
    response = requests.get(base_url, params={"x": location[1], "y": location[0]}, headers=headers)
    print(location)
    
    if response.status_code == 200:
        if roadview_response.status_code == 200:
            roadview_image_url = f"https://dapi.kakao.com/v2/local/geo/coord2address.json?x={location[0]}&y={location[1]}"
            roadview_image_response = requests.get(roadview_image_url, headers=headers)
            print(roadview_image_response.url)
            
            if roadview_image_response.status_code == 200:
                # This URL is not working yet.
                image_url = f"https://map2.daum.net/map/imageservice?FORMAT=PNG&SCALE=2.5&MX={location[1]}&MY={location[0]}&CX={location[1]}&CY={location[0]}&WIDTH={size[0]}&HEIGHT={size[1]}&SERVICE=KAKAO_ROADVIEW"
                print(image_url)

                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    with open("roadview.png", "wb") as f:
                        f.write(image_response.content)
                else:
                    print(f"Image_response Error: {image_response.status_code}, {image_response.text}")
            
            else:
                print(f"Roadview_image_response Error: {roadview_image_response.status_code}, {roadview_image_response.text}")
        else:
            print(f"Transform Error: {roadview_response.status_code}, {roadview_response.text}")
    else:
        print(f"Another response Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    api_key = "" # Client ID
    location = (462407, 1102810) 
    get_kakao_roadview_image(api_key, location)

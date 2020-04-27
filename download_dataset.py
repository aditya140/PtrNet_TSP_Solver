import requests
from zipfile import ZipFile
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    download_file_from_google_drive("0B2fg8yPGn2TCSW1pNTJMXzFPYTg","./data/tsp5.zip")
    with ZipFile('./data/tsp5.zip', 'r') as zipObj:
        zipObj.extractall('./data/')
    download_file_from_google_drive("0B2fg8yPGn2TCbHowM0hfOTJCNkU_","./data/tsp10.zip")
    with ZipFile('./data/tsp5.zip', 'r') as zipObj:
        zipObj.extractall('./data/')
    download_file_from_google_drive("0B2fg8yPGn2TCUVlCQmQtelpZTTQ","./data/tsp50_test.zip")
    with ZipFile('./data/tsp5.zip', 'r') as zipObj:
        zipObj.extractall('./data/')
    download_file_from_google_drive("0B2fg8yPGn2TCTWNxX21jTDBGeXc","./data/tsp5-20_train.zip")
    with ZipFile('./data/tsp5.zip', 'r') as zipObj:
        zipObj.extractall('./data/')
    download_file_from_google_drive("0B2fg8yPGn2TCaVQxSl9ab29QajA","./data/tsp50_train.zip")
    with ZipFile('./data/tsp5.zip', 'r') as zipObj:
        zipObj.extractall('./data/')

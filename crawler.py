import requests
import bs4 as bs
import os

rq = requests.Session()

def search_idols():
    SEARCH_URL  = "https://javmodel.com/jav/order_homepages.php?model_cat=6%20Stars%20JAV"

    page        = rq.get(SEARCH_URL)
    soup        = bs.BeautifulSoup(page.text, 'lxml')

    results     = []

    # "text-center" class contains idol names and bio links
    for search_result in soup.find_all(class_ = "text-center"):
        idol    = search_result.select("a[href]")[0]
        result  = { "profile_link"    :   f"https://javmodel.com{idol['href']}",
                    "name"            :   idol.text}

        results.append(result)

    return results

def get_img(idol):
    PATH = f'./data/{idol["name"]}'
    try:
        os.mkdir(PATH)
    except FileExistsError:
        pass

    page = rq.get(idol['profile_link'])

    soup = bs.BeautifulSoup(page.text, 'lxml')

    for image in soup.select("img[alt]"):
        file_name = image['alt']
        image_url = image['src']

        response = rq.get(image_url)
        if (response.status_code == 200):
            global img_count
            img_count += 1
            with open(f'{PATH}/{file_name}.jpg', 'wb') as f:
                f.write(response.content)

if (__name__ == "__main__"):
    try:
        os.mkdir("./data")
    except FileExistsError:
        pass

    idols       = search_idols()

    idol_count  = 0
    img_count   = 0

    for idol in idols:
        idol_count += 1
        print(f"Crawling {idol['name']} data ({idol_count}/{len(idols)})")
        get_img(idol)

    print(f'Crawled {img_count} pictures from {idol_count} idols')
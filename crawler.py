import os
from typing import Dict
import sqlite3

import requests
import bs4 as bs

rq = requests.Session()


def create_db():
    """
    Create idol database. WARNING: will erase existing db
    :return: connection, cur
    """
    conn = sqlite3.connect('idol.sqlite')
    cur = conn.cursor()

    try:
        cur.execute('DROP TABLE picture')
    except sqlite3.OperationalError:
        pass

    cur.execute('CREATE TABLE picture('
                   'idol_name varchar(40),'
                   'pic_name varchar(100))')

    return conn, cur


def search_idols():
    """
    Create a list of all idols
    :return: List of all idol names
    """
    SEARCH_URL = "https://javmodel.com/jav/order_homepages.php?model_cat=6%20Stars%20JAV"

    page = rq.get(SEARCH_URL)
    soup = bs.BeautifulSoup(page.text, 'lxml')

    results = []

    # "text-center" class contains idol names and bio links
    for search_result in soup.find_all(class_="text-center"):
        idol = search_result.select("a[href]")[0]
        result = {"profile_link": f"https://javmodel.com{idol['href']}",
                  "name": idol.text}

        results.append(result)

    return results


def get_img(idol: Dict):
    """
    Crawl idol images
    :param idol: dictionary (name, profile_link) of a specific idol
    :return: list of images of the idol
    """
    idol_name = idol['name']
    # idol_name = '_'.join(idol_name.split())

    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'data')
    # PATH = os.path.join(PATH, idol_name)

    # PATH = f'./data/{idol["name"]}'
    try:
        os.mkdir(PATH)
    except FileExistsError:
        pass

    page = rq.get(idol['profile_link'])

    soup = bs.BeautifulSoup(page.text, 'lxml')

    global img_count, conn, cur
    for image in soup.select("img[alt]"):
        # file_name = f'{idol_name}_{image["alt"]}.jpg'
        file_name = f'{image["alt"]}.jpg'
        image_url = image['src']

        response = rq.get(image_url)
        if response.status_code == 200:
            img_count += 1
            with open(f'{os.path.join(PATH, file_name)}', 'wb') as f:
                f.write(response.content)
            cur.execute('INSERT INTO picture VALUES(?, ?)', (idol_name, file_name))
            # print(cur.fetchone())


if __name__ == "__main__":
    try:
        os.mkdir('data')
    except FileExistsError:
        pass

    conn, cur = create_db()
    idols = search_idols()

    idol_count = 0
    img_count = 0

    for idol in idols:
        idol_count += 1
        print(f"Crawling {idol['name']} data ({idol_count}/{len(idols)})")
        get_img(idol)

    print(f'Crawled {img_count} pictures from {idol_count} idols')
    conn.commit()
    conn.close()

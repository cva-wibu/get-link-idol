import os
import requests
import sqlite3
import bs4 as bs

from typing import Dict

rq = requests.Session()
BASE_URL = "https://javmodel.com"


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


def crawl_idols_data():
    """
    Crawl every single idols, each ~10 images
    :return: a sqlite3 database of idol names and pictures
    """
    import re

    idol_count = 0
    img_count = 0

    first_page = "https://javmodel.com/jav/homepages.php?page=1"
    page = rq.get(first_page)

    total_idol_number = re.findall(r'Total (\d+?) JAVModels Found', page.text)[0]

    next_page = first_page

    # repeat until reach last result page
    while True:
        page = rq.get(next_page)
        soup = bs.BeautifulSoup(page.text, 'lxml')

        # "text-center" class contains idol names and bio links
        for search_result in soup.find_all(class_="text-center"):
            result = search_result.select("a[href]")[0]

            idol = {"profile_link": BASE_URL + result['href'],
                    "name": result.text}

            idol_count += 1
            print(f"Crawling {idol['name']} data ({idol_count}/{total_idol_number})")
            get_img(idol)

        try:
            # find next page url
            next_page = BASE_URL + (soup.find_all("a", string="Next")[0]['href'])
        except IndexError:
            # if not found then reached last page, break loop
            break

    print(f'Crawled {img_count} pictures from {idol_count} idols')


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

    crawl_idols_data()

    conn.commit()
    conn.close()

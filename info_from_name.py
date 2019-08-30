import requests
import bs4 as bs
import sys

rq = requests.Session()

def search_idols(search_query):
    SEARCH_URL  = "https://javmodel.com/jav/video_search.php?user_movie%2Cjap_user_movie%2Cmovie_cat%2Cmovie_cat_chi%2Cjap_title_text%2Ctitle_text%2Cref_no%2Cjap_fullname%2Cfullname_keyword="

    # there're multiple "selftag" css classes, and the class contains idol names is the first one
    IDOL_NAME   = 0

    page        = rq.get(SEARCH_URL + search_query)
    soup        = bs.BeautifulSoup(page.text, 'lxml')

    results     = []

    #"unstyled-list list-medium" class contains each search result
    for search_result in soup.find_all(class_ = "unstyled-list list-medium"):
        profile_link  = search_result.find_all(class_ = "selftag")[IDOL_NAME]['href']
        name          = search_result.find_all(class_ = "selftag")[IDOL_NAME].text

        idol          = { "profile_link"    :   profile_link,
                          "name"            :   name.strip()}

        if (idol not in results):
            results.append(idol)

    return results

def get_idol_info(profile_link):
    DB_URL      =   "https://javmodel.com/jav/"
    IDOL_TAGS   =   0
    CAREER      =   1
    FILM_CLASSES =   1

    page        =   rq.get(DB_URL + profile_link)
    soup        =   bs.BeautifulSoup(page.text, 'lxml')
    idol_bio    =   soup.find_all(class_ = "col-sm-4")[0].text

    idol_info   =   idol_bio.rstrip()

    # find idol's tags
    tags        =   soup.find_all(class_ = "unstyled-list list-medium")[CAREER].find_all("li")[IDOL_TAGS]
    idol_info   +=  "\n Tags: "

    for tag in tags.find_all("a"):
        idol_info += tag.text + ", "

    # find idol's video classes
    idol_info   +=  "\n Video Class: "    
    video_class =   soup.find_all(class_ = "unstyled-list list-medium")[CAREER].find_all("li")[FILM_CLASSES]

    for class_ in video_class.find_all("a"):
        idol_info += class_.text + ' '

    # find some idol's sample codes:
    idol_info   +=  "\n Sample codes: "

    for img in soup.find(id = "abc").find_all("img"):
        try:
            idol_info += f"{img['alt']}, "
        except (KeyError):
            pass

    return idol_info + "\n\n"

def get_search_query():
    # search query empty
    if (len(sys.argv) <= 1):
        print("Give me some names")
        sys.exit()

    search_query = ''
    for arg in sys.argv[1:]:
        search_query += arg + ' '

    return search_query

if (__name__ == "__main__"):

    search_query = get_search_query()

    search_results = search_idols(search_query)
    search_results = list(search_results)

    is_not_found = len(search_results) == 0
    if (is_not_found):
        print("404 idol not found")
        sys.exit()
    else:
        for idol in search_results:
            print(idol['name'])
            print(get_idol_info(idol['profile_link']))
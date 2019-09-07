import praw
import sys
from credentials import Credentials


def praw_init():
    USER_AGENT = "linux:com.cvawibu.getlinkidol:v0.1.1 (by /u/presariohg)"
    reddit = praw.Reddit(user_agent=USER_AGENT,
                         client_id=Credentials.CLIENT_ID,
                         client_secret=Credentials.CLIENT_SECRET,
                         username=Credentials.USERNAME,
                         password=Credentials.PASSWORD)

    return reddit


def get_search_query():
    # search query empty
    if len(sys.argv) <= 1:
        print("Give me some clues")
        sys.exit()

    search_query = ''
    for arg in sys.argv[1:]:
        search_query += arg + ' '

    return search_query


if __name__ == '__main__':

    search_query = get_search_query()

    reddit = praw_init()

    search_results = reddit.subreddit('JavDownloadCenter').search(search_query)
    search_results = list(search_results)

    is_not_found = len(search_results) == 0
    if is_not_found:
        print('Found no results with that query')
    else:
        for submission in search_results:
            submission.comment_sort = 'best'

            is_found_link = False

            for comment in submission.comments:
                # find all links in submission's comments
                if "https" in comment.body:
                    print(comment.body)
                    is_found_link = True

            if not is_found_link:
                print("\n\nSubmission found but cannot find any download links. Would you mind take a look yourself?")
                print(f' {submission.title} ({submission.shortlink})\n\n')

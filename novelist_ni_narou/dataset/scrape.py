# -*- coding: utf-8 -*-
import json
import os
import re
import urllib.error
import urllib.request
import string
import glob

from bs4 import BeautifulSoup

ROOTDIR = os.path.dirname(__file__)
SAVEDIR = os.path.join(ROOTDIR, "dataset")

def save(book_id: str) -> None:
    url = "http://ncode.syosetu.com/novelview/infotop/ncode/{}/".format(book_id)
    try:
        resource = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        return
    soup = BeautifulSoup(resource, "html.parser")
    info_soup = soup.find("table", {"id": "noveltable1"})  # type: BeautifulSoup
    if info_soup is None:  # R18
        return
    info_list = list(info_soup.find_all("tr"))  # type: List[BeautifulSoup]
    summary_line_soup = info_list[0]
    genre_line_soup = info_list[-1]
    summary = summary_line_soup.find("td").text
    genre = genre_line_soup.find("td").text
    matcher = re.match('(.*)〔(.*)〕', genre)
    genre_small, genre_large = matcher.groups()

    status_soup = soup.find("table", {"id": "noveltable2"})  # type: BeautifulSoup
    status_soups = status_soup.find_all("td") # type: List[BeautifulSoup]
    if len(status_soups) == 8:
        upload_date_soup, messages_soup, reviews_soup, bookmarks, total_point, points, public_satus, characters \
            = status_soup.find_all("td")
    elif len(status_soups) == 9:
        upload_date_soup, update_date_soup, messages_soup, reviews_soup, bookmarks, total_point, points, public_satus, characters \
            = status_soup.find_all("td")
    upload_date = upload_date_soup.text
    matcher = re.match('.*([0-9]+)件.*', messages_soup.text, flags=re.DOTALL)
    messages = matcher.groups()[0] if (not matcher is None) else -1
    matcher = re.match('.*([0-9]+)件.*', reviews_soup.text, flags=re.DOTALL)
    reviews = matcher.groups()[0] if (not matcher is None) else -1
    matcher = re.match('.*([0-9]+)件.*', bookmarks.text)
    bookmarks = matcher.groups()[0] if (not matcher is None) else -1
    matcher = re.match('.*([0-9]+)pt.*', total_point.text)
    total_point = matcher.groups()[0] if (not matcher is None) else -1

    with open(os.path.join(SAVEDIR, "{}.json".format(book_id)), "w+") as f:
        json.dump({
            "genre_large": genre_large,
            "genre_small": genre_small,
            "summary": summary,
            "upload_date": upload_date,
            "messages": messages,
            "reviews": reviews,
            "bookmarks": bookmarks,
            "total_point": total_point
        }, f)

for major_code in ["c", "d"]:
    for minor_code in string.ascii_lowercase:
        if len(glob.glob(os.path.join(SAVEDIR, "n*{}{}.json".format(major_code, minor_code)))) > 0:
            continue
        for i in range(10000):
            print("n{:04}{}{}".format(i, major_code, minor_code))
            save("n{:04}{}{}".format(i, major_code, minor_code))

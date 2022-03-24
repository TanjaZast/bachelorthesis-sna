from __future__ import division, unicode_literals 
import codecs
from bs4 import BeautifulSoup

f=codecs.open("/Users/tanjazast/Downloads/tanjalein___20211030/followers_and_following/following.html", 'r', 'utf-8')
document= BeautifulSoup(f.read()).get_text()

print(document)
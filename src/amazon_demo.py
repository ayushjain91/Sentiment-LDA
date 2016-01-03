from sentimentLDA import *
import os
import urllib
import tarfile
vocabSize = 50000


def readData():
    folders = ['dvd']

    reviews = []
    for folder in folders:
        for f in ['positive', 'negative']:
            with open('./../data/sorted_data_acl/' +  folder + '/' + f +'.review') as fin:
                xmlStr = ''.join(fin.readlines())
                reviewRegex = r"(?s)\<review_text\>(.*?)\<\/review_text\>"
                reviewsInFile = re.findall(reviewRegex, xmlStr)
                reviews.extend(reviewsInFile)
    return reviews

if not os.path.exists('./../data/sorted_data_acl'):
    urllib.urlretrieve ("https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz", "amazon_data.tar.gz")
    tf = tarfile.open('amazon_data.tar.gz')
    for member in tf.getmembers():
        tf.extract(member, './../data/')
    tf.close()
    os.remove('amazon_data.tar.gz')


reviews = readData()

sampler = SentimentLDAGibbsSampler(1, 2.5, 0.1, 0.3)
sampler.run(reviews, 200, "./../data/sorted_data_acl/dvd_reviews.dll", True)

sampler.getTopKWords(25)

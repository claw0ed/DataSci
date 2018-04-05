# twitters

from twitterscraper import query_tweets

if __name__ == '__main__':
    #print the retrieved tweets to the screen:
    for tweet in query_tweets("서울시장", 10): # 검색하고싶은 내용
        print(tweet.timestamp) # 작성시간
        print(tweet.text)      # 트윗내용

    #Or save the retrieved tweets to file:
    file = open("output.txt","wb")
    for tweet in query_tweets("서울시장", 10):
        file.write(str(tweet.timestamp).encode())
        file.write(tweet.text.encode('utf-8'))
    file.close()
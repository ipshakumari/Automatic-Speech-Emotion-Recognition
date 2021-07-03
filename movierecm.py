import webbrowser
import requests as HTTP
import time


def countdown(t):
    
    while t:
        mins, secs = divmod(t, 60)
        timer = f"Lets redirect you to the page in: {secs}"
        print(timer, end="\r")
        time.sleep(1)
        t -= 1



def getmovie(emotion):
    
    time.sleep(3)
    
    countdown(5)
    urlhere=""

    # IMDb Url for movie against emotion Sad
    if(emotion == "sad"):
        urlhere = 'https://www.imdb.com/list/ls050578618/'
  
    # IMDb Url for movie against emotion Happy
    elif(emotion == "happy"):
        urlhere = 'https://www.imdb.com/search/title/?genres=comedy&title_type=feature&explore=genres'
  
    # IMDb Url for movie against emotion Anger
    elif(emotion == "angry"):
        urlhere = 'https://www.imdb.com/list/ls053456706/'

    webbrowser.open_new(urlhere)
  


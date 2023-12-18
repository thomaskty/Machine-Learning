library()
library(glue)
library(tidytext)
library(stringr)
library(tidyverse)

# reading the folder 
file = list.files('/home/thomaskutty/textfiles folder/')

print(head(files))

# ggetting the path of each text file in the directory 
filename = glue(folderpath, files(1), sep = '')
head(filename)

# removing the trail spaces 
filename = trimes(filename)

# reading the text in the file 
filetext = glue(read_file(filename))
head(filetext) 


filetext = gsub("\\$", "", filetext)
head(filetext) 

# creating tokens 
tokenization is a way  of seperating a piece of text into small units called tokens. 
here tokens can be either words, characters, or subwords 


tokens = tibble(txt = filetext)  %>% unnest_tokens(word, text)
print(head(tokens)) 
# afin, bing, lo
sentiments = tokens %>% inner_join(get_sentiments("bing")) %>% 
count(sentiments) %>% spread(sentiment, n, fill = 0) %>%
mutate(sentiment = positive -negative)

get_sentiment()












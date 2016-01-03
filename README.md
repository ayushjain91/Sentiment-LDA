# Sentiment-LDA
This repository contains code to run a joint topic and sentiment model on text reviews. A Gibbs sampling based inferencer is implemented 
for a joint topic and sentiment model. 
For details, see [Sentiment Analysis with Global Topics and Local Dependency]
(https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1913/2215).

### Running the Code
Run the code with
```sh
$ python amazon_demo.py
```
This script downloads amazon reviews ([from here](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)) for 4 categories
-- *books, dvd, electronics, kitchen* -- and runs the **SentimentLDAGibbsSampler** on the DVD data.

### The Generative Process for a Review
1. Each topic-sentiment pair `(t,s)` has an associated latent word distribution `phi(t,s)`-- words like *delicious* have high probability for positive sentiment for topic *food*, as opposed to negative sentiment for topic *movies*.
2. For each document `d`:
    1. Sample a topic distribution `theta(d) ~ Dirichlet(alpha)`.
    2. For each topic `t`, sample a distribution of sentiments `pi(d,t) ~ Dirichlet(gamma)`.
    3. For every word `w` in `d`:
        -  Sample a topic `t ~ theta(d)`
        -  Sample a sentiment `s ~ pi(d,t)`
        -  Sample a word `w ~ phi(t,s)`

### Results

##### Top Words for Positive Sentiment 
['movi', 'like', 'dvd', 'watch', 'good', 'time', 'great', 'realli', 'love', 'think', 'want', 'know', 'thing', 'best', 'better', 'make', 'look', 'stori', 'year', 'say', 'film', 've', 'seen', 'music', 'enjoy']
##### Top Words for Negative Sentiment 
['film', 'charact', 'make', 'bad', 'life', 'scene', 'play', 'actual', 'stori', 'man', 'long', 'work', 'end', 'need', 'peopl', 'director', 'act', 'perform', 'come', 'hard', 'role', 'young', 'happen', 'way', 'view']





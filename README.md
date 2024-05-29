# gpt

Once upon a time, in the land of language, there existed a vast network of words and phrases. These words, known as tokens, lived in harmony, constantly communicating with one another to create meaningful sentences and stories. Much like the human brain, which brilliantly stores and understands the world through complex chemical interactions, these tokens relied on intricate relationships to convey meaning. Teaching machines to generate meaningful text involved unraveling these relationships and applying mathematical techniques to simulate this linguistic magic.

Imagine a sequence of tokens: T = t1, t2, t3, t4, t5, and so on. For a machine to predict the next token in a sequence, it needed to understand the context provided by the previous tokens. At any point, the information carried by previous tokens was crucial for generating the next one. The challenge was to design a system that could accurately capture and utilize these relationships.

Initially, one might think of a simple approach: averaging the information from t1 to t5 and using this average to predict the next token. However, this method proved to be too simplistic and ineffective. A more sophisticated approach was needed—one that could identify and emphasize the most relevant past tokens for any given token in the sequence.

Enter the concept of self-attention, a revolutionary idea that transformed how tokens communicated. Self-attention allowed each token to evaluate its importance relative to others in a data-dependent manner. It did this through the use of query and key vectors. The query vector represented the information a token was seeking, while the key vector represented the information a token contained.

To determine the affinities between tokens, the system performed a dot product between the query and key vectors. Tokens with aligned queries and keys had higher affinities, meaning they were more relevant to each other. This mechanism enabled the sequence to learn which tokens were most important to the current token.

1. Self-attention acted as a sophisticated communication mechanism, enabling tokens to exchange information effectively.
2. Unlike convolutional neural networks, self-attention lacked an inherent understanding of positional information, necessitating the addition of positional encodings to address spatial structure.
3. Communication occurred within the batch dimension, ensuring that tokens interacted within their context.
4. In certain applications, such as sentiment analysis, all tokens could communicate freely, forming an encoder block where no token was excluded.
5. The decoder block, used in autoregressive models, ensured that future tokens never influenced past ones, maintaining a strict temporal order.
6. Cross-attention introduced another layer of complexity, allowing queries from one pool of tokens to interact with keys and values from another, pulling information across different contexts.
7. The main attention formula required scaling by the square root of the key dimension (sqrt(d_k)). This adjustment ensured that the variance of the attention weights remained close to one, preventing the softmax function from becoming too saturated or too peaky.

In this way, self-attention became the cornerstone of modern language models, enabling machines to generate coherent and contextually relevant text by mimicking the intricate web of relationships that exist in human language. And thus, the tokens continued to live in harmony, weaving words into stories with the help of their newfound mathematical magic.





----------------------------------------

GPT Implementation

Story begins as we unfold the friendly relationships between words or a.k.a tokens. Human brain is a beautiful amalgamation of chemicals which magically stores and understands the world and in particular the relationship between the words. Although not everyone does :). While teaching machines on generating meanigful text/tokens, it becomes extremely important to use mathematical techniques and tricks that will help us build this releationship. Goal is simple for a given input text, at any time step at a given word or token we need to get the information from previous tokens or corpus. At a particular token this information is super important, because this is what will be used to generate the next token. In simple terms, for generating or predicting next token, having knowledge of text in question is important.

So question is how to design the relationship between the tokens and carry that information. Lets take an example here
T = t1, t2, t3, t4, t5, -, -, -, - where t5 needs to have information carried from t1 to t5 for predicting next token. The most naive way of doing this is by carrying the average of t1 to t5 and feed it to t5. We also at some point start looking at spatial structure of tokens a.k.a positional encodings. The average information is not very good way to design a firendly system for tokens. We need something better. At this point you can actually go back to mathematics and learn the ways in which these tokens can communicate? 

At token t5, we should know what tokens in the past are most important for t5 and which ones are closely related. e.g. vowels are interested in knowing about consonants in the past, consonants are interested in knowing about vowels in the past.

This is where Self attention solves the problem of getting information in a data dependent way! Each token will emit query and key vector.
Query vector: what information i am looking for?
Key vector: what information do i contain?
The way we get the effinities between the tokens now is by doing dot product between the Q & K's, so my query will dot product will all the keys which will become the wei. If the query and the key are aligned they will have a very high amount. So in the sequence we get to learn more about such tokens.

 1. ultimately attention is just a communication mechanism. It is a way to communicate between the tokens.
 2. attention mechanism in its basic form do not have idea about position (notion of space should be added), unlike convolutional neural networks
 3. across batch dimension, we are not communicating, we are communicating only within the batch dimension
 4. sometimes like for sentiment analysis, all tokens can communicate with each other, basically we call it encoder block, basically we just remove line with '-inf'
 5. above implementation is called decoder block (nodes from future never talk to the past nodes), this is a.k.a autoregressive model
 6. unlike self attention above where k,q,v are working on same node pool. The cross attention can have q from one node pool and k,v from another node pool, q is pulling information from second node pool
 7. in the main attention formula, we also have to divide by sqrt(d_k). d_k is the head size, also called scaled attention. 
    k, q are unit variance i.e. unit guassian inputs, 'wei' will be unit variance too. see below.
    'wei' is fed to softmax, its important to variance of wei near to 1. If variance is high, softmax will saturate and if variance is low, softmax will be very peaky.



------------------------------------
(py310) (base) manpreet.singh@192 gpt % /Users/manpreet.singh/opt/anaconda3/envs/py310/bin/python /Users/manpreet.singh/git/gpt/notebooks/model.py
step: 0, train loss: 4.2849, valid loss: 4.2823
step: 500, train loss: 1.9986, valid loss: 2.0906
step: 1000, train loss: 1.5954, valid loss: 1.7593
step: 1500, train loss: 1.4326, valid loss: 1.6421
step: 2000, train loss: 1.3378, valid loss: 1.5672
step: 2500, train loss: 1.2761, valid loss: 1.5293
step: 3000, train loss: 1.2223, valid loss: 1.5080
step: 3500, train loss: 1.1832, valid loss: 1.4854
step: 4000, train loss: 1.1405, valid loss: 1.4785
step: 4500, train loss: 1.1058, valid loss: 1.4883

No, no more rich of my sight a white direction
Make before him his kinsmanless present.

QUEEN MARGARET:
And welcome, Clarence, I'll to French not Warwick
Isabel, find givillion handly at this won.

KING HENRY VI:
The want sits of charity slowy of bleedings,
Ere four all the vasterance; to the bosoms
Of fair brookes which me,--pleased you not lives have other.
Did proclaim the sacret month a rock
Off morticly say: he lain,
I deserve a despended pardon and to night
Have title restite of so ever'd
(py310) (base) manpreet.singh@192 gpt %
------------------------------------
# Personal ESN Analysis
<img width="2791" height="1884" alt="image" src="https://github.com/user-attachments/assets/c1d4d688-e513-4c8a-98ff-53491461e4df" />

## Can we use ESNs as our bias parameter for RAGs?
**Emergent semantics** (Santini et al,  The International Federation for Information Processing 1999) are a form of data that occurs when a user looks at a piece of media. What does the user feel? Do they feel its a warm image? Do they think its a pretty image? It turns out, these form of semantics is highly dependent on the personna of the end user. 

Rahul et al formalized emergent semantics into a network (IEEE International Conference on Semantic Computing 2011). This mini-project aims to evaluate the feasibility of using emergent semantics formalized as a graph to create the parameters of RAG bias.

## Some Analysis
### Can CLIP be used to bias for RAG?
- A | CLIP is a generalized framework to contrast images and text. It still does not do a very good job at calculating the emergent semantic differences between image 1 and image 2. Here is an example of the embedding space on streamlit:
  <img width="2189" height="1304" alt="image" src="https://github.com/user-attachments/assets/af296564-0145-46db-af8d-dfa928438c80" />
  You can see here that the distance between the images of rivers is the same as it is to the building. Why? Most people would place these two closer to each other then the other two buildings. This is an example of (a seemingly vision problem but actually) emergent semantic.

- You can argue that its not fair, that some people will associate the rivers more closely together while others might associate that top river to the Staples in Iowa City more (if they can recognize that its the building and its map location). But this is actually what proves our point that semantics is not very universally defined. Furthermore, we can exploit this relationship to characterize the trait of being able to recognize that building, which is highly individual specific, and see if we can exploit that relationship somehow to create our RAG personality.

## Notes
imagedata is the original high resolution images. To get workdata, please run fix.sh to move the workdata from surveyapp/src/lib to the root. Please contact me at mzeng5[at]uiowa[dot]edu if you need the high resolution images.

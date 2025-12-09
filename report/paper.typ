= Superblob: Biasing Multimodal Documents in Graph-Based RAG. 
#show figure: set block(width: 50%)
#show figure: set align(center)

Mike Zeng \<mzeng5\@uiowa.edu>

Emergent semantics (Santini et al, The International Federation for Information Processing 1999) are a form of data that occurs when a user looks at a piece of media. What does the user feel? Do they feel its a warm image? Do they think its a pretty image? It turns out, these form of semantics is highly dependent on the personna of the end user.

Rahul et al formalized emergent semantics into a network (IEEE International Conference on Semantic Computing 2011).

#figure(
  image("images/generic.png"),
  caption:[ESN created via user survey],
)

In this project, we utilize ESNs for indexing and retrival, demonstrating how we can create personalized text/image categorization and how we can use these to bias LLM responses via inverted-semantic retrival RAG.

== Emergent Semantic Network Generation Step
Prior to conducting any analysis, we first develop and implement the survey platform for collecting the ESNs.
#figure(
  image("images/interface.png"),
  caption:[Survey User Interface],
)

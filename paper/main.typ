#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Superblob: Biasing Retrieval Augmentation with Multimodal Documents],
  abstract: [
    Retrieval Augmented Generation (RAG) systems typically rely on objective text-image matching, often failing to capture the subjective or "emergent" semantics that specific user groups associate with visual media. In this paper, we introduce a novel framework for biasing Large Multimodal Model (LMM) responses using Emergent Semantic Networks (ESNs). We propose a methodology for collecting and indexing user-specific image associations to create personalized semantic graphs, while also using these to verify and test multimodal systems.
  ],
  authors: (
    (
      name: "Mike Zeng",
      organization: [University of Iowa],
      email: "mike-zeng@uiowa.edu"
    ),

  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)


= Introduction
Emergent semantics, as defined by Santini et al, refer to the subjective data generated when a user interacts with a piece of media @santini1999user. Rather than objective descriptions, this concept captures the user's internal response: does the image feel "warm"? Is it perceived as "pretty"?. Crucially, this form of semantics is highly dependent on the persona of the end user.

To operationalize this concept, Rahul et al formalized emergent semantics into a network structure @singh2011characterization. This formalization allows for the structured representation of subjective associations as a graph, enabling the quantification of relationships that were previously considered purely qualitative.

In this project, we utilize Emergent Semantic Networks (ESNs) for indexing and retrieval. We demonstrate a framework for creating personalized text-image categorization and show how these structures can be used to bias Large Multimodal Model (LMM) responses via inverted-semantic Retrieval Augmented Generation (RAG) @lewis2020retrieval.

= Methodology
Given a set of users, we construct our RAG system as follows:

== Emergent Semantic Network Collection
Prior to conducting any analysis, we first develop and implement the survey platform for collecting the ESNs.
#figure(
  image("images/interface.png"),
  caption:[Three-step survey user interface with 30 steps, one for each image. This survey took on average \~10 minutes on average.],
)
=== Image Collection Step 
Thirty images were manually taken in Iowa City at various times. These image timestamps range from May 2023 to October 2025.
#figure(
  image("images/imagecollection.png"),
  caption:[The collection of Iowa City photos],
)


=== Image Relationship Matching
Next, the user is provided an image on the left panel. On the right panel, 29 other images are rendered. The user is to select which images they think are related to the left panel image. Note: if two images have already been described as related, the other one won't appear.
#figure(
  image("images/imagelabel.png"),
  caption:[Interface for selecting related images],
)

=== Image Relationship Description
The user is taken to the next page, in which they are then asked to describe what makes them associate the image on the left with the images on the right. The reason this is an entire page on its own is that our system uses edge weights; therefore, it is very important for the responses here to be descriptive and of high quality.
#figure(
  image("images/imagerel.png"),
  caption:[Interface for image edge labeling],
)


=== Graph Submission
The website walks the user through relationship matching + description for each of the 30 images. Finally, it allows the user to copy the TSV of the set in the form of [{Image 1, Image 2, edge weight}]. 
#figure(
  image("images/esn-render.png"),
  caption:[A visualization of an ESN graph using NetworkX and Plotly.],
)

=== Response Collection and Annotation
The survey is hosted on the data site. A Google Form is sent through a message, containing a link to the data site as well as a place to submit the response. 

The Google Form is then sent over various platforms. Two in-state undergraduate University of Iowa students were emailed and filled out the form, as well as four randomized subjects from SurveyCircle, three of whom were from the USA but did not know where Iowa City was, and one from the United Kingdom who did not know where Iowa was.

Note: While there are more biases introduced this method (e.g, student vs online) and many other possible confounding variables, we expect the "student" parameter will control enough to cluster the graphs differently enough. One should expect to see a lot less similarity between the online graphs, but the hope is that they are still closer to each other than to the student graphs.

=== Notable Limitations
It is notable that 30 images is a small set. However, the primary issue was that with more than 30 images, it was difficult to get a significant amount of responses. Each image expands the survey length by a factor of $N$ because it adds an additional step to the survey and another image to compare to, which increases the risk of low-quality edges (which we need to minimize) given our limited budget for surveying. The biggest risk with a small set in this scenario is that the results of the system are not generalizable to more complex image sets, such as a user's general Google Drive, or other city locations such as Chicago. This is a good area for future development.

It is also notable that a set of 4 is a very small user set. We hope to attain more data in the future to obtain more convincing evidence, but scaling the users familiar with Iowa City was difficult.

Once we obtain our set of 8 ESNs, we build our retrieval system. We start with a single graph $G_i$.

== ESN Graph Encoding with Clustering 
Let $G_i$ denote a graph represented as TSV. That is, it is a list $[{I_1, I_2, E_i},...]$, where $I_i$ denotes image I and $E_i$ denotes the edge weight. To reduce duplication of our edges, we cluster our edge weights using Agglomerative Clustering with a distance threshold of 0.4. Here, we use Cosine Similarity as the distance function on text embeddings generated by All-MiniLM-L6-v2 @simoulin2021train to output a clustering that looks like $[[E_1, E_2], [E_3], [E_4, E_5, E_6]]$. For each cluster, we iterate through all edges and create an image set $C_i$. An image $I_1$ is in this set if there exists a row (${I_1, I_2, E_i}$) in the graph such that $E_i in E[C_i]$, where $E[C_i]$ denotes the edges in that cluster. Then, we use the first value of $E[C_i]$ as the label. Ultimately, this gives us a dict of ${T_i, [I_1, I_2...]}$. $T_i$ is an edge weight (text) and the key of the dictionary, while $[I_i]$ is the value.

#figure(
  image("images/trailcondense.png"),
  caption:[Trail subgraph => {"trail", [img1, img2, img3]}],
)

For example, if we have clusters [["tree", "tree1"], ["dog", "animal"]], and TSV = [[img1, img2, "tree"], [img3, img5, "tree1"], [img5, img6], "dog"], then we will create a dict {"tree":[img1, img2, img3, img5], "dog":[img5, img6]}. 

Note: clustering is introduced here to prevent the retrieval from being fragmented. Some users may write the semantics differently when they encounter an image they missed or realize that it should be there. For example, when looking at two images of a road, the user might type "Road", and then when they see two images of roads again that they missed, they might type "Car Lane

As for the distance threshold = 0.4, we found this the most optimal for merging two very similar ideas (flat lot, flat plane), but also not merging of different ideas that appear to be similar (tree VS tree with lights), which is why we selected tr = 0.4 for most of the merging work.


== Retrieving via Inverse Semantics
We apply concepts of Inverse Indexes from search engines @influxdata_inverted_index to create a framework that allows one to input a general query $Q$ (text) and obtain a set of images $I$ on a set of clusters $C$. First, we use All-MiniLM-L6-v2 to generate a text embedding for $Q$. Then, we let $i$ denote the argmax of the cosine similarity of $(Q, T)$, where $T$ is the set of labels (keys of the dictionary). We return the image set $C_i$ to the query, as well as the CS(Q, T) value itself. Our function is defined as follows: query($Q, G_i$) => $C_i$, $S$.

#figure(
  image("images/images.png"),
  caption:[Query "Downtown" returning the images associated with the cluster with label "City"],
)

== RAG-form Idea Generation
Finally, the last step is to integrate our retrieval with Large Multimodal Models (LMM). We have an interface for single-turn LLM conversation, where the user asks a query $Q$ on a graph $G_i$ and gets an output $O$.

On user query submission, the system utilizes the $Q$="What is a city?" to run the inverse semantic search on the specific graph $G_i$. The search returns an image set $I$, and the highest matching $T_i$ label. We inject into the LMM context "You associate $T_i$ with the attached images $I$. Q".

For example, suppose we have the retrieval from Figure 8. We ask $Q$="What does a city mean?". The model will then be "You associate downtown with the attached images. What does a city mean?"
#figure(
  image("images/ragdemo.png"),
  caption:[Query "Downtown" returning the images associated with the cluster with label "City"],
)
= Evaluation Framework and Goal Objectives
Testing multimodal systems remains a difficult challenge. One known problem is the Language Prior Problem @goyal2016making, where LLMs can learn ground truth from text alone. For example, if I ask the question "What color is the apple?", then the LMM can answer it is red without even needing to read the image.

Furthermore, there are no ground truths for image properties. For example, depending on the user annotating an image, a picture of an apple can be classified as "disgusting" or "tasty". It follows that its relevance to the word search is dependent as such, which invalidates the use of F1 scores.

Instead, we use of Emergent Semantics to gives us capacities for testing and evaluating our system on images.
Emergent semantics is largely an image-specific property that we predict will be affected by the personality of the end users. Therefore, if the graphs of the users familiar with Iowa City are more similar to each other than the users not familiar with Iowa City, then we have a functioning retrieval system that is clearly able to learn and express image semantics collected from user groups. We aim to test if the system can express bias text and do it in a controlled manner (follows ESN differences).

=== Objective 1: Graph Comparison Function
We design and propose a quantifiable metric for measuring initial graph differences. We propose using the *Jaccard Index*, which will be defined as follows: Let an edge be a pair $[I_i, I_j] in G_i$, where $I_i$ and $I_j$ denote the first two columns of the ESN TSV file, and $G_i$ denotes which file. Let $E_1$ be the set of edges in $G_i$, and $E_2$ be the set of edges in $G_j$. Then the Jaccard Index is simply an intersection over union defined as follows: 

$
(E_1 inter E_2) / (E_1 union E_2)
$

For example, suppose we have $G_1 = [[I_1, I_2], [I_3,I_4]]$, and $G_2 = [[I_1, I_2], [I_2, I_3]]$. Then our Jaccard Index is $1/3 = 0.33$.

Because our graph uses a fixed set of nodes (N=30) and the related node structure is the only thing different, Jaccard Index for comparing graphs in this manner works sufficiently well. Values closer to 1 mean better similarity, and values closer to 0 mean no similarity. Note: All image names are alphabetically sorted to prevent an edge $[I_1, I_2]$ from being different than $[I_2, I_1]$.

Now note, we are also interested in finding if the 4 users familiar with Iowa City are distinct from the 4 users not from Iowa City. To do this, we utilize ANOSIM (Analysis of Similarities). Let $F$ and $U$ denote two sets (size N=4) of ESN graphs.

1. Construct a distance matrix with 8 columns and rows. Lay each $G_i in (F union U)$ as the label for each row and column. Essentially, we'll have a cross matrix with labels [$F_1, F_2, F_3, F_4, U_1, U_2, U_3, U_4$].
2. The cell entry of (i, j) is defined as Jaccard($G_i, G_j$).
3. We sort the distances in the matrix into three categories: [$F_i$ to $F_j$], [$U_i$ to $U_j$], [$F_i$ to $U_j$].
4. Sort these distances from largest to smallest. All distances [$F_i$ to $U_j$] are considered CROSS while all other distances are considered GROUP.
5. Compute the mean rank of CROSS, mean rank of GROUP.
6. Compute the R statistic as defined as

$
  R = (mu_("CROSS")  - mu_("INTER"))/(N)
$
For example, A = [1, 2, 3], B = [10, 11, 12] D(a, b) = |a-b| \
Dist = AD[1, 1, 2], BD [1, 1, 2], ABD = [9, 10, 11, 8, …] \
Ranked dist = 11, 10, 10, … 2, 1, 1, 1, 1. Assign each in order 1-15 respectively. The average rank for ABD is 4 and 6, giving us an R of 1. Notice: closer to 1 implies complete separation, closer to 0 implies no separation.


To determine the significance of the R statistic, we use a permutation test with n=[5,5] (note the mismatch; these thresholds are statistically significant when survey size is larger), R. The following thresholds are considered strong:
- 0.01 p-value needs $R>0.9$ (extremely strong separation)
- 0.05 p-value needs $ R>0.7$ (strong separation)
- 0.1 p-value needs $R>0.5$ (slight separation)
- Sub 0.5 R-statistic yields no meaningful separation.

As such, these are the thresholds for separation strength.

Now that we have selected methods for analyzing base ESN graph differences, we will look at evaluation of applications.


=== Objective 2: Image Set Retrieval Evaluation

We now aim to develop metrics for the inverse semantic retrieval phase as well as the RAG-form generation.

For the inverse semantic phase, we have a trickier issue comparing $G_i$ to $G_j$ because the edge weights are almost completely different.

Decisions: 
1. Use the union of possible edge weights VS use the intersection of edge weights. Union will penalize the graphs for edge weights that are not shared, while intersection focuses more on how the same edge is defined differently. In the ESN graphs, we observed that no graph shared a significant amount of edge weights, even between familiar users. Therefore, it is more compelling to use intersection, as all union data will get skewed by the terms that do not exist in the other (gets a Jaccard of 0). We will discuss this more in results.
2. Because we retrieve at the edge level, we will have multiple Jaccard scores for a single comparison of Graph $G_i$ to $G_j$. We need to aggregate these scores by using either mean or median. We use mean because it is desirable for the similarity score to be skewed up by familiar edges. E.g, a set of [0.2, 0.3, 0.5, 0.9, 0.9] should be skewed by the 0.9s, and not ignored.

We build an intersection of our 2 graphs by doing as follows: 
1. Take both the edge weights, concat them into a single set $U$.
2. Agglomerative clustering on set U, distance threshold of 0.4.
3. For each weight $w in U$, run query($G_i, w$) and query($G_j$, w). If we run cosine similarity ($w$, $T_i$) and CS($T_j$) and get values greater than 0.3 for both, we save the query and the results to an entry ${w, C_i, C_j}$, where $C_i$ and $C_j$ represent the images returned by query. 
4. We compute Jaccard($C_i, C_j$), and append it to the set J.
5. At the end, we compute the mean value of J.

=== LMM RAG Integration Measurement

Due to the cost of running LMMs, we select the following edges to generate a response: ["bridge", "city", "library", "Iowa Memorial Union (IMU)", and "Seamans Engineering"]. The former three are the most common edges for all graphs, the latter 3 are the most common for the familiar graphs.

We run the RAG with Qwen3-VL-8B @bai2025qwen3 and use cosine similarity on (All-MiniLM-L6-v2(graph1-output), All-MiniLM-L6-v2(graph2-output). The prompt here is: ["What do you see in the attached images? Describe in detail, as specific to the image as possible. A user has related these details to {label}. Based on that, what do you think are landmark features of {edge}? Output in a list. "]. This forces the model to think about the image features (prompted image CoT) and use them.

= Results
== RQ1: Do Emergent Semantic Network Graphs Vary between Personalities?
#figure(
  image("images/jcesnd.png"),
  caption:[Jaccard index heatmap for similarity of familiar (F) to unfamiliar (U), R=0.781. We construct the distance matrix for the two sets. Green means the two graphs are more similar, while red means that they are not.
],
)
The first note is that the 4x4 matrix of the lower left corner (inter-familiar persona comparisons) is largely green, sharing more than 30% similar edges. The 4x4 matrix of the upper right corner (inter-unfamiliar persona comparisons) is mostly orange/yellow, implying the unfamiliar people with Iowa City vary more. However, the 4x4 of the upper left corner (CROSS persona comparison) is largely orange/red, with one outlier with F1 and U2.

=== Takeaway
The calculations yield a p-value below 0.05 significance, implying strong significance in the group separations. With this basic analysis, we make the following claim: There is a strong ability for the ESNs to capture persona differences. We see that the graphs generated by users familiar with Iowa City are significantly closer to each other than the graphs generated by those not from Iowa City.

=== Analysis on Features
The level of variation in users unfamiliar with Iowa City is much larger than those familiar with Iowa City. This makes sense, as users more familiar with Iowa City are likely to be a more fixed set of personalities (it is a stronger selection filter) as compared to taking completely random people around the globe. However, even with all of that, the maximum similarity is about 0.5. This suggests that ESNs can vary extremely widely even fixing a single variable, implying there are a handful of confounding variables to analyze. 

However, the results of this do make sense because even though people are unfamiliar with Iowa City, they should react to images a certain way. For example, if they don't know what the Iowa Memorial Union is, a consistency between IMU and a building that looks like IMU would likely exist. Meanwhile, someone familiar with the IMU may link it to studying and never care about the other building. Then here, the difference between those unfamiliar is not large, and neither is the difference between those who do know the IMU, but there is a large, consistent difference between those familiar and those unfamiliar.

#figure(
  image("images/esnvis.png"),
  caption:[Left: Unfamiliar with Iowa City | Right: Familiar with Iowa City. The graphs for those familiar with Iowa City typically are more structured, forming "blobs" around certain concepts. Meanwhile, the unfamiliar ESN graphs typically generate circular graphs, largely because of how "genericness" applies to this context. For example, the Seamans Engineering Center may be very specific to a person who walks by it everyday, so they see it as very distinct. The sidewalks get labeled with it, the signs, surroundings, etc. Then when they re-encounter the images, they do not match it to visually similar images just because of the association.
],
)


Overall, we found a significant difference between users familiar with Iowa City and users not familiar with Iowa City, and we expect that any system that correctly captures the ESNs should exhibit this difference as well.

== RQ2: Can we build applications that can utilize the emergent semantics?
#figure(
  image("images/graphdiff.png"),
  caption:[Jaccard index heatmap for returned images of familiar (F) to unfamiliar (U) with mean. R=0.792. The lower left 4x4 is more green than the other regions, such as the CROSS region (upper left) being red, and the upper right 4x4 being orange. Visually, $F$ and $G$ do appear to be more clustered.],
)


#figure(
  image("images/uniongd.png"),
  caption:[Jaccard index heatmap for returned images of familiar (F) to unfamiliar (U) with mean, edge UNION. This will query the image search using every edge weight, not just one that exists in both. This results in exotic behavior, such as a query for "trees" returning images associated with "bees", resulting in a low R score (0.5) and high differences.],
)

Note: while it is important that this behavior exists (e.g., $F_1$ and $F_2$ label things differently; should get penalized), this also masks any difference because the ESN graphs started off different (best case graphs shared only 50% of edges). It is also important to remember that we are evaluating to see differences in behavior defined by both models (e.g., does a user think of different things on term "very dark"), not just pure term memory checks (does term exist in other). 

=== Takeaways
The calculation yields us a good p-value that there is separability between the image retrieval's system of the user semantics. Because of this, we make a claim: Our image retrieval tool is capable of capturing and also expressing differences in the ESNs at a small scale, and suggests that it is possible to build such a system.

=== Feature Analysis
What is particularly interesting is that F2-U1 were extremely different in structure (Fig 10), but very similar in query (Fig 13). 

It is also interesting to note that areas that were originally above a 0.33 were not affected much, while the areas that were around 0.3 were significantly more different. This is interesting because it suggests graph differences have a "filtering" threshold where the differences are penalized harsher. This makes sense though, as this incorporates the edge weights unlike the experimentation in RQ1.

=== Visual Analysis
#figure(
  image("images/genericsearch.png"),
  caption:[Image search across graphs with a generic search term "River". Note that the images returned by all graphs are similar, which makes sense due to how generic "River" is.],
)

#figure(
  image("images/f2u1.png"),
  caption:[Image search with F2 behaving like U1 by being searched "tree" where F2 doesn't have a good match for the term "tree". Notice how F1 and F2 are extremely different; F1 only focuses on the two trees that are illuminated in the night. Meanwhile, F2 returns a large corpus of documents like the unfamiliar graphs. This is likely because F2's labels tend to be more generic.],
)

#figure(
  image("images/specific.png"),
  caption:[Image search with "library", dividing F and U. Library is matched to all four F graphs, but none of the U graphs match it. As a result, the library term returns pretty similar images for the F graphs, but not for the U.],
)


It is notable that this verifies the issue with union edge testing. If this had been a test for union, all the U would be highly distinct from each other, which creates lots of noise. It is better to keep noise low and just use intersection.

Overall, this image search works well at expressing the differences in ESN graphs. In fact, user satisfication with these applications tend to be better than traditional methods (Appendix B). We now aim to translate this image search into tasks for LMMs.

== RQ3: Can Image Search be Translated to LMMs?
Here, we examine our selected edge weights and their ability to be expressed by the LMM.

#figure(
  image("images/lmm-library.png"),
  caption:[LMM output graph diff matrix on term "Library", R=0.52. Given varying image sets, the LMM output was clustered for the familiar graphs. However, it is not statistically significant; this demonstrates the limitation of the too-small set of users, LMMs add too much variance for this small set of data to be significant. However, this highlights how even though the image set being fed to the LMMs is significantly seperated for F and U graphs, the expression of these semantic features remains a challenge. 
],
)



#figure(
  image("images/IMUexpress.png"),
  caption:[LMM output graph diff matrix on term "Library", R=0.135. The term "IMU", even though it is a University of Iowa specific feature, is difficult to express. The R value is a lot lower, and does not get captured by RAG even though the returned images are very significant. Although, this could be caused by the fact IMU varies too much with non-Iowa City users.],
)

#figure(
  image("images/trailsim.png"),
  caption:[LMM output graph diff matrix on term "trail", R=-0.042. The LMM has no difference in the responses for the generic term trail. This makes sense, as our image set likely does not contain any information that a familiar user could ground the word trail to. This is interesting because it does tell us quite a lot about our image set, on how "library" > "IMU" > "trail" for quantifying Iowa City.
],
)

=== Takeaways
While we have great ability to capture the emergent semantics and express them through the retrieval, the usability of the retrieval information to an LMM exists but is more difficult. However, this allows us to determine which terms are more important to the dataset itself.

=== Analysis
The usability of the images is an interesting topic. It exists clearly, as some terms like "library" create some level of distinguishability between the familiar graphs. It is very possible that our data size is too limited here. 

=== Visual Analysis

#figure(
  image("images/liblargesimillegal.png"),
  caption:[LMM output of F1 vs U3, which are very similar (CS=0.73). The language prior issue is prominent here, where the LMM is extracting features from the bias of library without ever seeing a picture of a library. And also how different images are allowing it to get the right answer (outdoor plaza from tree image source).
],
)


#figure(
  image("images/liblargediff.png"),
  caption:[F1 and U2 having a larger difference (CS=0.4). Language prior issues/variance are not always the case. U2's library match did not return anything about a library, rather just rivers. This biases the LMM to focus on nature focus of a library instead, which causes it to be extremely different.
],
)


#figure(
  image("images/libflargesim.png"),
  caption:[LMM output comparing F1 to F2 matrix on term "Library", CS=0.8. When comparing graphs of familiar users, the generation of content is more similar (0.8 CS). This implies that F1 and F2 users are a lot more similar on their perspective of a library when expressed. Overall, the term library is expressed but very mildly. We find that while search is very excellent at capturing the differences, the ability to express the difference is a lot harder.
],
)


However, this may perhaps be useful at reverse analysis. We can determine which features are more unique to the set of familiar users.

= Future Work
Future work could expand the user sampling size. While it was difficult to do so in our case, it is certainly something feasible in the future. We also propose rigorous evaluation of multi-domain image collections, e.g., mixing Chicago with Iowa City and testing such.

= Conclusion
In this project, we propose Superblob, a system for expressing and capturing emergent semantics found in multimedia beyond text. This system can be used for various applications, such as semantic image clustering, image search, and RAG biasing. We have also proposed and showcased a method for testing and evaluating the performance of multimodal RAG by measuring how well they capture and express emergent semantic networks. Finally, we introduced the idea of pipelining user surveys to RAG bias, which is novel as ESN surveys can be designed to capture subtle differences in personality that is not easily observed or quantifiable.

Project Code: https://github.com/mzen17/Superblob/

Project Site: https://superblob.mzen.dev/


= Appendix

== Appendix A: Image Extraction in Comparison to CLIP

Here, we compare CLIP's ability to capture emergent semantics. We generate a graph with CLIP and create the edges as follows:

1. Use CLIP to compute all the image embeddings of the dataset.
2. For each image, compute CS($I_i$, I), comparing the image with all the other images.
3. Filter all images whose cosine similarity > 0.8.
4. Create an edge between those images.

We evaluate how similar this generated system is to any of the user-sampled graphs.

#figure(
  image("images/clipVSFU.png"),
  caption: [Graph table comparing Jaccard(CLIP, G)]
)

Here, we see that the Jaccard similarity of CLIP to any of the ESNs is extremely low. This implies that first of all, any graph generated using CLIP to try to capture emergent semantics does not match to anybody's graph, suggesting CLIP-based similarity does not align with human emergent semantic structure under our evaluation. In fact, it does not even fit the data well because the differences between it and U graphs and F graphs are not significantly different, despite that F and U are significantly different.

== Appendix B: User Data
#figure(
  image("images/responserate.png"),
  caption:[Chart showing response rates out of 10 sends at varying image counts. We find that N=30 gave us enough images so that there was differences in the graphs, but also just enough users to have some degree of significance. Surveys of length N>30 are unfeasible to run without paying the users.],
)

However, we decided to evaluate some application level usability of the ESNs. 3 familiar and 1 unfamiliar users were sampled and told to organize the 30 images into folders and compared the experience. Each user evaluates:
1. How intuitive is the system? E.g, how much thinking do you use when labeling the systems compared? A (1) means it feel unnatural, forcing logic and reasoning statements such as trying to express human language as code. (5) means it feels natural, e.g, telling what vibes one gets from a picture in free form language.
2. Ease of use. How familiar are you to this sort of method? How easy is it? (1) means it is very new to the user, (5) means that the actions are very easy to do (e.g, drag file across the screen).
3. Meaningful categories. How well do you think the categories/folders captures the meaning of the images? (1) means low amounts of folders, folders do not fully capture the meaning of images while (5) means not just many folders, but each folder seems to capture a new and interest aspect of the image.


#figure(
  image("images/psyctest.png"),
  caption: [Table comparing manual folder creation VS ESN labeling. We see that using ESNs to create these folders is highly intuitive, requiring much less thinking than manually creating folders. However, it takes longer and is less familiar to those creating folders, which gets it a lower ease of use score. However, we see the categories are much more meaningful]
)

The users were surveyed, the general consenses was that while the ESNs took longer and the interface was new to them, it was much easier to generate meaningful categories. The strongest difficulty for the users themselves was generating the set of folders, which required a lot of thinking and looking through all the images. The users concluded with that the ESNs provided a very natural interface for the creation and management of these folders.


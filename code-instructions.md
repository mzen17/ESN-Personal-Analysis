# How to run and setup the environment 
## A brief intro to the codebase.
workdata -- smaller variants of the photos for easier rendering

surveyapp -- Sveltekit with the survey app.

rag -- Applications for the ESNs
- demo.py | micro streamlit (deprecated with main dashboard) for showing image search
- diffg.py | graph difference functions, matrix generation
- graph.py | contains entity_collaspe (clustering) + query (maximal matching)
- system.py | working example code for Gemini API interactions
- rag-vis | micro streamlit (deprecated with main dashboard) for showing entity collaspe

imgtk -- tools for managing imagedata (full resolution, big photos)

analysis
- app.py | streamlit for showing CLIP generated space.
- bsl.py | Baseline CLIP generation ESN for graphs
- clip.py | CLIP space rendering algorithm
- ESN-vis.py | micro streamlit (deprecated with dasboard.py) for visualizing collected ESNs
- jaccard_matrix.py | generates the jaccard matrices for R calculations

dashboard.py -- Main dashboard for the interactive app using functions from the other code.

## Running the demo
First, run ```conda env create -f conda.yaml```.
Then, run ```streamlit run dashboard.py```, and the application will be open.

### Running the survey app
The survey app needs a JS environment. Run ```npm install```, then ```npm run dev``` to bring up the survey up.

Without a JS environment, it is much easier to just go to [https://superblob.mzen.dev/sv/](https://superblob.mzen.dev/sv/).

### Running the LLM 
It is easier to use [https://superblob.mzen.dev](https://superblob.mzen.dev) to run the LLM, as this is bootstrapped to a local Qwen3 server and is free to use, subject to reasonable rate limits.

Running gemini-run.py locally is significantly harder, since it runs using the local qwen server. However, one can go to the superblob website and test the prompts manually to see if they generate similar output.

All responses are saved and cached to data/gemini, even though the system uses Qwen3. 

## Recommendations
If it is not possible to run the code, either contact me at mike-zeng@uiowa.edu, or simply use the public demo on https://superblob.mzen.dev. Everything is accessible, except for the LLM usage (uses a basic password) to prevent DDOS.

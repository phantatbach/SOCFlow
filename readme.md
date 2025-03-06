This workflow is used to explore the second order dimensions for semantic analysis. It was developed within the [Nephological Semantics project](https://www.arts.kuleuven.be/ling/qlvl/projects/current/nephological-semantics) at KU Leuven by [Bách Phan-Tất](https://phantatbach.github.io) as part of his PhD and with the collaboration of Dirk Geeraerts, Dirk Speelman, Kris Heylen, Stefano De Pascale and Ángela Maria Gomez Zuluaga.

# Note:
- The results of the experiments show that choosing SOCs based on pure aggregated scores is arbitrary. So now I am developing a new approach of dimensionality reduction/feature selection. All the experiments are saved in the main notebook.

# Installation
# Use
- After running NephoSem, SemasioFlow and NephoVis, you should have a clear idea of what model you want to explore further.
- The input for this workflow would include:
    - The token in case you want to investigate the different parameter settings (e.g., time, time_no_det, time_win)
    - The name of the model of interest.

    - The dimensionality reduced .tsv output of the semclouds package (e.g., token_name.tsne.30.tsv)
    - the variables output of the semclouds package with the annotated information (e.g., token_name.variables.tsv)
    - The SOC matrix in the ouput folder (e.g., model_name.tcmx.soc.pac)

- token_coordinates:
    - First, run get_model_data to take the coordinates and other informations (e.g., HDBSCAN, annotations) of the tokens generated by your model of interest.
    - Then, run get_token_ids to visualise the tokens (basically it reproduces the output of level 3 in Nephovis but without the FOC words).
    - You can then select the token(s) of interest by Box selection.
    - You will then get a list of token_ids for SOC_Analysis

- SOC_Analysis
    - First, get the sub-matrix of the selected tokens using nephosem.
    - You could also filter by POS tags
    - Analyse the SOC dimensions.
        - There are 2 ways to analyse the clouds: sum and avg
            - sum will sum all the association scores of the SOCs
            - avg will sum all the association scores of the SOCs but then divided by the number of tokens as a way to normalise/for the prototypical instance
        - elbow_finder do either one of the 2 then find the elbow point.
        - soc_dist_vis visualises the 'distance' of the top n SOCs to the region.
        - get_context extracts the raw contexts of the tokens of the region for further analysis.

# Input folder:
- These files would need to be included in the 'input' folder    
    - The dimensionality reduced .tsv output of the semcloud package (e.g., lemma.tsne.30.tsv)
    - The distance matrix of the model under investigation (e.g., model.ttmx.dist.pac)
    - The variables output of the semcloud package with the annotated information (e.g., lemma.variables.tsv)
    - The SOC matrix in the output folder (e.g., model.tcmx.soc.pac)
    - The HDBSCAN output of semcloud if you plan to use it (lemma.rds)
- Running get_model_data would create another file (token-model.tsne.30.tsv) and save them to input/visualisation for later use.
    - You can either calculate HDBSCAN with your own parameters or
    - Use the available HDBSCAN results from semcloud
- The submatrix of only the tokens under investigation (using nephosem) must also be outputted here.

# Output folder:
- Top n SOCs and their association scores would be outputted here by running soc_dist_vis.
- You can also get the raw context for the tokens under investigation here by running get_context.

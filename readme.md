This workflow is used to explore the second order dimensions for semantic analysis. It was developed within the [Nephological Semantics project](https://www.arts.kuleuven.be/ling/qlvl/projects/current/nephological-semantics) at KU Leuven by [Bách Phan-Tất](https://phantatbach.github.io) as part of his PhD and with the collaboration of Dirk Geeraerts, Dirk Speelman, Kris Heylen, Stefano De Pascale and Ángela Maria Gomez Zuluaga.

# Installation
# Use
- After running NephoSem, SemasioFlow and NephoVis, you should have a clear idea of what model you want to explore further.
- The input for this workflow would include:
    - The token in case you want to investigate the different parameter settings (e.g., time, time_no_det, time_win)
    - The name of the model of interest.
    - The dimensionality reduced .tsv output of the semclouds package (e.g., token_name.tsne.30.tsv)
    - the variables output of the semclouds package with the senses annotated (e.g., token_name.variables.tsv)
    - The SOC matrix in the ouput folder (e.g., model_name.tcmx.soc.pac)

- First, run get_model_data to take the coordinates of the tokens generated by your model of interest.
- Then, run get_token_ids to visualise the tokens (basically it reproduces the output of level 3 in Nephovis but without the FOC words).
- By selecting a region of interests, you would get a list of tokens in that region.
- Get the sub-matrix of the selected tokens.
- Analyse the SOC dimensions.
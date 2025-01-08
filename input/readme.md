These files would need to be included in the 'input' folder    
    - The dimensionality reduced .tsv output of the semclouds package (e.g., token_name.tsne.30.tsv)
    - the variables output of the semclouds package with the senses annotated (e.g., token_name.variables.tsv)
    - The SOC matrix in the ouput folder (e.g., model_name.tcmx.soc.pac)

Running get_model_data would create another file (token-model_name.tsne.30.tsv) to be used later.

The submatrix of only the tokens under investigation (using nephosem) must also be outputted here.
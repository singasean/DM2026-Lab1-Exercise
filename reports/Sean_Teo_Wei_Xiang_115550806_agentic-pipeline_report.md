# Your report

First, I loaded the 20-Newsgroups text using `load_dataset_tool`. The dataset consisted of documents distributed across four distinct categories: `alt.atheism`, `soc.religion.christian`, `comp.graphics`, and `sci.med`.

Following the data discovery, I granted the agent access to the custom student tool workspace. I constructed the initial high-dimensional feature space by calling `build_dtm_tool` on the text column, which generated a Document-Term Matrix representing term frequencies across the loaded dataset.

To reduce noise and optimize computational efficiency, I applied the `variance_filter_tool` with a threshold of 0.01. This successfully removed near-constant and low-variance terms that hold little discriminative value for text classification.

Finally, I projected the filtered matrix into a 2D space for visualization using the `reduce_dimensions_tool`. The resulting scatter plot demonstrates the clustering tendencies of the documents based on their original categories.

![Dimension Reduction Plot](../plots/Sean_Teo_Wei_Xiang_agentic-pipeline_plot_reduce_dimensions_6)
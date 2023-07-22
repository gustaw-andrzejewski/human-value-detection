# human-value-detection
This project presents an approach of using Graph Neural Networks
(GNNs) for the multi-label classification task of detecting human values
from argumentative texts. Unlike traditional Natural Language Process-
ing (NLP) models, which operate on linear sequences of text, this approach
leverages the interconnections between different arguments and encapsu-
lates the semantic richness of the argumentative texts into node features
using BERT embeddings. The data was structured into an undirected
graph, with nodes representing arguments and edges established based
on the Jaccard similarity of arguments’ values. Experimentation revealed
that graph-based models could potentially be useful for such a task. How-
ever, as the project was just an investigation of graph-based methods for
multi-label text classifications using relationships between the texts and
the approach used for connecting the nodes was based on the labels, it has
limitations associated with the model’s reliance on existing node connec-
tions for new argument classification so ideas for future work include an
iterative refinement process for classification and consideration of alter-
native features for establishing node connections or a different approach
to constructing arguments connections.

The project was based on the following task:
https://touche.webis.de/semeval23/touche23-web/

Here are the results I have obtained:
https://api.wandb.ai/links/gustaw_andrzejewski/4qn9rxj8

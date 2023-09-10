# human-value-detection
This project presents an approach of using Graph Neural Networks
(GNNs) for the multi-label classification task of detecting human values
from argumentative texts. Unlike traditional Natural Language Process-
ing (NLP) models, which operate on linear sequences of text, this approach
leverages the interconnections between different arguments and encapsulates 
the semantic richness of the argumentative texts into node features
using BERT embeddings. The data was structured into a directed
graph, with nodes representing arguments and edges established based
on the cosine similarity of arguments’ values. Experimentation revealed
that graph-based models could potentially be useful for such a task, but don't seem to 
exceed the results obtained by regular large language model fine-tuning. However, 
as the project was just an investigation of graph-based methods for
multi-label text classifications using relationships between the texts and
the approaches avilable in the literature show that utilizing the label connections
seems to bring even more promising results, perhaps combining both of these approaches would
yield more satisfactory results.

The project was based on the following task:
https://touche.webis.de/semeval23/touche23-web/

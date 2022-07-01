# APT-KGL: An Intelligent APT Detection
System Based on Threat Knowledge and
Heterogeneous Provenance Graph Learning

## Abstract
APTs (Advanced Persistent Threats) have caused serious security threats worldwide. Most existing APT detection systems
are implemented based on sophisticated forensic analysis rules. However, the design of these rules requires in-depth domain
knowledge and the rules lack generalization ability. On the other hand, deep learning technique could automatically create detection
model from training samples with little domain knowledge. However, due to the persistence, stealth, and diversity of APT attacks, deep
learning technique suffers from a series of problems including difficulties of capturing contextual information, low scalability, dynamic
evolving of training samples, and scarcity of training samples. Aiming at these problems, this paper proposes APT-KGL, an intelligent
APT detection system based on provenance data and graph neural networks. First, APT-KGL models the system entities and their
contextual information in the provenance data by a HPG (Heterogeneous Provenance Graph), and learns a semantic vector
representation for each system entity in the HPG in an offline way. Then, APT-KGL performs online APT detection by sampling a small
local graph from the HPG and classifying the key system entities as malicious or benign. In addition, to conquer the difficulty of
collecting training samples of APT attacks, APT-KGL creates virtual APT training samples from open threat knowledge in a
semi-automatic way. We conducted a series of experiments on two provenance datasets with simulated APT attacks. The experiment
results show that APT-KGL outperforms other current deep learning based models, and has competitive performance against
state-of-the-art rule-based APT detection systems.

## The code framework
- Create_graph. py is the script to build the Provenance Graph;
- Data_prepare. py is the script to build the heterogeneous graph;
- Meta. Py is the meta-path setting;
- Model_hetero.py is a model for pre-training; Run_model_for_graphsampling_v2. py is a subgraph sampled and trained.
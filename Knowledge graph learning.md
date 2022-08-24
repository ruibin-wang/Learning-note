# Knowledge representation learning


<font size=3> **Definition** </font>

    Knowledge graph embedding is a approach to ***transform Knowledge Graphs*** (nodes, edges and feature vectors) ***into a low dimensional continuous vector*** space that preserves various graph structure information, etc.  



Then this note will present four section about the knowledge representation learning. including representation space, scoring function, encoding models and auxillary information.

**A map of all algorithms[9]:**


<center class='half'>
<img src=./Pictures/KG_embedding/figure9.png > Figure 1
</center>

---


<font size=3>**1. Representation space (representing entities and relations)**</font></end>

including the  vector, matrix and tensor space, also the  complex vector space, Gaussian space and manifold.

The embedding space should follow three conditions, i.e., differentiability, calculation possibility, and definability of a scoring function

<center class='half'>
<img src=./Pictures/KG_embedding/figure10.png> Figure 2
</center>



* Point-wise space( Euclidean space):

    As showned in the Figure 2(a) above.

    * **TransE**[1]  represents entities and relations in d-dimension vector space, $ h,r,t \in R^ d $


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure1.png > Figure 3       
        </center>


        Score function: 

        $$-||h + r - t||_{1/2}$$ 

        ```python
        def TransE(self, head, relation, tail, mode):
            if mode == 'head-batch':
            score = head + (relation - tail)
            else:
            score = (head + relation) - tail

            score = self.gamma.item() - torch.norm(score, p=1, dim=2)
            return score
        ```

        <font color=FireBrick>*This model fails in case of the one to many relation and many to many relation.* To overcome this deficit new model TransH is proposed.</font>



    * **TransH[2]**

        this medhod is trying to solve the limitations of TransE. Score function is similar to the TransE. 


        As shown in the Figure 3(b), the vectors $h$ and $t$ are projected in the relation hyperplane.

        Scoring function: 
        
        $$-||(h-w_r^Thw_r)+r-(t-w_r^Ttw_r)||_2^2$$


        


        To track the problem of insufficiency of a single space for both entities and relations. <font color=FireBrick>TransR[3] projected entities into relation space.</font>


    * **TransR [3]**

        TransR models entities and relation in different embedding space.Each entity is mapped into relation spaceby a projection matrix $ M_r \in R^ {d*d*d} $.

        Score function:  $ -||M_rh + r - M_rt||_2 $

        


    * **Neural Tensor Network (NTN) [10]**


* Complex vector space:







* Guassian distribution:




* Manifold and group







<font size=3> **2. Scoring function (measuring the plausibility of facts)**</font>




<font size=3> **3.  Encoding models (modeling the semantic interaction of facts)**</font>




<font size=3> **4.  Auxiliary information (utilizing external information)** </font>









---


[1] Antoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. 2014. *A semantic matching energy function for learning with multi-relational data: Application to word-sense disambiguation.* Mach. Learn., 94(2):233–259.

[2] Zhen Wang, J. Zhang, Jianlin Feng, and Z. Chen. 2014. *Knowledge graph embedding by translating on hyperplanes.* In AAAI

[3] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. *Learning entity and relation embeddings for knowledge graph completion.* In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, AAAI’15, page 2181–2187. AAAI Press.

[4] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. 2019a. *Rotate: Knowledge graph embedding by relational rotation in complex space.* CoRR, abs/1902.10197.

[5] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, and Jie Wang. 2019. *Learning hierarchy-aware knowledge graph embeddings for link prediction.* CoRR, abs/1911.09419.

[6] Maximilian Nickel et al. *“A Three-Way Model for Collective Learning on Multi-Relational Data”* international conference on machine learning (2011): n. pag.

[7] Alberto Garc ́ıa-Dur ́an, Antoine Bordes, and Nicolas Usunier. 2014. *Effective blending of two and threeway interactions for modeling multi-relational data.* In Machine Learning and Knowledge Discovery in Databases, pages 434–449, Berlin, Heidelberg. Springer Berlin Heidelberg.

[8] Yinwei Wei, Xiangnan He, Xiang Wang, Richang Hong, Liqiang Nie, and Tat Seng Chua. 2019. *MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video.*MM 2019 - Proc. 27th ACM Int. Conf. Multimed., pages 1437–1445.

[9] Ji, S., Pan, S., Cambria, E., Marttinen, P. and Philip, S.Y., 2021. *A survey on knowledge graphs: Representation, acquisition, and applications.* IEEE Transactions on Neural Networks and Learning Systems, 33(2), pp.494-514.

[10] Richard Socher, Danqi Chen, Christopher D. Manning and Andrew Y. Ng. “Reasoning With Neural Tensor Networks for Knowledge Base Completion” neural information processing systems (2013): n. pag.







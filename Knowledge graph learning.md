# Knowledge Representation Learning


<font size=3> **Definition** </font>

    Knowledge graph embedding is a approach to ***transform Knowledge Graphs*** (nodes, edges and feature vectors) ***into a low dimensional continuous vector*** space that preserves various graph structure information, etc.  



Then this note will present four section about the knowledge representation learning. including representation space, scoring function, encoding models and auxillary information.

**A map of all algorithms[9]:**


<center class='half'>
<img src=./Pictures/KG_embedding/figure9.png >
</center>

<p align='center'> <font color=DarkOliveGreen>Figure 1 </font></p>

---


<font size=3>**1. Representation space (representing entities and relations)**</font></end>

including the  vector, matrix and tensor space, also the  complex vector space, Gaussian space and manifold.

The embedding space should follow three conditions, i.e., differentiability, calculation possibility, and definability of a scoring function

<center class='half'>
<img src=./Pictures/KG_embedding/figure10.png>
</center>

<p align='center'> <font color=DarkOliveGreen> Figure 2 </font></p>


* Point-wise space( Euclidean space):

    As showned in the Figure 2(a) above.

    * **TransE**[1]  represents entities and relations in d-dimension vector space, $ h,r,t \in R^ d $ it is shown in the Figure 3(a)


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure1.png >    
        </center>

        <p align='center'> <font color=DarkOliveGreen> Figure 3 </font></p>


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

        TransR models entities and relation in different embedding space. Each entity is mapped into relation space by a projection matrix $ M_r \in R^ {d \times d \times d} $. As shown in Figure 3(c)

        Score function:  
        
        $$ -||M_rh + r - M_rt||_2 $$

        


    * **Neural Tensor Network (NTN) [10]**

        NTN models entities across multiple dimensions by a bilinear tensor neural layer.  The relational interaction between head and tail $h_T \hat M t$ is captured as a tensor denoted as $\hat M \in R^{d \times d \times d}$

        As shown in the Figure 4(a)


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure11.png >    
        </center>

        <p align='center'> <font color=DarkOliveGreen> Figure 4 </font></p>

    

    * **Hierarchy-Aware Knowledge Graph Embedding （HAKE） [11]**

        Without using the Cartesian coordinate system, HAKE captures semantic hierarchies by mapping entities into the polar coordinate system, i.e., entity embeddings $e_m \in R^d$ and $e_p \in [0,2 \pi)^d$ in the modulus and phase part, respectively.


        As shown in Figure 4(b). *The entity with smaller radious belongs to higher up in the hierachy.*  The angle between them represents the variation in the meaning. This model has two components, one to map the modulus and the other one to map the angle. 


        Score function:

        distant function only consider modulo part:    $d_{r,m} = ||h_m \circ r_m - t_m||_2$

        phase part: $\lambda||sin((h_p+r_p-t_p)/2)||_1$
        


        Mapping the entity in the polar coordinate space.

        $$-||h_m \circ r_m - t_m||_2 - \lambda ||sin((h_p+r_p-t_p)/2)||_1$$   

        $$h_p, r_p, t_p \in[0,2\pi)^d$$



    * **TransD [12]**

        Scoring function: 

        $$-||(w_rw_h^T + I)h + r - (w_rw_t^T + I)t||_2^2$$
        $$h,t,w_h,w_t \in R^d \qquad r, w_r \in R^k$$


    * **TransA [13]**

        Scoring function:

        $$(|h+r-t|)^TW_r(|h+r-t|)$$
        $$h,t, r \in R^d, M_r \in R^{d \times d} $$


    * **TransM [14]**

        Scoring function:

        $$-\theta _r ||h+r-t||_{1/2} $$


   

        <font color=red> *Conclusion* </font>

         TransE by assuming that the added embedding of $h + r$ should be close to the embedding of $t$ with the scoring function. 
         
         TransH projects entities and relations into a hyperplane, TransR introduces separate projection spaces for entities and relations, and TransD constructs dynamic mapping matrices $M_{rp}=r_ph_p^T + I$ and $M_{rt} = r_pt_p^T + I$ by the projection vectors $h_p, tp, rp \in R^n$. 
         
         By replacing Euclidean distance, TransA  uses Mahalanobis distance to enable more adaptive metric learning.
        



    





* Complex vector space: 

    $h,t,r$ can also be presented in the complex space. As is shown in the Figure 1(b), vectors can capture both symmetric and antisymmetric relations.  
    
    $h, t, r \in C^d$,  $h = Re(h)+i Im(h)$.
    
    
    
    * **ComplEX [15]**
        
        Hermitian dot product is used to do composition for relation, head and the conjugate of tail.

        scoring function:

        $$Re(<r,h, \overline t>) = Re(\sum_{k=1}^K r_kh_k \overline t_k)$$
        $$h, t, r \in C^d$$


        ```python
        def ComplEx(self, head, relation, tail, mode):
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_relation, im_relation = torch.chunk(relation, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                score = re_head * re_score + im_head * im_score
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                score = re_score * re_tail + im_score * im_tail

            score = score.sum(dim = 2)
            return score
        ```        


    * **RotatE[4]**

        the relation from head to tail is modeled as rotation in a complex plane, It is motivated by Euler's identy $e^{i \theta} = cos(\theta) + i*sin(\theta)$.

        The scoring function is measures the angular distance.

        $$d_r = ||h \circ r-t ||_2$$
        
        $$h,t,r \in C^d$$


        Where $h, r, t$ is the k-dimension embedding of the head, relation and tail,restricting the $|r_i| = 1$. <font color=green> relations are in the unit circle and $\circ$ represents element-wise Hadmard product.</font>    




        ```python
        def RotatE(self, head, relation, tail, mode):
            pi = 3.14159265358979323846
            
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

            #Make phases of relations uniformly distributed in [-pi, pi]

            phase_relation = relation/(self.embedding_range.item()/pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)

            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                re_score = re_score - re_head
                im_score = im_score - im_head
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                re_score = re_score - re_tail
                im_score = im_score - im_tail

            score = torch.stack([re_score, im_score], dim = 0)
            score = score.norm(dim = 0)

            score = self.gamma.item() - score.sum(dim = 2)
            return score

        ```


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure4.png width = 70%>
        </center>

        <p align=center> <font color=DarkOliveGreen> Figure5 </font> </p>



    * **QuatE [16]**

         QuatE extends the complex-valued space into hypercomplex $h, t, r \in H^d$ by a quaternion $Q = a + bi + cj + dk$ with three imaginary components, where the quaternion inner product. 
         
         the Hamilton product $h \bigotimes r$, is used as compositional operator for head entity and relation.


    

    <font color=red> *Conclusion* </font>

    With the introduction of the rotational Hadmard product in complex space, RotatE can also capture inversion and composition patterns as well as symmetry and antisymmetry.


    QuatE uses Hamilton product to capture latent inter-dependencies within the four-dimensional space of entities and relations and gains a more expressive rotational capability than RotatE.




    
    







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

[11] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, & Jie Wang (2019). Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction national conference on artificial intelligence.

[12] Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, & Jun Zhao (2015). Knowledge Graph Embedding via Dynamic Mapping Matrix international joint conference on natural language processing.

[13] Han Xiao, Minlie Huang, Yu Hao, & Xiaoyan Zhu (2015). TransA: An Adaptive Approach for Knowledge Graph Embedding.. arXiv: Computation and Language.

[14]  Fan, M.; Zhou, Q.; Chang, E.; and Zheng, T. F. 2014. Transition-based knowledge graph embedding with relational mapping properties. In Proceedings of the 28th Pacific Asia Conference on Language, Information, and Computation, 328–337.

[15] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier, & Guillaume Bouchard (2016). Complex embeddings for simple link prediction international conference on machine learning.

[16] Shuai Zhang, Yi Tay, Lina Yao, & Qi Liu (2019). Quaternion Knowledge Graph Embeddings neural information processing systems.






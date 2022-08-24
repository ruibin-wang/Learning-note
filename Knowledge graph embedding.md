# Knowledge representation learning models

1. Definition

    Knowledge graph embedding is a approach to ***transform Knowledge Graphs*** (nodes, edges and feature vectors) ***into a low dimensional continuous vector*** space that preserves various graph structure information, etc.  


    ***A map of all algorithms[9]:***

    <img src=./Pictures/KG_embedding/figure9.png>


2.  Knowledge representation learning models

    * Translational distance models

        <font color=DarkGreen>*Use distance-based measures to generate the similarity score for a pair of entities and their relationship.*</font>

        * TransE[1]

            Score function: $-||h + r - t||_{1/2}$ 

            ```python
            def TransE(self, head, relation, tail, mode):
                if mode == 'head-batch':
                score = head + (relation - tail)
                else:
                score = (head + relation) - tail

                score = self.gamma.item() - torch.norm(score, p=1, dim=2)
                return score
            ```


            <img src=./Pictures/KG_embedding/figure1.png width = 35%>

            <font color=FireBrick>This model fails in case of the one to many relation and many to many relation. To overcome this deficit new model TransH is proposed.</font>


        * TransH[2]

            this medhod is trying to solve the limitations of TransE. Score function is similar to the TransE. 

            <img src=./Pictures/KG_embedding/figure2.png width = 35%>

            The vectors h and t are projected in the relation hyperplane.


        

        * TransR[3]

             TransR models entities and relation in different embedding space.Each entity is mapped into relation space.

             Score function:  $ -||M_rh + r - M_rt||_2 $

             <img src=./Pictures/KG_embedding/figure3.png width = 50%>



        * RotatE[4]

            the relation from head to tail is modeled as rotation in a complex plane, It is motivated by Euler's identy.

            The scoring function is measures the angular distance.

            $$d_r = ||h ◦ r-t ||_2$$ 

            Where h, r, t is the k-dimension embedding of the head, relation and tail,restricting the $|r_i| = 1$. <font color=green> relations are in the unit circle and ◦ represents element wise product.</font>    




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
            <img src=./Pictures/KG_embedding/figure4.png width = 70%>



        * HakE[5]

        The previous methods fails to capture the semantic hierarchies. This model models the hierarchy in the entities as concentric circles in **polar coordinate**. *The entity with smaller radious belongs to higher up in the hierachy.*  The angle between them represents the variation in the meaning. This model has two components, one to map the modulus and the other one to map the angle. 


        Score function:

        distant function only consider modulo part:    $d_{r,m} = ||h_m ◦ r_m - t_m||_2$

        phase part: $\lambda||sin((h_p+r_p-t_p)/2)||_1$
        


        Mapping the entity in the polar coordinate space.

        $$-||h_m ◦ r_m - t_m||_2 - \lambda||sin((h_p+r_p-t_p)/2)||_1$$   

        $$h_p, r_p, t_p \in[0,2\pi)^k$$



    * Semantic Matching models

        Use similarity-based scoring function.


        * RESCAL[6] (Statistical Relational Learning Approach) 

            <font color=Fuchsia> Three-way model which performs fairly good for relationships which occur frequently **but it performs poor for the rare relationships and leads to major over-fitting.**</font>

            $$f_r(T) = h^tM_rt = \sum_{i=0}^{d-1}\sum_{j=0}^{d-1} [M_r]_{ij}* [h]_i *[t]_j$$

            where $h, t \in R^d$ are vector representation of entities, and $M_r \in R^{d*d}$ is a matrix representation of $r^{th}$ relation.

            <font color=green> In a simple way: $h^TM_rt$ </font>

            we use weighted sum of all the pairwise interactions between the latent features of the entities $h$ and  $t$.

            $\chi_{ijk}=1$ means exist a relation and if $\chi_{ijk}=0$ means their relation is unknown.

            <center class="half">

            <img src=./Pictures/KG_embedding/figure5.png width = 40%><img src=./Pictures/KG_embedding/figure6.png width = 47%> 


            </center>





        * TATEC[7]  stands for Two And Three-way Embeddings Combination.

            *first stage use two different embeddings, and then combine and fine-tuning them* 

            two way interactions: $f_r^1(h,t)= h^Tr + t^Tr + h^TDt$

            three way interactions: $f_r^2(h,t)=h^TM_rt$

            $$h_r(h,t) = f_r^1(h,t) + f_r^2(h,t)$$
            $$h_r(h,t) = h^Tr + t^Tr + h^TDt + h^TM_rt$$


            <font color=FireBrick> Time complexity and the space complexity of TATEC is same as RESCAL as TATEC extends RESCAL. </font>


        * DistMult [8]



        * HolE



        * ComplEx


        * ANALOGY








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



        ```python
            def pRotatE(self, head, relation, tail, mode):
                pi = 3.14159262358979323846
                
                #Make phases of entities and relations uniformly distributed in [-pi, pi]

                phase_head = head/(self.embedding_range.item()/pi)
                phase_relation = relation/(self.embedding_range.item()/pi)
                phase_tail = tail/(self.embedding_range.item()/pi)

                if mode == 'head-batch':
                    score = phase_head + (phase_relation - phase_tail)
                else:
                    score = (phase_head + phase_relation) - phase_tail

                score = torch.sin(score)            
                score = torch.abs(score)

                score = self.gamma.item() - score.sum(dim = 2) * self.modulus
                return score

        ```

        









































[1] Antoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. 2014. *A semantic matching energy function for learning with multi-relational data: Application to word-sense disambiguation.* Mach. Learn., 94(2):233–259.

[2] Zhen Wang, J. Zhang, Jianlin Feng, and Z. Chen. 2014. *Knowledge graph embedding by translating on hyperplanes.* In AAAI

[3] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. *Learning entity and relation embeddings for knowledge graph completion.* In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, AAAI’15, page 2181–2187. AAAI Press.

[4] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. 2019a. *Rotate: Knowledge graph embedding by relational rotation in complex space.* CoRR, abs/1902.10197.

[5] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, and Jie Wang. 2019. *Learning hierarchy-aware knowledge graph embeddings for link prediction.* CoRR, abs/1911.09419.

[6] Maximilian Nickel et al. *“A Three-Way Model for Collective Learning on Multi-Relational Data”* international conference on machine learning (2011): n. pag.

[7] Alberto Garc ́ıa-Dur ́an, Antoine Bordes, and Nicolas Usunier. 2014. *Effective blending of two and threeway interactions for modeling multi-relational data.* In Machine Learning and Knowledge Discovery in Databases, pages 434–449, Berlin, Heidelberg. Springer Berlin Heidelberg.

[8] Yinwei Wei, Xiangnan He, Xiang Wang, Richang Hong, Liqiang Nie, and Tat Seng Chua. 2019. *MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video.*MM 2019 - Proc. 27th ACM Int. Conf. Multimed., pages 1437–1445.

[9] Ji, S., Pan, S., Cambria, E., Marttinen, P. and Philip, S.Y., 2021. *A survey on knowledge graphs: Representation, acquisition, and applications.* IEEE Transactions on Neural Networks and Learning Systems, 33(2), pp.494-514.

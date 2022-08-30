# Knowledge Representation Learning


<font size=3> **Definition** </font>

Knowledge graph embedding is a approach to ***transform Knowledge Graphs*** (nodes, edges and feature vectors) ***into a low dimensional continuous vector*** space that preserves various graph structure information, etc.  



Then this note will present four section about the knowledge representation learning. including representation space, scoring function, encoding models and auxillary information.

**A map of all algorithms[9]:**


<center class='half'>
<img src=./Pictures/KG_embedding/figure9.png>
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
        <img src=./Pictures/KG_embedding/figure1.png>    
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


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure12.png>
        </center>

        <p align=center> <font color=DarkOliveGreen> Figure6 </font> </p>



    

    <font color=red> *Conclusion* </font>

    With the introduction of the rotational Hadmard product in complex space, RotatE can also capture inversion and composition patterns as well as symmetry and antisymmetry.


    QuatE uses Hamilton product to capture latent inter-dependencies within the four-dimensional space of entities and relations and gains a more expressive rotational capability than RotatE.



* Guassian distribution:

    * **KG2E [17]**
    
        Inspired by Gaussian word embedding， KG2E  introduces Gaussian distribution to deal with the *(un)certainties of entities and relations*.

        KG2E  embedded entities and relations into multi-dimensional Gaussian distribution $H \sim  N( \mu_h, \sum_h )$ and $T \sim N(\mu_t, \sum_t)$.

        The mean vector $\mu$ indicates entities and relations' position, and the covariance matrix $\sum$ models their (un)certainties.

        Following the translational principle, the probability distribution of entity transformation can be presented as:

        $$H-T = P_e \sim N(\mu_h-\mu_t, \sum_h+ \sum_t)$$


        <font color='red'> Scoring function:</font>

        $$\int _{x \in R^{k_e}} {N(x; \mu_r, \sum_r) \ log(\frac{N(x; \mu_e, \sum_e)}{N(x; \mu_r, \sum_r})}dx $$


        $$\mu_h , \mu_t \in R^d$$  

        $$log{\int_{x \in R^{k_e}} N(x; \mu_e, \sum_e)N(x; \mu_r, \sum_r) dx}$$

        $$\sum_h, \sum_t \in R^{d \times d}$$



    * **TransG [18]**

        TransG represents entities with Gaussian distributions, while it draws a mixture of Gaussian distribution for relation embedding, where the $m^{th}$ component translation vector of relation $r$ is denoted as

        $$u_{r,m}=h-t \sim N(\mu_t-\mu_h,(\sigma_h^2+\sigma_t^2)I)$$

        Scoring function

        $$\sum_i \pi_r^i exp(-\frac{||\mu_h+\mu_r^i-\mu_t||}{\sigma_h^2+\sigma_t^2})$$

        





* Manifold and group

    A manifold is a topological space, which could be defined as a set of points with neighborhoods by the set theory.

    The group is algebraic structures defined in abstract algebra.

    <font color=green>
    Previous point-wise modeling is an ill-posed algebraic system where the number of scoring equations is far more than the number of entities and relations. 
    
    Moreover, embeddings are restricted in an overstrict geometric form even in some methods with subspace projection.

    </font>


    * **ManifoldE [19]**

        This method introduced two settings of manifold-based embedding, Sphere and Hyperplane. examples can be found in Figure1(d) and Figure6


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure13.png width=70%>
        </center>

        <p align=center> <font color=DarkOliveGreen> Figure6: Visualization of embedding for Manifold-based models. (a) corresponds to the Sphere setting where all the tail entities are supposed to lay in the sphere. As Clock Dial is matched by the two facts, it should lay in both spheres. (b) corresponds to the Hyperplane setting where Clock Dial should lay and does lay in both hyperplanes, making embedding more precise. </font> </p>

        
        For the sphere setting, Reproducing Kernel Hilbert Space is used to represent the mainfold function.

        Hyperplane is introduced to enhance the model with intersected embeddings. 


        ManifoldE relaxes the real-valued point-wise space into manifold space with a more expressive representation from the geometric perspective. When the manifold function and relation-specific manifold parameter are set to zero, the manifold collapses into a point.


        $$||M(h,r,t)-D_r^2||_2^2$$

        $$h,r,t \in R^d$$


    * **MuRP[20]**

        Knowledge graph relations exhibit multiple properties, such as symmetry, asymmetry, and transitivity. Certain knowledge graph relations, such as hypernym and has_part, induce a hierarchical structure over entities, suggesting that embedding them in hyperbolic rather than Euclidean space may lead to improved representations.  Based on this intuition, MuRP focus on embedding multi-relational knowledge graph data in hyperbolic space. [not convinced]
        
        In the Hyperbolic space, a multidimensional Riemannian manifold with a constant negative curvature  $-c \ (c>0) : B^{d,c} = \{ x \in R^d: ||x||^2  < \frac{1}{c}\}$

        MuRP represents the multi-relational knowledge graph in Poincare ball of hyperbolic space $B_c^d = {x \in R^d:c||x||^2 < 1}$. <font color=red>*While it fails to capture logical patterns and suffers from constant curvature.*</font>


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure14.png>
        </center>

        <p align=center> <font color=DarkOliveGreen> Figure7 </font> </p>


        Scoring function: As shown in the figure7.

        Poincare ball of hyperbolic space
        
        $$-d_B(exp_0^c(R \ log_0^c(h), t \ \oplus_c r)^2 + b_h +b_t)$$

        $$h,t,r \in B_c^d \qquad b_h,b_t \in R$$

    

    * **TorusE [21]**

        TorusE solves the regularization problem of TransE via embedding in an n-dimensional torus space which is a compact Lie group.

        The torus space is defined as:

        $$\pi \ : R^n \rightarrow T^n \ , x \rightarrow [x]$$

        Entites and relations are denoted as  $[h], [r], [t] \in T^n$

        As shown in the Figure7

        <center class='half'>
        <img src=./Pictures/KG_embedding/figure15.png width = 70%>
        </center> 

        <p align=center> <font color=DarkOliveGreen> Figure7 </font> </p>

        Scoring function:

        $$min_{(x,y) \in ([h]+[r]) \times [t]} ||x-y||_i$$

    * **DihEdral [22]**

        DihEdral proposes a dihedral symmetry group preserving a 2-dimensional polygon. It utilizes a finite non-Abelian group to preserve the relational properties of symmetry/skew-symmetry, inversion, and composition effectively with the rotation and reflection properties in the dihedral group.

        <center class='half'>
        <img src=./Pictures/KG_embedding/figure16.png width = 50%>
        </center> 

        <p align=center> <font color=DarkOliveGreen> Figure8 </font> </p>

        Scoring function

        $$\sum_{l=1}^L h^{(l) \mathrm T}R^{(l)t^{(l)}}$$

        $$h^{(l)}, t^{(l)} \in R^2 \qquad R^{(l)} \in D_K$$ 

        






<font size=3> **2. Scoring function (measuring the plausibility of facts)**</font>


Distance-based scoring function measures the plausibility of facts by calculating the distance between entities, where addictive translation with relations as $h + r \approx t$ is widely used.

Semantic similarity based scoring measures the plausibility of facts by semantic matching. It usually adopts a multiplicative formulation, i.e., $h^TM_r \approx t^T $, to transform head entity near the tail in the representation space.

<center class='half'>

<img src=./Pictures/KG_embedding/figure17.png width = 50%>

</center> 

<p align=center> <font color=DarkOliveGreen> Figure9 </font> </p>


* **Translational distancebased scoring**

    * Structural embedding (SE) uses two projection matrices and $L1$ distance to learn structural embedding as

        $$f_r(h,t)=||M_{r,1}h-M_{r,2}t||_{L_1}$$
    
    *  Another way is aims to learn embeddings by representing relations as translations from head to tail entities. 

        $$f_r(h,t)=||h+r-t||_{L_1/L_2}$$
    


        TransE by assuming that the added embedding of $h + r$ should be close to the embedding of $t$ with the scoring function. 
         
         TransH projects entities and relations into a hyperplane, TransR introduces separate projection spaces for entities and relations, and TransD constructs dynamic mapping matrices $M_{rp}=r_ph_p^T + I$ and $M_{rt} = r_pt_p^T + I$ by the projection vectors $h_p, tp, rp \in R^n$. 
         
         By replacing Euclidean distance, TransA  uses Mahalanobis distance to enable more adaptive metric learning.

        Previous methods used additive score functions, TransF relaxes the strict translation and uses dot product as $f_r(h, t) = (h + r)^Tt$.

        ITransF enables hidden concepts discovery and statistical strength transferring by learning associations between relations and concepts via sparse attention vectors, with scoring function defined as

        $$f_r (h, t) =∥\alpha_r^H · D · h + r − α_r^T · D · t∥_t$$

        $D \in R^{n \times d \times d}$ is stacked concept projection matrices of entities and relations and $\alpha_r^H \ , \alpha_r^T$




        

    <center class='half'>
    <img src=./Pictures/KG_embedding/figure18.png>
    </center> 
    <p align=center> <font color=DarkOliveGreen> Figure9 </font> </p>

    **TransAt**[23]  integrates relation attention mechanism with translational embedding, as shown in the Figure9. The scoring function of TransAt is shown as below:

    $$P_r(\sigma(r_h)h) + r - P_r(\sigma(r_t)t)$$

    $$h,t,r \ \in R^d$$

    **TransMS** [24]  transmits multi-directional semantics with nonlinear functions and linear bias vectors, with the scoring function as

    $$f_r(h,t) = ||-tanh(t \circ r) \circ h + r - tanh(h \circ r) \circ t + \alpha \cdot (h \circ t)||_{l_{1/2}}$$

    **KG2E** in Gaussian space and **ManifoldE**  with manifold also use the translational distance-based scoring function. KG2E uses two scoring methods, i.e, asymmetric KL-divergence and symmetric expected likelihood.



* **Semantic similarity-based scoring**

    * **SME[23]**  

        proposes to semantically match separate combinations of entity-relation pairs of $(h, r)$ and $(r, t)$.  Its scoring function is defined with two versions of matching blocks - linear and bilinear block, i.e.,

        linear matching block:
        $$g_{left}(h,t)=M_{l,1}h^T+M_{l,2}r^T+b_l^T$$

        $$g_{right}(r,t)=M_{r,1}t^T+M_{r,2}r^T+b_r^T$$


        
        bilinear form is:
        $$g_{left}(h,r)= (M_{l,1}h) \ \circ \  (M_{l,2}r) +b_l^T $$
        $$g_{right}(r,t)= (M_{r,1}t) \ \circ \  (M_{r,2}r) +b_r^T $$

        **Scoring function:**

        $$f_r(h,t) = g_{left}(h,r)^T \ g_{right}(r,t)$$


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure19.png>
        </center> 
        <p align=center> <font color=DarkOliveGreen> Figure10 </font> </p>


    * **DistMult [24]**

        By restricting relation matrix $M_r$ to be diagonal for multi-relational representation learning, DistMult proposes a simplified bilinear formulation defined as:

        $$f_r(h,t)= h^T \ diag(M_r) \ t$$
        $$h,r,t \in R^d$$
    


    * **HolE[25]**

        To capture productive interactions in relational data and compute efficiently, HolE introduces a circular correlation of embedding, which can be interpreted as a compressed tensor product, to learn compositional representations.




    * **HolEx[26]**

        HolEx defines a perturbed holographic compositional operator.

        $$p(a,b;c) = (c \ \circ a) \ \star \ b $$

        where $c$ is a fixed vector.  HolEx interpolates the HolE and full tensor product method.  It can be viewed as linear concatenation of perturbed HolE


    * **ANALOGY[27]**

        Focusing on multi-relational inference  ANALOGY models analogical structures of relational data. It's scoring function is defined as:

        $$f_r(h,t) = h^TM_rt$$

        with relation matrix constrained to be normal matrices in linear mapping, i.e.,  $M_r^TM_r = M_rM_r^T$ for analogical inference.

    * **CrossE[28]**

        Crossover interactions are introduced by CrossE with an interaction matrix $C \in R^{{n_r} \times d}$ to simulate the bi-directional interaction between entity and relation. The relation specific interaction is obtained by looking up interaction matrix ascr = x>r C. By combining the interactive representations and matching with tail embedding, the scoring function is defined as

        $$f(h,r,t) = \sigma (tanh(c_r \ \circ h + c_r \ \circ \ h \circ \ r + b) \ t^T)$$


    





    **HolE** with Fourier transformed in the frequency domain can be viewed as a special case of **ComplEx**, which connects holographic and complex embeddings. The analogical embedding framework can recover or equivalently obtain several models such as **DistMult**, ComplEx and HolE by restricting the embedding dimension and scoring function.

    <center class='half'>
    <img src=./Pictures/KG_embedding/figure20.png width = 70%>
    </center> 
    <p align=center> <font color=DarkOliveGreen> Figure11 </font> </p>




<font size=3> **3.  Encoding models (modeling the semantic interaction of facts)**</font>

Linear models formulate relations as a linear/bilinear mapping by projecting head entities into a representation space close to tail entities. Factorization aims to decompose relational data into low-rank matrices for representation learning. Neural networks encode relational data with non-linear neural activation and more complex network structures by matching semantic similarity of entities and relations.

* **Linear/Bilinear Models**





* **Factorization Models**




* **Neural Networks**





* **Convolutional Neural Networks**





* **Recurrent Neural Networks**





* **Transformers**





* **Graph Neural Networks (GNNs)**













<center class='half'>
<img src=./Pictures/KG_embedding/figure21.png>
</center> 
<p align=center> <font color=DarkOliveGreen> Figure12 </font> </p>




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

[10] Richard Socher, Danqi Chen, Christopher D. Manning and Andrew Y. Ng. “*Reasoning With Neural Tensor Networks for Knowledge Base Completion*” neural information processing systems (2013): n. pag.

[11] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, & Jie Wang (2019). *Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction* national conference on artificial intelligence.

[12] Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, & Jun Zhao (2015). *Knowledge Graph Embedding via Dynamic Mapping Matrix* international joint conference on natural language processing.

[13] Han Xiao, Minlie Huang, Yu Hao, & Xiaoyan Zhu (2015). *TransA: An Adaptive Approach for Knowledge Graph Embedding*.. arXiv: Computation and Language.

[14]  Fan, M.; Zhou, Q.; Chang, E.; and Zheng, T. F. 2014. *Transition-based knowledge graph embedding with relational mapping properties.* In Proceedings of the 28th Pacific Asia Conference on Language, Information, and Computation, 328–337.

[15] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier, & Guillaume Bouchard (2016). *Complex embeddings for simple link prediction* international conference on machine learning.

[16] Shuai Zhang, Yi Tay, Lina Yao, & Qi Liu (2019). *Quaternion Knowledge Graph Embeddings* neural information processing systems.

[17] Shizhu He, Kang Liu, Guoliang Ji, & Jun Zhao (2015). *Learning to Represent Knowledge Graphs with Gaussian Embedding* conference on information and knowledge management.

[18] Han Xiao, Minlie Huang, & Xiaoyan Zhu (2016). *TransG : A Generative Model for Knowledge Graph Embedding* meeting of the association for computational linguistics.

[19] Han Xiao, Minlie Huang, Yu Hao, & Xiaoyan Zhu (2015). *From One Point to A Manifold: Orbit Models for Knowledge Graph Embedding*.. arXiv: Artificial Intelligence.

[20] Ivana Balažević, Carl Allen, & Timothy M. Hospedales (2019). *Multi-relational Poincaré Graph Embeddings* arXiv e-prints.

[21] Takuma Ebisu, & Ryutaro Ichise (2017). *TorusE: Knowledge Graph Embedding on a Lie Group* arXiv: Artificial Intelligence.

[22] Canran Xu, & Ruijiang Li (2019). *Relation Embedding with Dihedral Group in Knowledge Graph* arXiv: Computation and Language.

[23] Xavier Glorot, Antoine Bordes, Jason Weston, & Yoshua Bengio (2013). *A Semantic Matching Energy Function for Learning with Multi-relational Data* international conference on learning representations.

[24] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, & Li Deng (2014). *Embedding Entities and Relations for Learning and Inference in Knowledge Bases* arXiv: Computation and Language.

[25] Maximilian Nickel, Lorenzo Rosasco, & Tomaso Poggio (2015). *Holographic Embeddings of Knowledge Graphs* national conference on artificial intelligence.

[26] Yexiang Xue, Yang Yuan, Zhitian Xu, & Ashish Sabharwal (2018). *Expanding Holographic Embeddings for Knowledge Completion* neural information processing systems.

[27] Hanxiao Liu, Yuexin Wu, & Yiming Yang (2017). *Analogical Inference for Multi-relational Embeddings* international conference on machine learning.

[28] Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein, & Huajun Chen (2019). *Interaction Embeddings for Prediction and Explanation in Knowledge Graphs* web search and data mining.

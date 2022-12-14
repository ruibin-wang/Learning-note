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


        Scoring function of NTN is 

        $$f_r(h,t) = r^ \mathrm T \ \sigma(h^ \mathrm T \hat{M}t + M_{r,1}h+ M_{r,2}t+b_r )$$

        where $b_r \in R^k$ is the bias for relation $r$, $M_{r,1}$ and $M_{r,2}$ are relation-specific weight matrices.

        It can be regarded as a combination of MLPs and bilinear models.

        As shown in the Figure 4(a)


        <center class='half'>
        <img src=./Pictures/KG_embedding/figure11.png >    
        </center>

        <p align='center'> <font color=DarkOliveGreen> Figure 4 </font></p>

    

    * **Hierarchy-Aware Knowledge Graph Embedding ???HAKE??? [11]**

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
    
        Inspired by Gaussian word embedding??? KG2E  introduces Gaussian distribution to deal with the *(un)certainties of entities and relations*.

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

        $$f_r (h, t) =???\alpha_r^H ?? D ?? h + r ??? ??_r^T ?? D ?? t???_t$$

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

        $$f_r(h,t)= h^\mathrm T \ diag(M_r) \ t$$
        $$h,r,t \in R^d$$
    


    * **HolE[25]**

        To capture productive interactions in relational data and compute efficiently, HolE introduces a circular correlation of embedding, which can be interpreted as a compressed tensor product, to learn compositional representations.




    * **HolEx[26]**

        HolEx defines a perturbed holographic compositional operator.

        $$p(a,b;c) = (c \ \circ a) \ \star \ b $$

        where $c$ is a fixed vector.  HolEx interpolates the HolE and full tensor product method.  It can be viewed as linear concatenation of perturbed HolE


    * **ANALOGY[27]**

        Focusing on multi-relational inference  ANALOGY models analogical structures of relational data. It's scoring function is defined as:

        $$f_r(h,t) = h^\mathrm T \ M_rt$$

        with relation matrix constrained to be normal matrices in linear mapping, i.e.,  $M_r^TM_r = M_rM_r^T$ for analogical inference.

    * **CrossE[28]**

        Crossover interactions are introduced by CrossE with an interaction matrix $C \in R^{{n_r} \times d}$ to simulate the bi-directional interaction between entity and relation. The relation specific interaction is obtained by looking up interaction matrix ascr = x>r C. By combining the interactive representations and matching with tail embedding, the scoring function is defined as

        $$f(h,r,t) = \sigma (tanh(c_r \ \circ h + c_r \ \circ \ h \circ \ r + b) \ t^ \mathrm T)$$


    





    **HolE** with Fourier transformed in the frequency domain can be viewed as a special case of **ComplEx**, which connects holographic and complex embeddings. The analogical embedding framework can recover or equivalently obtain several models such as **DistMult**, ComplEx and HolE by restricting the embedding dimension and scoring function.

    <center class='half'>
    <img src=./Pictures/KG_embedding/figure20.png width = 70%>
    </center> 
    <p align=center> <font color=DarkOliveGreen> Figure11 </font> </p>




<font size=3> **3.  Encoding models (modeling the semantic interaction of facts)**</font>

Linear models formulate relations as a linear/bilinear mapping by projecting head entities into a representation space close to tail entities. Factorization aims to decompose relational data into low-rank matrices for representation learning. Neural networks encode relational data with non-linear neural activation and more complex network structures by matching semantic similarity of entities and relations.



<center class='half'>
<img src=./Pictures/KG_embedding/figure21.png>
</center> 
<p align=center> <font color=DarkOliveGreen> Figure12 </font> </p>



* **Linear/Bilinear Models**

    Linear/bilinear models encode interactions of entities and relations by applying linear operation as:

    $$g_r(h,t) = M_r^T 
    \begin{pmatrix}
    h \\
    t\\
    \end{pmatrix}
    $$

    Canonical methods with linear/bilinear encoding include SE, SME, DistMult, ComplEx, and ANALOGY. 

    TransE with L2 regulariazation, 1-d vector linear transformation scoring function:

    $$||h+r-t||_2^2 \ = 2r^T(h-t) - 2h^Tt+||r||_2^2+||h||_2^2+||t||_2^2$$


    * **SimplE**

    to solve the independence embedding issue of entity vectors in canonical Polyadia decomposition, SimplE [48] introduces the inverse of relations and calculates the average canonical Polyadia score of $(h, r, t)$ and $(t, r^{???1}, h)$ as

    $$f_r(h,t) = \frac{1}{2}(h \circ rt + t \circ r't)$$

    where $r'$ is the embedding of inversion relation. 

    Embedding models in the bilinear family such as RESCAL, DistMult, HolE and ComplEx can be transformed from one into another with certain constraints



* **Factorization Models**

    * **RESCAL[6]** (Statistical Relational Learning Approach) 

        <font color=Fuchsia> Three-way model which performs fairly good for relationships which occur frequently **but it performs poor for the rare relationships and leads to major over-fitting.**</font>

        $$f_r(T) = h^tM_rt = \sum_{i=0}^{d-1}\sum_{j=0}^{d-1} [M_r]_{ij}* [h]_i *[t]_j$$

        where $h, t \in R^d$ are vector representation of entities, and $M_r \in R^{d*d}$ is a matrix representation of $r^{th}$ relation.

        <font color=green> In a simple way: $h^TM_rt$ </font>

        we use weighted sum of all the pairwise interactions between the latent features of the entities $h$ and  $t$.

        $\chi_{ijk}=1$ means exist a relation and if $\chi_{ijk}=0$ means their relation is unknown.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure5.png width = 40%><img src=./Pictures/KG_embedding/figure6.png width = 47%> 
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure13  </font> </p>


        <font color=red>
        two-way model means use 2 dimension tensor $d \times d$  

        three-way means use 3 dimension tensor $d \times d \times d$
        </font>


    * **Latent factor model (LFM) [29]**

        **LFM** which extends RESCAL by decomposing
        $$R_k = \sum_{i=1}^d \alpha_i^ku_iv_i^ \mathrm T$$

    * **TuckER[30]**

        TuckER learns to embed by outputting a core tensor and embedding vectors of entities and relations.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure22.png width=50%>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure14 </font> </p>


        $$??(e_s, r, e_o) = W ??_1 e_s ??_2 w_r ??_3 e_o$$



    * **LowFER[31]**

        LowFER proposes a multimodal factorized bilinear pooling mechanism to better fuse entities and relations. It generalizes the TuckER model and is computationally efficient with low-rank approximation.

        Scoring function:

        $$(S^k \ diag(U^Th) \ V^T \ r)^Tt$$
        $$h,r,t \in R^d$$


* **Neural Networks**

    Encoding models with linear/bilinear blocks can also be modeled using neural networks, for example, **SME**. Representative neural models include **multi-layer perceptron (MLP)**, **neural tensor network (NTN)**, and **neural association model (NAM)**. They generally feed entities or relations or both into deep neural networks and compute a semantic matching score.


    * **MLP[32]**

        MLP encodes entities and relations together into a fully-connected layer, and uses a second layer with sigmoid activation for scoring a triple as

        $$f_r(h,t) = \sigma(w^T \ \sigma(W[h,r,t]))$$

        where $W$ is the weight matrix, $[h,r,t]$ is a concatenation of three vectors.
    
        


* **Convolutional Neural Networks (CNN)**

    * **ConvE[33]**  
    
        ConvE uses 2D convolution over embeddings and multiple layers of nonlinear features to model the interactions between entities and relations by reshaping head entity and relation into 2D matrix. 

        Scoring function:


        $$f_r(h,t) = \sigma(vec( \sigma([M_h ; M_r] * \omega)) \mathbf{W}) \ t$$

        <center class="half">
        <img src=./Pictures/KG_embedding/figure23.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure15 </font> </p>


    * **ConvKB[34]**

        ConvKB adopts CNNs for encoding the concatenation of entities and relations without reshaping (Figure12a). Its scoring function is defined as

        $$f_r(h,t) = concat(\sigma([h,r,t] * \omega)) \ ?? \mathbf{w} $$

        
        <center class="half">
        <img src=./Pictures/KG_embedding/figure24.png width=45%>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure16 </font> </p>

    
    * **HypER[35]**

        HypER utilizes hypernetworkH for 1D relation-specific convolutional filter generation to achieve multi-task knowledge sharing, and meanwhile simplifies 2D ConvE. It can also be interpreted as a tensor factorization model when taking hypernetwork and weight matrix as tensors.

        Scoring function

        $$\sigma (vec(h*vec^{-1}(w_rH)) \mathbf{W}) \ t$$


    <font color=red> Conclusion </font>

    The concatenation of a set for feature maps generated by convolution increases the learning ability of latent features. Compared with ConvE, which captures the local relationships, ConvKB keeps the transitional characteristic and shows better experimental performance.



* **Recurrent Neural Networks (RNN)**

    The recurrent networks can capture long-term relational dependencies in knowledge graphs.

    * **RNN-based model[36] [37]**

        RNN-based model over the relation path to learn vector representation without and with entity information, respectively.


    * **Recurrent Skipping Networks (RSN)[38]**


        RSN (Figure12c) designs a recurrent skip mechanism to enhance semantic representation learning by distinguishing relations and entities.

        The relational path as $(x_1, x_2, . . . , x_T)$ with entities and relations in an alternating order is generated by random walk, and it is further used to calculate recurrent hidden state $h_t = tanh(W_hh_{t-1} \ +W_xx_t \ +b)$, the skipping operation is conducted as:

        $$h_t^{'} = \left \{
            \begin{aligned}
            h_t  \ \ \   x_t \in \epsilon  \\
            S_1h_t \ + \ S_2x_{t-1} \ \ \ \  x_t \in R
            \end{aligned}
            \right.$$



* **Transformers**

    Transformer-based models have advantages in representing the contextualized text.


    * **CoKE[39]**

        CoKE employs transformers to encode edges and path sequences.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure25.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure17 </font> </p>


    * **KG-BERT[40]**

        entities and relations are encoded into a large BERT network.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure26.png width=50%>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure18 </font> </p>



* **Graph Neural Networks (GNNs)**

    connectivity structure under an encoder-decoder framework.

    * **R-GCN[41]**

        <center class="half">
        <img src=./Pictures/KG_embedding/figure27.png width=50%>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure19  Diagram for computing the update of a single graph node/entity (red) in the R-GCN model. Activations (d-dimensional vectors) from neighboring nodes (dark blue) are gathered and then transformed for each relation type individually (for both in- and outgoing edges). The resulting representation (green) is accumulated in a (normalized) sum and passed through an activation function (such as the ReLU). This per-node update can be computed in parallel with shared parameters across the whole graph. </font> </p>

        R-GCN proposes relation-specific transformation to model the directed nature of knowledge graphs. *R-GCN takes the neighborhood of each entity equally.*

    
    * **SACN [42]**

        SACN  introduces weighted GCN (Figure12 b), which defines the strength of two adjacent nodes with the same relation type, to capture the structural information in knowledge graphs by utilizing node structure, node attributes, and relation types. The decoder module called **Conv-TransE** adopts ConvE model as semantic matching metric and preserves the translational property. By aligning the convolutional outputs of entity and relation embeddings with *C* kernels to be $M(h, r) \in R^{C \times d}$, its scoring function is defined as

        $$f_r(h,t) = g(vec(M(h,r)) \ W) \ t$$

        <center class="half">
        <img src=./Pictures/KG_embedding/figure28.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure20 </font> </p>

    
    * **graph attention network (GATs)**

        GATs with multi-head attention as encoder to capture multi-hop neighborhood features by inputing the concatenation of entity and relation embeddings.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure29.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure21 </font> </p>


    * **CompGCN [44]**

        CompGCN proposes entity-relation composition operations over each edge in the neighborhood of a central node and generalizes previous GCN-based models.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure30.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure22 </font> </p>



<font size=3> **4.  Auxiliary information (utilizing external information)** </font>

* **Textual Description**

    Knowledge Graph and Text Jointly Embedding[45]:   proposed two alignment models for aligning entity space and word space by introducing entity names and Wikipedia anchors.Objective function is:

    $$L = L_K+L_T+L_A$$

    * **DKRL[46]**
    
        extends TransE to learn representation directly from entity descriptions by a convolutional encoder.

    * **SSP[47]**  
    
        captures the strong correlations between triples and textual descriptions by projecting them in a semantic subspace. Objective function is :

    $$L = L_{embed} + \mu L_{topic}$$

    in which $L_{topic}$ is the textual description.



* **Type Information**

    entites always can be represented with hierarchical classes or types.

    * **SSE[48]**

        SSE incorporates semantic categories of entities to embed entities belonging to the same category smoothly in semantic space.


    * **TKRL[49]**

        TKRL proposes type encoder model for projection matrix of entities to capture type hierarchy. Noticing that some relations indicate attributes of entities.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure31.png width=50%>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure23 </font> </p>


    * **KR-EAR[50]**

        KR-EAR categorizes relation types into attributes and relations and modeled the correlations between entity descriptions.


        <center class="half">
        <img src=./Pictures/KG_embedding/figure32.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure24 </font> </p>

    * **Hierarchical Relation Structure(HRS)**

        HRS extended existing embedding methods with hierarchical relation structure of relation clusters, relations, and sub-relations.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure33.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure25 </font> </p>




* **Visual Information**

    e.g., entity images can be utilized to enrich KRL.

    * **Image-embodied IKRL[52]**

        Containing cross-modal structure-based and imagebased representation, encodes images to entity space and follows the translation principle. The cross-modal representations make sure that structure-based and image-based representations are in the same representation space.

        <center class="half">
        <img src=./Pictures/KG_embedding/figure34.png>
        </center>
        <p align=center> <font color=DarkOliveGreen> Figure26 </font> </p>

    
    Ref [53] gave a detailed review of using additional information.



* **Uncertain Information**

    * knowledge graphs contain uncertain information with a confudence score assigned to every relation fact. ProBase[54], NELL[55], and ConceptNet[56].


    *  uncertain embedding models aim to capture uncertainty representing the likelihood of relational facts.



<font size=3> **5. Summary** </font>

  <font color=red>

Four question for developing a noval KRL:

(1)  which representation space to choose?

(2)  how to measure the plausibility of triplets in a specific space?

(3)  which encoding model to use for modeling relational interactions?

(4)  whether to utilize auxiliary information?

</font>

When developing a representation learning model, appropriate representation space should be selected and designed carefully to match the nature of encoding methods and balance the expressiveness and computational complexity.









---


[1] Antoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. 2014. *A semantic matching energy function for learning with multi-relational data: Application to word-sense disambiguation.* Mach. Learn., 94(2):233???259.

[2] Zhen Wang, J. Zhang, Jianlin Feng, and Z. Chen. 2014. *Knowledge graph embedding by translating on hyperplanes.* In AAAI

[3] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. *Learning entity and relation embeddings for knowledge graph completion.* In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, AAAI???15, page 2181???2187. AAAI Press.

[4] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. 2019a. *Rotate: Knowledge graph embedding by relational rotation in complex space.* CoRR, abs/1902.10197.

[5] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, and Jie Wang. 2019. *Learning hierarchy-aware knowledge graph embeddings for link prediction.* CoRR, abs/1911.09419.

[6] Maximilian Nickel et al. *???A Three-Way Model for Collective Learning on Multi-Relational Data???* international conference on machine learning (2011): n. pag.

[7] Alberto Garc ????a-Dur ??an, Antoine Bordes, and Nicolas Usunier. 2014. *Effective blending of two and threeway interactions for modeling multi-relational data.* In Machine Learning and Knowledge Discovery in Databases, pages 434???449, Berlin, Heidelberg. Springer Berlin Heidelberg.

[8] Yinwei Wei, Xiangnan He, Xiang Wang, Richang Hong, Liqiang Nie, and Tat Seng Chua. 2019. *MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video.*MM 2019 - Proc. 27th ACM Int. Conf. Multimed., pages 1437???1445.

[9] Ji, S., Pan, S., Cambria, E., Marttinen, P. and Philip, S.Y., 2021. *A survey on knowledge graphs: Representation, acquisition, and applications.* IEEE Transactions on Neural Networks and Learning Systems, 33(2), pp.494-514.

[10] Richard Socher, Danqi Chen, Christopher D. Manning and Andrew Y. Ng. ???*Reasoning With Neural Tensor Networks for Knowledge Base Completion*??? neural information processing systems (2013): n. pag.

[11] Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, & Jie Wang (2019). *Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction* national conference on artificial intelligence.

[12] Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, & Jun Zhao (2015). *Knowledge Graph Embedding via Dynamic Mapping Matrix* international joint conference on natural language processing.

[13] Han Xiao, Minlie Huang, Yu Hao, & Xiaoyan Zhu (2015). *TransA: An Adaptive Approach for Knowledge Graph Embedding*.. arXiv: Computation and Language.

[14]  Fan, M.; Zhou, Q.; Chang, E.; and Zheng, T. F. 2014. *Transition-based knowledge graph embedding with relational mapping properties.* In Proceedings of the 28th Pacific Asia Conference on Language, Information, and Computation, 328???337.

[15] Th??o Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier, & Guillaume Bouchard (2016). *Complex embeddings for simple link prediction* international conference on machine learning.

[16] Shuai Zhang, Yi Tay, Lina Yao, & Qi Liu (2019). *Quaternion Knowledge Graph Embeddings* neural information processing systems.

[17] Shizhu He, Kang Liu, Guoliang Ji, & Jun Zhao (2015). *Learning to Represent Knowledge Graphs with Gaussian Embedding* conference on information and knowledge management.

[18] Han Xiao, Minlie Huang, & Xiaoyan Zhu (2016). *TransG : A Generative Model for Knowledge Graph Embedding* meeting of the association for computational linguistics.

[19] Han Xiao, Minlie Huang, Yu Hao, & Xiaoyan Zhu (2015). *From One Point to A Manifold: Orbit Models for Knowledge Graph Embedding*.. arXiv: Artificial Intelligence.

[20] Ivana Bala??evi??, Carl Allen, & Timothy M. Hospedales (2019). *Multi-relational Poincar?? Graph Embeddings* arXiv e-prints.

[21] Takuma Ebisu, & Ryutaro Ichise (2017). *TorusE: Knowledge Graph Embedding on a Lie Group* arXiv: Artificial Intelligence.

[22] Canran Xu, & Ruijiang Li (2019). *Relation Embedding with Dihedral Group in Knowledge Graph* arXiv: Computation and Language.

[23] Xavier Glorot, Antoine Bordes, Jason Weston, & Yoshua Bengio (2013). *A Semantic Matching Energy Function for Learning with Multi-relational Data* international conference on learning representations.

[24] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, & Li Deng (2014). *Embedding Entities and Relations for Learning and Inference in Knowledge Bases* arXiv: Computation and Language.

[25] Maximilian Nickel, Lorenzo Rosasco, & Tomaso Poggio (2015). *Holographic Embeddings of Knowledge Graphs* national conference on artificial intelligence.

[26] Yexiang Xue, Yang Yuan, Zhitian Xu, & Ashish Sabharwal (2018). *Expanding Holographic Embeddings for Knowledge Completion* neural information processing systems.

[27] Hanxiao Liu, Yuexin Wu, & Yiming Yang (2017). *Analogical Inference for Multi-relational Embeddings* international conference on machine learning.

[28] Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein, & Huajun Chen (2019). *Interaction Embeddings for Prediction and Explanation in Knowledge Graphs* web search and data mining.

[29] Rodolphe Jenatton, Nicolas Le Roux, Antoine Bordes, & Guillaume Obozinski (2012). *A latent factor model for highly multi-relational data* neural information processing systems.

[30] Ivana Bala??evi??, Carl Allen, & Timothy M. Hospedales (2019). *TuckER: Tensor Factorization for Knowledge Graph Completion*.. arXiv: Learning.

[31] Saadullah Amin, Stalin Varanasi, Katherine Ann Dunfield, & G??nter Neumann (2020). *LowFER: Low-rank Bilinear Pooling for Link Prediction* international conference on machine learning.

[32] Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, & Wei Zhang (2014). *Knowledge vault: a web-scale approach to probabilistic knowledge fusion* knowledge discovery and data mining.

[33] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, & Sebastian Riedel (2017). *Convolutional 2D Knowledge Graph Embeddings* arXiv: Learning.

[34] Dai Quoc Nguyen, Tu Dinh Nguyen, Dat Quoc Nguyen, & Dinh Phung (2017). *A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network* north american chapter of the association for computational linguistics.

[35] Ivana Bala??evi??, Carl Allen, & Timothy M. Hospedales (2018). *Hypernetwork Knowledge Graph Embeddings* international conference on artificial neural networks.

[36] Matt Gardner, Partha Pratim Talukdar, Jayant Krishnamurthy, & Tom M. Mitchell (2014). *Incorporating Vector Space Similarity in Random Walk Inference over Knowledge Bases* empirical methods in natural language processing.

[37] Arvind Neelakantan, Benjamin Roth, & Andrew McCallum (2015). *Compositional Vector Space Models for Knowledge Base* Completion international joint conference on natural language processing.

[38] Lingbing Guo, Zequn Sun, & Wei Hu (2019). *Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs* international conference on machine learning.

[39] Quan Wang, Huang Pingping, Haifeng Wang, Songtai Dai, Jiang Wenbin, Jing Liu, Yajuan Lyu, Zhu Yong, & Hua Wu (2019). *CoKE: Contextualized Knowledge Graph Embedding*.. arXiv: Artificial Intelligence.

[40] Liang Yao, Chengsheng Mao, & Yuan Luo (2019). *KG-BERT: Bert for knowledge graph completion* arXiv: Computation and Language.

[41] Michael Sejr Schlichtkrull, Thomas Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, & Max Welling (2017). *Modeling Relational Data with Graph Convolutional Networks* european semantic web conference.

[42] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, & Bowen Zhou (2018). *End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion* national conference on artificial intelligence.

[43] Deepak Nathani, Jatin Chauhan, Charu Sharma, & Manohar Kaul (2019). *Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs* meeting of the association for computational linguistics.

[44] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, & Partha Pratim Talukdar (2019). *Composition-based Multi-Relational Graph Convolutional Networks Learning*.

[45] Zhen Wang, Jianwen Zhang, Jianlin Feng, & Zheng Chen (2014). *Knowledge Graph and Text Jointly Embedding* empirical methods in natural language processing.

[46] Ruobing Xie, Zhiyuan Liu, Jia Jia, Huanbo Luan, & Maosong Sun (2016). *Representation learning of knowledge graphs with entity descriptions* national conference on artificial intelligence.

[47] Han Xiao, Minlie Huang, & Xiaoyan Zhu (2016). *SSP: Semantic Space Projection for Knowledge Graph Embedding with Text Descriptions* arXiv: Computation and Language.

[48] Shu Guo, Quan Wang, Bin Wang, Lihong Wang, & Li Guo (2015). *Semantically Smooth Knowledge Graph Embedding* international joint conference on natural language processing.

[49] Ruobing Xie, Zhiyuan Liu, & Maosong Sun (2016). *Representation learning of knowledge graphs with hierarchical types* international joint conference on artificial intelligence.

[50] Yankai Lin, Zhiyuan Liu, & Maosong Sun (2016). *Knowledge representation learning with entities, attributes and relations* international joint conference on artificial intelligence.

[51] Zhao Zhang, Fuzhen Zhuang, Meng Qu, Fen Lin, & Qing He (2018). *Knowledge Graph Embedding with Hierarchical Relation Structure* empirical methods in natural language processing.

[52] Ruobing Xie, Zhiyuan Liu, Huanbo Luan, & Maosong Sun (2016). *Image-embodied Knowledge Representation Learning* international joint conference on artificial intelligence.

[53] Quan Wang, Zhendong Mao, Bin Wang, & Li Guo (2017). *Knowledge Graph Embedding: A Survey of Approaches and Applications* IEEE Transactions on Knowledge and Data Engineering.

[54] Wentao Wu, Hongsong Li, Haixun Wang, & Kenny Q. Zhu (2012). *Probase: a probabilistic taxonomy for text understanding* international conference on management of data.

[55] Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam R. Hruschka, & Tom M. Mitchell (2010). *Toward an architecture for never-ending language learning* national conference on artificial intelligence.

[56] Robert Speer, Joshua Chin, & Catherine Havasi (2016). *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge* national conference on artificial intelligence.
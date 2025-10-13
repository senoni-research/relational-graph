# RELIA: Relational Intelligence for Anticipation

by senoni-research


## Introduction

**RELIA** is a comprehensive solution for demand forecasting and inventory planning that leverages relational graph modeling to capture complex dependencies in retail data. Instead of treating store sales as isolated time series, RELIA builds a *graph* of relationships – linking stores, products, and other entities – to learn patterns that traditional models might miss. By training a graph-based model on historical sales and inventory data, RELIA can transfer insights across related products and locations, improving forecasts especially for sparse or new (cold-start) cases. The approach extends ideas from recent Relational Graph Transformer (RelGT) research, tailoring them to the inventory domain and introducing domain-specific enhancements.


## Graph-Based Modeling and Transfer Learning

At the core of RELIA is a **relational graph** representing the retail ecosystem. Nodes in the graph include **store nodes** and **product nodes**, and an edge between a store and product denotes a sales relationship. Each sales event (units sold in a given period) is recorded as a temporal edge with attributes (e.g. the week and quantity sold). By using a NetworkX **MultiGraph** data structure, RELIA preserves all historical transactions as separate timestamped edges. This rich graph context allows the model to learn from *co-movement patterns*: if Product A and Product B sell together at Store X, or if Store Y shows similar seasonal trends to Store Z, these relationships inform the forecast. Crucially, the graph’s structure enables a form of **transfer learning**: a new product with scant data can still benefit from connections to similar products or stores. In practice, this means **cold-start scenarios** are handled more gracefully – the model infers demand for a new or low-sales item by examining its neighbors in the graph (e.g. same category products or the same product in other stores). By sharing learned representations across the network, RELIA generalizes patterns from high-data regions to low-data ones, effectively transferring knowledge within the graph.


## Design of the Relational Graph Model

**Graph Construction:** RELIA’s graph is heterogeneous, capturing multiple relationship types. The primary edges are **“sold” edges** between store and product nodes, labeled with the week and units sold. Additionally, **“has_inventory” edges** mark when a store had a product in stock in a given week (even if zero sales occurred) to distinguish zero sales due to lack of stock versus lack of demand. This conditional inclusion of zero-sales events (only when inventory was present) means the graph encodes both positive sales and true demand gaps. Nodes and edges carry features: each node has a type (store or product) and associated attributes (e.g. store format, product category), and each edge has temporal features (time of sale, and an indicator if that store-product pair was seen before). These features are fed into the model to inform its predictions.

**Model Architecture:** The RELIA model builds upon a graph neural network with attention mechanisms inspired by transformers. In the enhanced version, each node aggregates information from its neighbors through multi-head attention. For example, a product node will attend to all stores that sold it, and a store node will attend to all products it sold, weighing each neighbor’s contribution. With **3 layers of attention** and a hidden dimension of 64–128, the model iteratively refines node embeddings by mixing information from one-hop neighbors. Importantly, RELIA’s model is *time-aware*: when predicting a future edge (sale) between a store and product, it considers whether that pair had an interaction in the past and how recent it was. A special temporal feature indicates if a given store-product pair has been “seen” before the forecast time, which the model uses to modulate its confidence. The output of the model is a score for each potential edge, which can be interpreted as the probability of a meaningful sale in the forecast period. This learned edge scorer is trained with a binary classification objective (sale vs. no-sale) using historical data and temporal train/validation splits to avoid leakage. Although currently framed as a classification problem (predicting the likelihood of a sale event), an enhancement in progress is to change this to a **regression task** – directly predicting the expected units sold for a more explicit demand forecast.

**Transfer Learning in Practice:** Thanks to the graph architecture, RELIA inherently supports inductive generalization – it can incorporate new nodes or edges without retraining from scratch. For instance, if a new store is added, connecting it to the product nodes (via initial sales or inventory edges) allows the model to immediately position the new store in the learned embedding space, drawing on similarities to existing stores. Similarly, a new product can be linked to existing stores or to a product category node (if using a product hierarchy), enabling the model to apply learned category-level patterns. This capability is a form of transfer learning: the model’s knowledge (in its node embeddings and attention weights) transfers to new parts of the graph. In summary, the relational graph approach not only learns the training data relationships but also provides a framework to *extend* those relationships to new data in a flexible, data-driven manner.


## Warehouse and Supply Node Integration

An important extension of RELIA is the integration of **warehouse/supply nodes** into the graph. In many retail supply chains, warehouses or distribution centers play a key role in inventory availability: a stock-out at a warehouse can cascade to multiple stores. To capture this, RELIA’s graph can include warehouse nodes connected via “supplies” edges to store or product nodes. There are a few ways to model these relationships:



* **Warehouse–Product edges:** linking a warehouse to a product indicates that the warehouse stocks that product. The edge might carry attributes like on-hand inventory levels or lead times. This creates an indirect path between a warehouse and all stores it supplies with that product.

* **Warehouse–Store edges:** linking a warehouse to a store means that store is serviced by that warehouse. This could be complemented with temporal edges denoting shipments or restocks from the warehouse to the store.

* **Multi-hop relationships:** A store node could connect to a warehouse node, which in turn connects to a product node. In this case, a store and product might be connected through a two-hop path (store→warehouse→product), indicating supply availability.


Integrating warehouses expands the graph into a **tri-partite structure** (stores, products, warehouses) from the original bipartite store-product graph. The RELIA model can handle this because it treats node types and edge types as features. We simply introduce a new node type “warehouse” (with its own embedding) and new edge relations (“supplies” or “stocks”) alongside the existing “sold” and “has_inventory” relations. During message passing, attention can propagate along these new edges: for example, a store node attending to its connected warehouse could learn how warehouse stock levels affect that store’s sales. In practice, incorporating warehouse data enables **supply-aware forecasting** – the model can learn, for instance, that if the supplying warehouse for a product is out of stock, the probability of sales at the store will drop, even if there is demand. This leads to more realistic predictions that account for supply constraints. It also aids in regional transfer learning: stores served by the same warehouse might exhibit similar demand patterns due to geographic or logistical commonalities, and the model can leverage that shared warehouse connection to transfer knowledge among those stores.


## Predictive Alerting and Quality Assurance

Beyond forecasting, RELIA provides features for **predictive alerting** and **quality assurance (QA)** in inventory management. Because the model outputs a probability (or forecasted volume) for every store–product pair, we can set up automated alerts based on these predictions:



* **Stock-Out Risk Alerts:** If the model predicts a very low probability of sales for an item that normally sells (or significantly lower forecasted units than usual), it could indicate a potential stock-out or supply issue. An alert can prompt teams to check inventory levels or expedite replenishment.

* **Demand Surge Alerts:** Conversely, a spike in the predicted probability or quantity (far above baseline) for a product at a store might signal an upcoming demand surge (e.g. due to an event or trend). This early warning enables proactive stock adjustments.

* **Anomaly Detection and Data QA:** RELIA’s graph can highlight anomalies in the data. For example, if a product shows zero sales but the model was highly confident of a sale (given inventory was present and similar stores sold it), it may flag a data quality issue (perhaps sales were not recorded correctly) or a local issue at that store. QA analysts can investigate such flags. Moreover, because the model is explainable via graph paths (it can identify which neighboring nodes most influenced a prediction), we can trace *why* a prediction was made. For instance, an alert can be accompanied by an explanation like “High demand predicted because neighboring stores saw increased sales of this product category last week,” which aids human analysts in trust and verification.


In practice, implementing predictive alerting involves defining thresholds on the model’s outputs (probabilities or forecast errors). RELIA can be calibrated to estimate well-calibrated probabilities, so that a probability output truly reflects likelihood of a sale. This reliability is important for alerts – e.g., only trigger an out-of-stock alert when predicted sale probability falls below, say, 10% while normally it’s 70%+. Similarly, continuous monitoring of forecast vs actual outcomes can feed a QA dashboard. Over time, these alerts and QA checks help maintain inventory health: preventing lost sales due to stock-outs, avoiding overstocks by catching demand drops, and ensuring data integrity.


## Enhancements over RelGT and Implementation Details

RELIA draws inspiration from the Relational Graph Transformer (RelGT) architecture but diverges in key ways to better serve the inventory forecasting problem. Below we outline differences and how we approach each enhancement:



* **Heterogeneous Graph Structure:** RelGT was introduced for relational databases, using multi-element tokenization to handle heterogeneous, temporal data. RELIA similarly deals with a heterogeneous graph (stores, products, etc.), but we explicitly incorporate domain-specific node and edge types (e.g. warehouse nodes, co-purchase edges). This means our model must handle multiple relation types. We implement this by encoding the edge type into the attention mechanism (e.g., using separate learnable parameters per relation or an edge-type embedding). Thus, RELIA extends RelGT’s idea of heterogeneous graphs by adding new relationship types unique to retail (like “supplies” edges for warehouses or “co-purchased” edges between products frequently bought together).

* **Temporal Multi-Graph vs. Static Relations:** A novel challenge we addressed is representing *repeated interactions over time*. In initial experiments, using a standard graph structure collapsed multiple transactions into one, losing historical signal. RELIA uses a MultiGraph to retain all timestamped edges, ensuring that the model learns from the full time series of interactions rather than just the last transaction. In contrast, the generic RelGT might treat time as an attribute or separate dimension. We found that explicitly modeling each time-stamped edge and using time-aware sampling (train on earlier periods, test on later) was crucial for preventing data leakage and capturing trends. Additionally, RELIA can incorporate **temporal features** like recency or rolling statistics as node attributes to enhance temporal awareness beyond what RelGT’s base architecture provides.

* **Feature Engineering and Hierarchies:** RELIA heavily leverages retail-specific features. For example, each product is linked to a product hierarchy (department, category) which the model encodes as additional nodes or node features. Each store has attributes like format or region. These features help the model cluster similar nodes together in the embedding space. While RelGT would accept generic relational features, our enhancement is a careful feature engineering step: we include log-scaled degree (how many connections a node has) and hop-distance features to inform the model of network position. Incorporating these boosts accuracy by giving the model context (e.g., distinguishing a small convenience store from a large outlet via the store’s degree or format). Implementing this involved creating embedding vectors for categorical features and concatenating them to the node representations before applying attention layers.

* **Model Architecture and Scale:** RelGT uses a transformer-style architecture to attend across relational data. RELIA’s enhanced model employs a **multi-head attention neighbor aggregator** which can be seen as a light-weight RelGT tailored to graph neighborhoods. We limit attention to one-hop neighbors (with optional multi-hop sampling for efficiency) rather than attending over the entire graph, which keeps computation tractable even as we scale to ~172K edges. Training the model for ~15 epochs on three years of data can be done overnight on CPU or faster with a GPU. This pragmatic approach ensures RELIA is deployable in real business settings without requiring the enormous resources that a full transformer on millions of nodes might. Essentially, we balance complexity and practicality: adopting the core idea of relational attention from RelGT but simplifying it in scope.

* **Forecasting Objective:** A key difference is our end goal – **predictive forecasting for inventory decisions**. RelGT in literature was often evaluated on predicting held-out relational links or values. RELIA extends this to a time-series forecasting context. We are enhancing the model to predict actual demand quantity (a regression on units) rather than just a probability of sale. Implementing this involves changing the training target from binary classification (sale/no-sale) to a numerical value (e.g., log of units sold) and likely adjusting the loss function to something like mean squared error or quantile loss (to handle uncertainty in demand). This is an ongoing enhancement, but early steps include calibrating the current probability outputs and combining them with historical averages to produce a quantity estimate. The probabilistic output of the classifier can act as a weight on a base forecast: for instance, multiplying the probability of sale by an expected units (from historical data) yields a demand prediction per period. While not as direct as a learned regression, this intermediate approach provides a bridge until a full regression model is in place.

* **Explainability and QA Tools:** Unlike a vanilla RelGT implementation, RELIA is being built with interpretability in mind (crucial for business acceptance). We log the attention weights and neighbor contributions for predictions, which helps in creating explanations for forecasts (“demand forecast is high because similar stores in the region showed increased sales”). This is an enhancement on top of the model – effectively an analytic layer – but it’s informed by the graph structure (tracing paths and important neighbors). Additionally, we incorporate a calibration step (using techniques like isotonic regression on validation data) to ensure the predicted probabilities are reliable, which is not typically part of RelGT but important when integrating forecasts into inventory decision rules and alert thresholds.


In summary, RELIA adapts and extends the RelGT paradigm with a focus on **retail inventory forecasting**. By integrating supply chain nodes, leveraging temporal multi-graphs, and focusing on actionable outputs (like order quantities and alerts), RELIA provides a robust, extensible framework. It maintains the strengths of relational graph learning – capturing heterogeneity and complex relations – while adding the domain-specific tweaks needed for real-world accuracy. The result is a state-of-the-art forecasting engine that not only predicts demand more accurately but also offers transparency and adaptability for continuous improvement.

**Sources:** The design and improvements of RELIA are informed by both the literature on relational graph transformers and our experimental findings on multi-year store-product data. Continued development (such as adding co-purchase edges and longer training) is underway to further enhance its performance and scope, solidifying RELIA as a next-generation solution for inventory management.


## Implementation Notes and Clarifications (Q&A)

**Q: Is time represented as a node or as an attribute on edges in our graph?
** **A:** Time is **not a node** in our graph; it is stored as an attribute on the edges. We use a temporal **MultiGraph** where multiple edges can connect the same store–product pair, each edge labeled with a date (e.g., `time = YYYYMMDD`). During training or evaluation, when we consider an anchor query at time *t* (a specific store–product pair at time *t*), our sampler includes only those edges **earlier** than *t*. This way, the model only sees past interactions up to the anchor time.

**Q: When performing graph convolution (message passing), do parallel edges (multiple edges between the same two nodes) lead to multiple messages?
** **A:** No – our GNN aggregates messages **per neighbor node**, not per edge. Even if a store and product have many historical transactions (parallel edges), we do *not* send one message per transaction. Instead, we **fold** the information from all pre-*t* edges into summary features on that neighbor. For example, features like “seen_before count”, “recency of last purchase”, or “degree” capture the effect of multiple past edges between the store and product before time *t*. The GNN’s attention layer then treats the product node as a single neighbor with those aggregated features. In short, edge multiplicity influences the neighbor’s features (and attention weights), but it does not spawn multiple separate messages in the convolution. (If we ever wanted one message per event, we would need a different approach, such as an edge-focused GNN or introducing **event nodes** – see below.)

**Q: What are *event nodes* in this context, and why might we use them?
** **A:** *Event nodes* are a modeling concept where each individual interaction (event) is represented as its own node in the graph, rather than just an edge. In our case, an event could be a single store selling a product on a specific date with certain quantity. If we used event nodes, the graph structure would change from a simple bipartite store–product graph to a **tripartite** graph: store → event → product. Each event node would carry attributes like the relation type (e.g. “sold”), the time of the event, units sold, promotional flag, etc. The store and product would both connect to that event node.

Using event nodes has some advantages: it allows the GNN to pass one message **per event**, preserving fine-grained temporal order and details of each occurrence. Time and event features naturally live on those event nodes, simplifying how we include temporal information. It also elegantly handles multi-edge situations (each parallel edge becomes its own event node). However, there are trade-offs: the graph becomes much larger (potentially an event node for every transaction), making sampling and training heavier. We would also need to be careful to **exclude future events** (enforce time &lt; *t*) when sampling, to prevent information leakage. Currently, **we do not use event nodes** – we opted to summarize past events through features on the existing nodes (as described above). Introducing event nodes could be a future enhancement if we decide we need true edge-level message passing fidelity.

**Q: Would modeling at the *event level* (one message per edge/event) significantly improve our model (e.g., the VN2 system)?
** **A:** It could yield some improvement, but likely **only modest gains** relative to the complexity cost. Handling each event separately might help in scenarios like very intermittent sales or cold-start products, where each transaction’s context (quantity, day of week, promotion, etc.) carries extra signal that a single aggregated feature might miss. With edge-level messages, the model’s attention could weight each past sale individually, potentially recognizing patterns (e.g., a spike every holiday, or consistent small weekly orders).

However, the downsides are significant: the graph to sample becomes much larger (more nodes = slower sampling, especially with our NetworkX-based pipeline), and the implementation becomes more complex (defining an event node schema, ensuring no future leaks, etc.). Our recommendation is to **start small**: we can approximate the benefits by adding richer **event-level features** to the existing node representation. For instance, for each store–product pair we could include features like sales in each of the last *N* weeks (or a decay-weighted sum of recent sales), effectively giving the model a vector of recent demand history without explicitly adding all events as nodes. The model’s attention or MLP could then attend to those features similarly to how it would attend to individual events. If we ever experiment with true event nodes, we’d likely restrict to the last *N* events per pair and perhaps limit to one-hop neighbors, to keep the subgraph manageable. We would only proceed with such an event-level GNN if it demonstrated a clear performance boost (for example, more than +0.01 in AUC or +0.02 in average precision) without untenable runtime costs.

**Q: A recent paper (Dwivedi *et al.*, *Relational Graph Transformer*, arXiv 2505.10960) provides recommendations for relational temporal graphs. What does it propose, in summary?
** **A:** The **Relational Graph Transformer (RelGT)** paper proposes a Transformer-based architecture designed specifically for relational and temporal graph data (such as multi-table databases represented as graphs)[arxiv.org](https://arxiv.org/abs/2505.10960#:~:text=Graph%20Transformer%20,establishing%20Graph). The key ideas from that paper are:



* **Multi-element tokenization:** Instead of representing each entity as a single node embedding, decompose each node into multiple tokens capturing different aspects (they use five: the node’s raw features, its type, its hop distance from a query, a time attribute, and a local structural descriptor)[arxiv.org](https://arxiv.org/abs/2505.10960#:~:text=Graph%20Transformer%20,Our). This preserves rich information that might be lost if everything is fused into one vector.

* **Local and global attention:** Apply Transformers on sampled subgraphs with **local attention** (nodes attend to other nodes within the sampled neighborhood) and also introduce a few **global “centroid” tokens** that can attend to all subgraph nodes[arxiv.org](https://arxiv.org/abs/2505.10960#:~:text=architecture%20combines%20local%20attention%20over,architecture%20for%20Relational%20Deep%20Learning). These global tokens are learnable vectors that provide a form of context or memory that spans across the entire database or graph, helping to capture long-range patterns or global schema information.

* **Schema/time-aware encodings:** Instead of standard positional encodings (which don’t work well on arbitrary graphs), RelGT builds representations that respect the database schema and temporal ordering. Essentially, it injects knowledge of the node types, the relationships, and time intervals directly into the token embeddings or attention mechanism, rather than relying on unstructured position encodings.

* **Performance:** In experiments on the RelBench benchmark of 21 tasks, this approach matched or outperformed traditional GNN models by up to ~18% on those tasks[arxiv.org](https://arxiv.org/abs/2505.10960#:~:text=architecture%20combines%20local%20attention%20over,architecture%20for%20Relational%20Deep%20Learning), showing that Transformers can be very effective for relational data when designed with these considerations in mind.


**Q: Is our current implementation following the Relational Graph Transformer approach?
** **A:** **Partially.** We have incorporated some of the same **principles**, but not the full design. On the plus side, our model already handles heterogeneous node types (stores, products, and we plan to add warehouses) and uses a form of **local attention** by focusing on a sampled egonet around the anchor (store–product pair) at time *t*. We also include temporal information in our features (e.g., recency, “seen before” flags anchored to time *t*) and use categorical embeddings for certain attributes – these align with RelGT’s goals of schema- and time-awareness.

However, we have **not yet implemented** RelGT’s more advanced techniques: we currently represent each node as a single combined feature vector (as opposed to multi-element tokens), we do not use any global centroid tokens, and our positional/structural encodings are simpler (hand-crafted features like degree or hop count, rather than a learned token for “hop distance” or “local structure”). In short, our approach is inspired by similar challenges, but it is a more traditional hybrid of GNN and simple attention mechanisms, whereas RelGT formalizes a full Transformer architecture for this domain.

If we wanted to move closer to the RelGT approach, we would consider breaking each node’s information into multiple tokens and possibly adding global context tokens, as described next.

**Q: What does it mean to “add multi-element tokens per node (type, hop, time bucket, local-structure)” in practice?
** **A:** This means representing each node with a **set of separate token embeddings** instead of one unified embedding. In our current model, for example, a product node might have a single feature vector combining its category, sales history features, etc. With multi-element tokenization, we would split that into multiple parts so that the Transformer can attend to each aspect independently. For a given node, we could create tokens such as:



* **Type token:** an embedding representing the node’s type (e.g. “store”, “product”, or “warehouse”). This gives the model a clear signal of the entity’s role.

* **Hop-distance token:** an embedding for the topological distance from the anchor node. For instance, a neighbor that is directly connected to the query store–product would have hop distance 1, a neighbor-of-neighbor would have hop distance 2, etc. This token helps the model distinguish immediate relations from more distant context.

* **Time-bucket token:** an embedding encoding the recency of this node’s interaction relative to time *t*. We might bucket the last interaction time into categories (e.g., “within last week”, “within last month”, “over a year ago”) and embed that. This gives a temporal context for how current or stale a connection is.

* **Local-structure token:** an embedding summarizing the node’s local graph structure. This could be something like the degree of the node (bucketed into low/medium/high degree), or whether it participates in certain motifs (e.g., triangles). It provides a sense of how connected or central the node is in the graph.

* **Feature token:** one or more tokens for the node’s actual features (continuous or categorical attributes). For a product, this might encode its category, price band, etc.; for a store, it might encode store format or region.


With this setup, if a subgraph sample has *M* nodes, instead of feeding *M* node vectors into the model, we would feed *5M* (in this example) token vectors. The Transformer can then learn to pay more attention to certain tokens (maybe the time tokens and feature tokens are crucial for forecasting, while hop or structure tokens modulate their effect). After the Transformer layers, the tokens belonging to a single node could be aggregated (e.g., via concatenation or attention pooling) back into a single representation for that node when making the final prediction. Essentially, this approach gives the model a richer, factorized representation of each node’s information.

**Q: Would adopting such multi-element tokenization likely benefit our VN2 model’s performance?
** **A:** It **might help**, but we expect any gains to be relatively modest. The benefit of multi-element tokens is that the model can more flexibly learn which aspects of a node are important for prediction, rather than relying on us to blend them correctly in a single vector. This could improve performance for cases like cold-start products or unusual stores, where, say, the “type” or “time” aspect needs to be weighted just right. It also could help the model differentiate influences: for example, it might learn that for distant neighbors (hop 2) the time since last interaction is critical, whereas for direct neighbors (hop 1) the node’s own features are more important.

However, introducing this will significantly increase the sequence length for the Transformer (multiple tokens per node means many more tokens to attend to), which raises computation time and memory. It also complicates the data pipeline (we must construct and manage these tokens). Given that our current feature engineering already captures many of these aspects (we explicitly input type indicators, hop counts, recency values, etc., albeit not as separate tokens), the improvement might be incremental.

A sensible approach would be to **pilot test** this on a smaller scale: for example, implement a variant of our model where each node is represented by, say, 3–5 tokens as described, and see if the validation metrics (AUC, average precision) improve meaningfully. If we observe a clear uptick (e.g., >1% relative gain or a noticeable boost in rare-case predictions) and it doesn’t degrade runtime too much, then it could be worth incorporating. If not, the added complexity may not be justified.

**Q: What changes to our current implementation or architecture would be needed to incorporate the ideas above (event nodes or multi-element tokens)?
** **A:** To incorporate **event-level message passing**, we would need to refactor the data schema: introduce a new event node type and alter the sampler to pull in event nodes between stores and products. Each event node would carry attributes of a single transaction and connect to the store and product involved. We’d then adjust the GNN to handle three types of nodes (store, product, event) and ensure messages can pass through event nodes appropriately. We’d also need logic to limit events to those before time *t*, and possibly to cap the number of events per pair to keep things efficient. This is a non-trivial change, affecting data processing, model architecture, and training procedure.

For **multi-element tokens**, the changes are mostly on the model input representation side. We’d construct multiple embedding matrices: e.g., one for node type, one for hop distance (with a small number of possible hops), one for time buckets, one for structure buckets, etc. In the data loader, instead of producing one feature vector per node, we produce several token indices per node. The model’s forward pass would then create embeddings for each token and feed the whole sequence into the Transformer layers. Finally, we would add a mechanism to aggregate tokens back to a node-level representation (or directly predict from token representations, depending on design). These changes would primarily affect the model definition and the feature preparation pipeline.

In both cases, careful engineering and incremental testing would be needed. The event-node approach is a bigger overhaul to the graph **schema**, whereas the multi-token approach is a substantial change to the **model encoding**. We would likely pursue one at a time (if at all), starting with the approach that promises the most gain for the least complexity (as noted, probably the tokenization is easier to try than full event nodes, or perhaps simpler still, adding richer time-bucket features without full tokenization).


---

**Q: Shifting focus to the inventory domain – Why do we sometimes not have enough *on-hand* inventory at a store?
** **A:** There can be several reasons a store ends up with insufficient on-hand inventory, and our model tries to surface contributing factors. Often, it’s a combination of supply and demand factors such as:



* **Supply delays or shortages:** Perhaps the supplying warehouse or vendor did not replenish the store in time (e.g., long lead times or missed shipments). In the graph, a store’s connection to its warehouse (or the lack of recent shipping events) would indicate this. If our model sees that a store–product edge has very few recent **incoming shipments** from the warehouse side, it may attribute the stockout risk to supply issues.

* **Sudden demand spikes:** The product might have sold much faster than usual (e.g., due to a local event or promotion), depleting stock unexpectedly. Our model captures demand spikes through features like recent sales velocity or by comparing against peer stores (via the product node’s connections to other stores). A surge in the product’s sales at similar stores might signal a trend that caused this store to run out if it wasn’t anticipated.

* **Data or trigger issues:** In some cases, the system might not have registered an order that should have been placed (for instance, a sensor or integration failure could mean the automatic reorder didn’t trigger). Our approach includes an “alert” mechanism for **missing triggers** – if the model strongly expects an order should have happened (based on low inventory and high forecasted demand) but no order/ship event is recorded, it flags this as a likely cause.

* **Historical understocking patterns:** Some store–product combinations might consistently be under-forecasted (perhaps due to seasonality or new product introduction). The model’s attention over the store’s neighbors (other related products, or the product’s performance in other stores) can help explain that the on-hand inventory was low because the system didn’t predict the demand well.


When a user asks, “Why don’t I have enough on hand?”, our system can provide an explanation combining these factors – for example: *“Store X is out of Product Y because the last delivery from Warehouse W was 10 days ago (longer than the usual lead time of 3 days), and sales in the past week spiked 50% above average. No replenishment trigger was recorded when stock fell below threshold.*” Such explanations come from the model’s learned features and attention weights, augmented by simple business rules and Shapley-value analyses on the features if needed.

**Q: Can the system predict “Am I going to run out of inventory today?” (stockout risk in the very short term)?
** **A:** Yes, we aim to estimate the probability of a **stockout within the next day** (or any given short horizon) for each store–product. To do this, we model the demand distribution for that horizon and compare it against the inventory available. Concretely, for each store–product we compute:



* **Current inventory position:** on-hand units at the store, plus any incoming units that will arrive within the next day (if there is a shipment already en route). Let’s call this StodayS_{\text{today}}Stoday​.

* **Expected demand distribution in the next day:** We use recent data (e.g., daily sales in similar time periods) to estimate how likely a sale is on a given day and how many units might sell. This can be a **zero-inflated** model – for many products a given day could have zero sales with some probability, and if sales occur, there’s a distribution for the positive demand. Based on historical daily sales, we estimate parameters like the probability of any sale pdp_dpd​, and the mean/variance of sales given that there is a sale (denoted μd+,σd+\mu^+_d, \sigma^+_dμd+​,σd+​ for positive-demand days). From this, we can derive an expected demand over the day (let’s call it μ1day\mu_{1\text{day}}μ1day​) and a plausible range (variance σ1day2\sigma_{1\text{day}}^2σ1day2​).

* **Stockout probability:** Given the distribution of demand, we compute the probability that demand will exceed StodayS_{\text{today}}Stoday​. For example, if we approximate the demand over one day as a normal distribution (using the mean and variance above for simplicity), we can calculate a *z*-score: z=Stoday−μ1dayσ1dayz = \frac{S_{\text{today}} - \mu_{1\text{day}}}{\sigma_{1\text{day}}}z=σ1day​Stoday​−μ1day​​. A high positive *z* means stock is more than enough to cover expected demand; a negative *z* means stock is likely insufficient. We then convert this to a probability: P(stockout)≈P(demand>Stoday)=1−Φ(z)P(\text{stockout}) \approx P(\text{demand} > S_{\text{today}}) = 1 - \Phi(z)P(stockout)≈P(demand>Stoday​)=1−Φ(z) (where Φ\PhiΦ is the standard normal CDF).


The system would present the result as, for instance, *“There is a 30% chance you will stock out of Product Y by end-of-day.”* This probability-based approach gives a more nuanced view than a simple yes/no flag, and it accounts for uncertainty in demand. The model’s forecasts (mean and variance of demand) are derived from its learned representation of that store–product and similar ones (so it can incorporate factors like trend, seasonality, and recent acceleration or deceleration in sales).

**Q: How does the system determine the top 5 or 10 products (UPCs) that need action today?
** **A:** The system will rank products by urgency and potential impact, to highlight those that most need attention (for example, to reorder or to shift inventory). Two main considerations drive this ranking:



* **Stockout risk and demand impact:** We consider the probability of stockout (as described above) *and* the expected demand of the product. If a product is likely to run out and its demand (or revenue) is high, it gets a high priority because a stockout there would cause significant lost sales. This can be quantified as an expected **shortage cost** or lost sales metric. Essentially, Risk Score≈P(stockout)×expected sales (or profit) during the shortage period\text{Risk Score} \approx P(\text{stockout}) \times \text{expected sales (or profit) during the shortage period}Risk Score≈P(stockout)×expected sales (or profit) during the shortage period.

* **Inventory imbalance or overstock risk:** We also consider if there’s **excess inventory** risk for a product, although the question is phrased around “action” which usually implies avoiding stockouts. Primarily we focus on shortage, but a product with extreme overstock might also deserve action (e.g., mark down or stop further orders). For now, the “top actions” list is more about preventing near-term stockouts and critical shortages.


Using these factors, each store will get a list like: *Product A (85% stockout risk, high demand) → Order now; Product B (60% risk, medium demand) → Consider ordering; Product C (stockout risk low but a big sales item in promotion next week) → Flag for check*, etc. We typically take the top N (5 or 10) by this priority score. This is essentially implementing a simplified **newsvendor priority**: items where the expected cost of understocking (shortage cost) is highest are bubbled to the top. The output is a daily “Act Now” list of SKUs for each store or manager, along with recommended actions (order, transfer, expedite shipment) and a brief reason (e.g., “Projected to sell out in 2 days with no replenishment on the way.”).

**Q: Sometimes it seems an order from the store to the warehouse wasn’t triggered when it should have been (perhaps due to missing information). How does the system handle cases where no order was placed but inventory was needed?
** **A:** We have a safeguard in place for detecting **missing reorder triggers**. The model knows the patterns of orders and shipments: for example, if a product usually reorders weekly or when stock falls below a certain threshold, the absence of an expected order is an anomaly. If our model predicts a high likelihood that an order *should* have occurred (because inventory dropped low and demand was forecasted high), but we see no corresponding order or shipment event in the data, we flag that situation. This could be due to data issues (the store’s system failed to send the order, or the warehouse system didn’t log it) or process issues (a human oversight).

The system’s response is twofold: **alert and conservative action**. It will alert that “Store X likely needed a replenishment of Product Y that did not occur as expected.” At the same time, to prevent harm, it can recommend a conservative remedial action – for instance, *“Place an urgent order for Product Y to cover the next few days as a safeguard.”* Essentially, the model acts as a **watchdog** that catches scenarios where the usual pull signal might have failed to register. This reduces the chance that a data glitch causes a shelf-out. In addition, these instances are recorded so that process owners can investigate and fix any underlying integration issues.

**Q: We sometimes face excess inventory in certain stores. How will the system help with that?
** **A:** Excess inventory (overstock) is the flip side of stockouts – too much product sitting in the store. Our approach addresses this in a few ways:



* **Identifying excess:** The model can predict not just if you’ll run out, but also if you’re likely to **not sell a significant portion of your on-hand stock** in the foreseeable future. If a store has, say, 100 units on hand but expected sales are 5 per week, that’s excessive. Such items might not appear in the “top 5 action” list for shortages, but they can be flagged for review in an **excess report**.

* **Transfers and rebalancing:** If one store has excess and another store is at risk of shortage for the same product, the system can suggest a **redistribution** (store-to-store transfer) if it’s feasible and cost-effective. For example, *“Store A has 20 extra units of Product Z, while Store B might stock out; consider transferring 10 units to Store B.”* This way, we alleviate the excess at one location and avoid a shortfall at another.

* **Ordering guidance:** The model’s recommendations inherently aim to prevent excess by not ordering too much in the first place. It does this by forecasting demand and carrying costs (the classic newsvendor balance). For instance, if a product is slow-moving (class C item), the model will only trigger a refill if the stock is truly needed, and it will aim for smaller order quantities, whereas for a fast mover (class A), it’s more liberal. This policy is sometimes implemented as different service levels or β\betaβ values in a newsvendor formula for A/B/C categories – effectively a higher in-stock probability target for important items than for less important ones.


If an excess has already happened, besides transfers, the system might recommend actions like **promotions or markdowns** to accelerate sales, but those are typically outside the scope of an automated replenishment tool and more of an advisory to planners. In summary, the system won’t just push inventory – it also watches for places where it should **pull back or hold off** to avoid piling up stock.

**Q: Our current model is a “pull” system (stores order from warehouses when needed). How can we enable a more “push” system, where warehouses proactively send inventory to stores in anticipation of needs?
** **A:** To shift from pure pull to a more **anticipatory push**, we need to incorporate warehouse nodes and logic for proactive shipping into the model. Here’s how we propose doing that:



* **Augment the graph with warehouse entities:** We introduce **warehouse nodes** alongside store and product nodes. There will be new edge types with their own relationships and attributes. For example, a **ship** edge from a warehouse to a store (warehouse → store) when inventory is delivered, and an **order** edge from a store to a warehouse (store → warehouse) when a requisition is placed. These edges would carry quantities and dates. We’d also include static or slowly changing edges like **stock** (warehouse → product, indicating on-hand inventory at the warehouse for that product), and perhaps edges for **lead time** (warehouse ↔ store, carrying the typical delivery lead time or transit time between that warehouse and store). This extended schema allows the model to understand supply availability and delays.

* **Forecast at the store, allocate from the warehouse:** The model can use its demand forecast for each store–product (say, expected demand over the next *H* days) to anticipate what the store *will* order. Instead of waiting for the store’s order, the warehouse could decide to **ship proactively**. We would compute an optimal or heuristic push quantity q(w,s,p)q(w, s, p)q(w,s,p) for each warehouse–store–product tuple. This might involve solving a small allocation problem: the warehouse knows its own on-hand inventory of product *p* and the needs of all the stores it serves, and it tries to allocate that inventory in advance. The objective would be to minimize the combined expected stockout costs and holding costs (and shipping costs) across the network. In practice, a simpler heuristic might suffice: *push to the stores with the highest stockout risk first*, capped by available stock at the warehouse and by what each store can receive (no over-filling a store).

* **Lead times and coordination:** Because a store can be served by multiple warehouses (and warehouses serve multiple stores), the push logic must also decide *from which warehouse* to send product. The model will consider **lead times** and current warehouse stock. For example, if Warehouse W1 can deliver overnight but is low on Product P, and Warehouse W2 has plenty of P but a 3-day lead time, the system might push what W1 has now and suggest W2 to send more to arrive a bit later. It will also look at potential conflicts, like two stores both needing the same limited batch of inventory from one warehouse – in which case it allocates optimally or flags if the warehouse itself needs resupply.

* **Preventing over-push:** A push system must also guard against sending too much. The model will respect constraints like store capacity (no sending more than the shelf or backroom can handle) and case-pack minimums (ship in full case increments). It will also only push when the confidence in demand is high enough – effectively, the model’s threshold for pushing could be tuned to avoid a flurry of unnecessary transfers.


By adding this push capability, the warehouse becomes an active participant that looks at the **whole network** (all its stores) and tries to balance inventory. The result should be that stores get inventory *before* they run out, ideally without needing to manually place an order, especially for high-volume products. It’s like giving the warehouse a “lookahead” based on the model’s foresight. The core engine (our model) supporting this needs the extended graph and some optimization logic layered on top of the predictions.

**Q: How do we handle the scenario where a store has multiple possible supplying warehouses, or a warehouse supplies many stores?
** **A:** This is the **multi-warehouse, multi-store allocation** problem, which our approach addresses through the graph relationships and the push logic:



* Each store–warehouse pair has a known **lead time** and possibly a capacity or preference. The model can learn or be given these as features (for instance, Warehouse W1 is primary for Store S, and W2 is secondary backup).

* When predicting needs, the model will evaluate all warehouses that can reach a store in time. If one warehouse is low on stock or slow, another might be chosen. We effectively compute the **marginal benefit** of sending one more unit from each candidate warehouse to the store. The warehouse that can fulfill the need with the least expected cost or highest service level will be assigned the task in the push plan.

* On the warehouse side, if a warehouse has limited inventory of a product and several stores need it, we again use an allocation strategy to maximize overall service level. This could involve prioritizing the store with the most urgent need or splitting inventory proportional to demand and criticality. The system can flag if a warehouse cannot meet all demands so that upstream suppliers or planners know to rush more stock to that warehouse.


In summary, the model doesn’t just make independent decisions per store; it also considers the network structure. The graph approach naturally lends itself to this: a warehouse node connected to many store nodes can aggregate needs and distribute resources, and a store node connected to multiple warehouse nodes can draw from the best source. Our implementation will likely include a post-processing step that takes model forecasts and solves a small optimization (or heuristic) for the allocation, as described earlier, to handle these multi-warehouse scenarios.

**Q: How do we incorporate additional signals or data (like richer demand history or stockout events) without exploding the graph complexity?
** **A:** We plan to use **“event buckets”** and additional features rather than explicitly adding every event as a node or every metric as an edge. For example, instead of an event node for each day of sales, we can attach to each store–product pair a summarized history: say, sales in each of the last 4 weeks, or flags indicating if the item stocked out in the last week. These become features in the model. In practice, the model could treat each time bucket as a small token or just as a vector input to an MLP that outputs an embedding. This gives a flavor of temporal richness (the model can see if sales are increasing week-over-week, or if there was a zero sales week indicating a possible stockout) without needing to create new nodes for each week.

Similarly, we add **state features at prediction time**. When the model is assessing a store–product at time *t*, we feed in the current known quantities: how many are on hand at the store right now, how many units are on order or in transit to the store, the product’s case pack size (minimum order quantity), maybe the shelf capacity or presentation minimum (so the model knows if the current stock is below a planogram minimum). These features inform the model’s prediction and recommendation, again without requiring structural graph changes.

In summary, by using feature engineering on nodes/edges (including temporal bucket features and current state features), we enrich the model’s input substantially while keeping the graph structure simpler. This avoids the combinatorial blow-up of adding, say, an edge for every single daily inventory reading or an “on hand” node, etc. It’s a pragmatic way to capture more signal with minimal schema changes.

**Q: What kind of alerts or QA (quality assurance) outputs does the system provide to users and operators?
** **A:** We’ve touched on a couple of these already. The system will not just silently compute numbers – it will produce actionable alerts/reports each day. Key ones include:



* **“Act Now” list:** As discussed, a daily list of top K products per store (or overall) that likely need immediate intervention (order more, push from warehouse, transfer, etc.). This list is essentially answering “What do I need to take action on today?” It includes the item, the recommended action (and quantity, if applicable), and reason codes. Reason codes could be generated from the model’s rationale: e.g., *“High stockout risk (85%) within 3 days”*, *“No incoming shipment scheduled”*, *“Recent sales surge”*, etc.

* **Missing trigger alerts:** If the model detects a probable missing order event (as described earlier), it will flag it specifically. That alert might say *“Order likely missed: Product P at Store S should have been reordered yesterday (inventory fell to 0), but no order was recorded.”* This prompts a user to investigate and also to manually correct any immediate problem.

* **Excess inventory alerts:** A periodic report or alert for items that have far more stock than needed. This might be weekly rather than daily, and could say *“Potential overstock: Store S has 60 days of supply for Product Q (class C item). Consider reducing orders or transferring stock.”
*
* **Data quality/QA alerts:** If there are anomalies in the data – for example, a store that usually sells 100 units a day shows 0 sales for three days in a row, the system can flag that as *“check data for Store S/Product P – sudden drop to zero may indicate an issue (or the product went out of stock without record).”* This overlaps with the stockout alert, but is more explicitly pointing to data gaps or suspicious patterns.


All these alerts are part of ensuring the model’s outputs are **actionable** and that the system is trustworthy. We want users to not just get predictions, but also guidance and early warning on both opportunities and problems.

**Q: Are there business rules or policy constraints integrated into the model’s recommendations?
** **A:** Yes, we incorporate several **policy guardrails** to ensure the recommendations make practical sense and align with business strategy:



* **Service level targets by item class:** If the business classifies products into A, B, C categories (A being high-priority items with high service level targets, C being lower priority), we adjust the model’s threshold or cost parameters accordingly. In a newsvendor context, this means using a higher target in-stock probability (or lower stockout penalty relative to holding cost) for A items than for C items. So A items will be kept at a higher level of availability, while C items might be allowed to run leaner.

* **Order quantity constraints:** We respect case packs and minimum order quantities. If a product’s case pack is 24 units, the model won’t recommend ordering 5 units; it will suggest 24 (or some multiple) at a time. Similarly, we can impose **max caps** if needed – e.g., don’t send more than 4 weeks of supply in one go, to avoid overstocking, unless explicitly directed.

* **Space and display considerations:** We ensure that recommendations do not exceed what the store can handle. If a store can only display 50 units of a product and has no extra backroom space, pushing it 200 units is not sensible. The model (or a post-process) will cap suggestions to reasonable limits (like not exceeding max shelf capacity or some multiple of daily sales cover).

* **Preventing oscillations and unnecessary churn:** To avoid the system triggering tiny orders too frequently (which can increase logistics cost), we might enforce a rule like a minimum time between reorders for slow-moving items, or a minimum threshold of expected shortage before an action is taken. This smooths out the plan so that, for example, we don’t ship something today only to pull it back tomorrow.

* **Excess mitigation policies:** As mentioned, if a store already has excess, the model might hold off further replenishment (even if the pure forecast might have suggested some) until that excess is burned off, unless there is a risk of a stockout due to timing.


These policy layers are configurable and can be adjusted. They act as a sanity-check on the raw model output, marrying the advanced forecasting with real-world operational constraints.

**Q: How will we evaluate and measure the success of this enhanced system?
** **A:** We will evaluate the model and system on both **prediction accuracy** and **decision outcomes**:



* **Predictive performance:** We’ll continue to track metrics like AUC (Area Under the Curve) and Average Precision for the model’s core predictions (e.g., whether an item will stock out or whether an order will be needed). We’ll also look at calibration of the probability forecasts – for instance, if we say an item has a 30% stockout risk, over many such items, about 30% should actually stock out (this can be measured by a Brier score or Expected Calibration Error). Good calibration means the probabilities can be trusted.

* **Service level and fill rate:** On the outcome side, we’ll measure how often stores actually run out of stock with the new system in place versus before. The **fill rate** (percentage of demand met from shelf stock) should improve. We’ll also check if we maintain target service levels for different item classes.

* **Inventory cost metrics:** We’ll compare the total inventory on hand and stockout costs to a baseline (like the current system or a heuristic). The newsvendor principle is about balancing holding cost against stockout cost, so we’ll compute the expected cost of inventory (holding + shortage costs) under the new system vs. the old. A successful implementation should reduce the sum of these costs, or achieve the same cost for a higher service level.

* **Days of cover stability:** We expect the system to lead to more stable **days-of-inventory** on hand. We can track how variable the inventory levels are and aim to reduce panic last-minute orders or feast-and-famine cycles.

* **User-visible improvements:** If this is a tool for planners or store managers, we can track the adoption and effectiveness of the recommendations. For example, the **precision and recall of the “Act Now” alerts** – do the items we flagged indeed turn out to need intervention (high precision), and did we catch most of the actual problems that occurred (recall)? Also, we can gather feedback: are the explanations helpful, does it reduce workload, etc.

* **Benchmarking advanced methods:** If we try advanced modeling tweaks (like the RelGT-inspired tokenization or event nodes), we will do A/B tests or offline experiments to see if they improve the predictive metrics (AUC, etc.) or the ultimate business metrics. Any change to the model is justified by a combination of statistical improvement and practical impact in simulations.


By monitoring these metrics, we’ll ensure the system is actually delivering value: fewer stockouts, less excess, and clearer guidance for the inventory management team. Each increment (be it the push logic, new features, or transformer architecture improvements) will be validated against these criteria. The goal is a well-balanced system that moves us closer to an optimal inventory practice (and in doing so, also aligns with cutting-edge research insights like RelGT, but always with an eye on real-world results).

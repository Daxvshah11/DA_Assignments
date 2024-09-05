# REPORT

## `AOI`

### Pre-Processing

>* Dropped _`DOL VID`_ column as it was unique for all the rows and will not give us any kind of pattern nor can it be generalised
>
>* Dropped _`State`_ as there was only 1 Unique value for it so no point in generalising or finding pattern related to it
>
>* Dropped _`Postal Code`_ as it was redundant with columns like City & Counties
>
>* Dropped _`Legilative District`_ since not useful in understanding data in simpler terms
>
>* Dropped _`Position`_ as it was completely useless since that level of granularity was not required
>
>* Generalised _`City`_ column to _`Location`_ and considered City, County, State as the initial address. Ignored Cities in order to generalise them to counties
>
>* _`Brand`_ and _`Model Year`_ correlations shows us that, top car manufacturers from the dataset such as Tesla, Nissan, Chevorlet and Ford have some distinct patterns with the year manufactured, which helped us splitting Model year values into 3 categories.
>
>* High co-relation between 2 columns suggests:
 >   * They are related to each other and can be used to predict one another
 >   * For example, if we have 2 columns `A` and `B` and they have high correlation, then we can predict the value of `A` if we know the value of `B` and vice versa.
>
>* Using the `1:1` co-relation matrix, tried to gauge the redundancies in the columns and dropped the columns which were highly co-related with each other


### Generalisations

> * Generalised car brands according to their origin countries, regions they belong to and the type of cars they manufacture
>
> * Generalised the counties according to the cardinal directions they belong to in that state
>
> * Generalised the manufacturing years into `EV Trend Eras` to understand the trends in the rise and fall of particular types of vehicles
>
> * Also generalised precentages and counts or means and modes etc according to the needs of that column
    >      * _eg. generalising $>=60\%$ of BEV manufactured vehicles of a particular brand region to be called as a BEV Dominant Region_




## `BUC`

### Pre-Processing

> * Removing all the rows with even a single NaN value as they are very less in number relatively so will not affect the outcomes
>
> * Dropping multiple columns similar to those in `AOI` as they are not adding up to the final analysis. They are namely:
    >   * DOL VID
    >   * 2020 Census Tract
    >   * Vehicle Location
    >   * Legislative District
    >   * Base MSRP
    >   * City
    >   * State
    >   * Postal Code
>
> * Re-ordered the columns in descending order of their cardinality in order to make them get pruned quicker and earlier in the algorithm


### In-Memory BUC Algo

> * Implemented BUC which is not dependent on the allotted memory (main memory). Assumes that all the CUBES will directly fit into the memory and running it recursively
>
> * It includes __2 codes__:
    >   * BUC `with minsup`
    >   * BUC `without minsup`
>
> * Ran the BUC without minsup as well as the one with minsup (on multiple diff minsup values) and compared their runtime performance results as shown in the plot


### Out-Memory BUC Algo

> * Implemented BUC which is dependent on the allotted memory (main memory). No assumption regarding all the CUBES will fitting directly into the memory
>
> * To make the implementation more efficient, used `paging` to store the intermediate results and then merge them to get the final result
>
> * Using disk storage to perform the calculations and computations of BUC with memory limitations
>
> * Ran this version of BUC (with minsup implementation) on 6 different values of `memory limit` & 6 different values of `minsup` altogether and compared their runtime performance results as shown in the plot


### Optimization Technique

Using _`MINSUP`_ as an optimization technique:
> * `minsup` is a parameter which is used to prune the search space of the BUC algorithm when going recursively
>
> * It is used to remove the cubes which have a `row count` less than the minsup value
>
> * This helps in reducing the number of cubes that need to be calculated and hence reduces the time complexity of the algorithm
>
> * This is a very useful technique when the data is very large and the number of cubes is very high
>
> * Overall, improves the runtime & performance
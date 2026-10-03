# Theory Background

## 3.1 Set functions and submodularity
- **Submodularity**: The retrieval objective $F_\lambda$ is submodular because the facility location term $\sum_i \max_{j\in S} s(i,j)$ exhibits diminishing returns. Adding a case covering an already-covered legal issue adds less value than covering a new issue.
- **Monotonicity**: Since $s(i, j) \ge 0$, $\max_{j\in S} s(i,j)$ is non-decreasing as $S$ grows. The modular relevance term is also non-negative and monotone.

## 3.2 The Objective
$$ F_\lambda(S) = (1 - \lambda) \cdot \frac{1}{K} \cdot \sum_{d\in S} \tilde{r}(d)  +  \lambda \cdot \frac{1}{N} \cdot \sum_{i\in P} \max_{j\in S} s(i,j) $$
- **Term 1**: Modular (Relevance).
- **Term 2**: Facility Location (Coverage).
- A non-negative combination of monotone submodular functions is monotone submodular, hence $F_\lambda$ has the $1 - 1/e$ greedy approximation guarantee.

## 3.3 Algorithms
- **Greedy**: Chooses the element with maximum marginal gain at each step.
- **Lazy Greedy**: Uses a max-heap of upper bounds to evaluate only the top elements, saving evaluations since marginal gains are submodular (they only shrink).
- **Stochastic Greedy**: Evaluates on a random sample to achieve a fast randomized approximation.

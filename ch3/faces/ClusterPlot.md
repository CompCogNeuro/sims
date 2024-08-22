A **cluster plot** displays the similarity relationships among a set of patterns in a hierarchically clustered tree diagram. It is often the most useful way of visualizing high-dimensional similarity structure. See the taDataAnal page for information on how to generate a cluster plot.

The algorithm for creating the cluster plot is a recursive process of grouping together the most similar patterns, and then grouping together groups of patterns into higher-order clusters. It is "greedy" in just grouping the closest items or groups, and proceeds until there is nothing left to be grouped that isn't already grouped.

To read the plot, start at the right-hand side, which contains the most similar "leaves" in the tree. Items that share a common vertical bar with horizontal lines connecting them to that common vertical bar are all grouped together at the same level of similarity. Often this is a pair of items, but if multiple items have an identical level of similarity among themselves, then they can all be grouped together. Moving one level back to the left, groups that are themselves grouped together in a cluster have greater shared similarity among their respective items relative to other such groupings. And so on..

The length of the horizontal lines indicates the distance between items within the group. Typically this is computed using a *Euclidean* distance metric (i.e., square root of sum-of-squared differences), but many others are available.

If you see just a vertical line and no horizontal line segments for the leaves, this means that those items have zero distance between them.

There is no absolute meaning to the length of horizontal position of clusters between disconnected clusters -- the distance is only accurate for items within a shared cluster.


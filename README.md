# EpiGen: generator of epidemic cascades on synthetic graphs

This Python package can be used to generate discrete time epidemic cascades with either the SIR Model or the SIS Model.
It can also reduce to the SI model, by setting the probability of recovery to 0.
The software then can generate observations on the epidemic cascades.

As for contact graphs, either static (fixed) contact graphs or dynamical contact graphs can be used.  The package can generate static graph on its own (available for Random Regular Graphs, Erdős–Rényi (gnp), Barabasi-Albert, Watts-Strogatz, and more), or it can be provided with a NetworkX graph
For dynamical graph, the contacts can be provided with a `npz` file containing a list of contacts, or the software can create a dynamical contact network in which a new graph is generated for each day of contacts.

Graph generation is made with the NetworkX package, while epidemic generation requires the Numba package. The Pandas package is used for the handling of observations.
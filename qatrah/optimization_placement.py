from typing import Any, Dict, List, Optional, Tuple

import dimod
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import oopnet as on
import pandas as pd
import plotly.graph_objects as go
from matplotlib import rcParams
from qiskit import Aer
from qiskit.algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer


def convert_oopnet_junctions_to_df(net):
    df = pd.DataFrame(columns=["node", "xcoord", "ycoord", "elevation", "demand"])
    for j in on.get_junctions(net):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "node": j.id,
                            "xcoord": j.xcoordinate,
                            "ycoord": j.ycoordinate,
                            "elevation": j.elevation,
                            "demand": j.demand,
                        }
                    ]
                ),
            ]
        )

    df = df.reset_index(drop=True)
    return df


def convert_oopnet_nodes_to_df(net, default_elevation=0, default_demand=0):
    df = pd.DataFrame(columns=["node", "xcoord", "ycoord"])
    for j in on.get_nodes(net):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "node": j.id,
                            "xcoord": j.xcoordinate,
                            "ycoord": j.ycoordinate,
                            "elevation": default_elevation,
                            "demand": default_demand,
                        }
                    ]
                ),
            ]
        )

    df = df.reset_index(drop=True)
    return df


def get_df_from_oopnet_nodes(net):
    nodes = convert_oopnet_nodes_to_df(net)
    junctions = convert_oopnet_junctions_to_df(net)
    nodes = nodes.merge(junctions, on="node", how="outer")
    drop_y(nodes)
    rename_x(nodes)
    return nodes


def convert_oopnet_pipes_to_df(net):
    df = pd.DataFrame(
        columns=[
            "id",
            "node1",
            "node2",
            "length",
            "diameter",
            "roughness",
            "minorloss",
            "status",
        ]
    )
    for j in on.get_pipes(net):
        new_row = pd.DataFrame(
            [
                {
                    "id": j.id,
                    "node1": j.startnode.id,
                    "node2": j.endnode.id,
                    "length": j.length,
                    "diameter": j.diameter,
                    "roughness": j.roughness,
                    "minorloss": j.minorloss,
                    "status": j.status,
                }
            ]
        )
        df = pd.concat([df, new_row])

    df = df.reset_index(drop=True)
    return df


def drop_y(df):
    to_drop = [x for x in df if x.endswith("_y")]
    df.drop(to_drop, axis=1, inplace=True)


def rename_x(df):
    for col in df:
        if col.endswith("_x"):
            df.rename(columns={col: col.rstrip("_x")}, inplace=True)


def create_graph(edges):
    G = nx.from_pandas_edgelist(
        edges,
        "node1",
        "node2",
        edge_attr=["length", "diameter", "roughness"],
        create_using=nx.DiGraph(),
    )
    return G


def add_attributes(G, nodes):
    nx.set_node_attributes(
        G,
        nodes[["xcoord", "node"]].set_index("node", drop=True)["xcoord"].to_dict(),
        "xcoord",
    )
    nx.set_node_attributes(
        G,
        nodes[["ycoord", "node"]].set_index("node", drop=True)["ycoord"].to_dict(),
        "ycoord",
    )
    nx.set_node_attributes(
        G,
        nodes[["demand", "node"]].set_index("node", drop=True)["demand"].to_dict(),
        "demand",
    )
    nx.set_node_attributes(
        G,
        nodes[["elevation", "node"]]
        .set_index("node", drop=True)["elevation"]
        .to_dict(),
        "elevation",
    )

    return G


def generate_centrality_factor(G, edges_attr):
    bb = nx.edge_betweenness_centrality(G, normalized=True)
    edges_attr["bce"] = 0
    for s, t in bb.keys():
        edges_attr.loc[
            edges_attr[(edges_attr["node1"] == s) & (edges_attr["node2"] == t)].index,
            "bce",
        ] = bb[(s, t)]
    return edges_attr


def diameter_length_factor(edges_attr):
    edges_attr["Z"] = edges_attr["length"] / 2 + edges_attr["diameter"] / 2
    return edges_attr


def create_edge_weight(edges_attr, A, B):
    edges_attr["weight"] = A * edges_attr["bce"] + B * edges_attr["Z"]
    return edges_attr


def generate_accessibility_factor(edges_attr, P):
    edges_attr["access"] = np.random.choice(
        [0, 1], size=edges_attr.shape[0], p=[1 - P, P]
    )
    return edges_attr


def plot_network(G, plot_name="Sensors graph"):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0 = G.nodes[edge[0]]["xcoord"]
        y0 = G.nodes[edge[0]]["ycoord"]
        x1 = G.nodes[edge[1]]["xcoord"]
        y1 = G.nodes[edge[1]]["ycoord"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x = G.nodes[node]["xcoord"]
        y = G.nodes[node]["ycoord"]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>Network graph made with Python",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


class SensorPlacement:
    REQUIRED_PARAMS = ["nodes", "edge", "nb_sensors"]

    def __init__(self, params) -> None:
        for label in self.REQUIRED_PARAMS:
            if label not in params:
                raise ValueError(f"Please provide {label} in params.")

        self.params = params.copy()
        self.params["nodes"] = self.params["nodes"]
        self.params["nb_sensors"] = self.params["nb_sensors"]
        self.params["edges"] = self.params["edge"]
        self.params["n_nodes"] = len(self.params["nodes"])
        self.params["n_edges"] = len(self.params["edges"])
        self.params["A"] = self.params["A"] if "A" in self.params else 1
        self.params["B"] = self.params["B"] if "B" in self.params else 30
        self.params["C"] = self.params["C"] if "C" in self.params else 5
        self.params["D"] = self.params["D"] if "D" in self.params else 1

        self.results: Dict[str, SensorPlacementResults] = {}
        self.offset: Optional[float] = None
        self._quadratic_program: Optional[QuadraticProgram] = None
        self._lin_terms: Optional[Dict[str, float]] = None
        self._quadratic_terms: Optional[Dict[Tuple[str, str], float]] = None

    @property
    def offset(self) -> float:
        if self._offset is None:
            (
                self._lin_terms,
                self._quadratic_terms,
                self.__offset__,
            ) = self.generate_coefficent()
        return self._offset

    @property
    def lin_terms(self) -> Dict[str, float]:
        if self._lin_terms is None:
            (
                self._lin_terms,
                self._quadratic_terms,
                self._offset,
            ) = self.generate_coefficent()
        return self._lin_terms

    @property
    def quadratic_terms(self) -> Dict[Tuple[str, str], float]:
        if self._quadratic_terms is None:
            (
                self._lin_terms,
                self._quadratic_terms,
                self._offset,
            ) = self.generate_coefficent()
        return self._quadratic_terms

    @property
    def quadratic_program(self) -> QuadraticProgram:
        if self._quadratic_program is None:
            self._quadratic_program = self.gen_quadratic_program()
        return self._quadratic_program

    def generate_coefficent(
        self,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
        nodes = self.params["nodes"]
        edges = self.params["edges"]
        nb_sensors = self.params["nb_sensors"]
        A = self.params["A"]
        B = self.params["B"]
        C = self.params["C"]
        D = self.params["D"]

        lin_terms: Dict[str, float] = {}
        quadratic_terms: Dict[Tuple[str, str], float] = {}

        demand_sum, access_sum = 0, 0
        for index, row in nodes.iterrows():
            node = row["node"]
            x = row["xcoord"]
            y = row["ycoord"]
            demand = row["demand"]
            access = row["access"]
            demand_sum += 1 - demand
            access_sum += access
        demand_sum *= C
        access_sum *= D
        weights = 0
        for index, row in edges.iterrows():
            weights += row["weight"]

        offset = demand_sum + access_sum + (nb_sensors ^ 2) * B + A * weights
        for index, row in edges.iterrows():
            lin_terms[row["node1"]] = lin_terms.get(row["node1"], 0) - A * row["weight"]
            lin_terms[row["node2"]] = lin_terms.get(row["node2"], 0) - A * row["weight"]

        for index, row in nodes.iterrows():
            node = row["node"]
            lin_terms[node] = lin_terms.get(node, 0) - 2 * nb_sensors * B

        for i, row in nodes.iterrows():
            for j in range(i, nb_sensors):
                quadratic_terms[(row["node"], nodes[j]["node"])] = (
                    quadratic_terms.get((row["node"], nodes.iloc[j]["node"]), 0) + 2 * B
                )

        for i, row in edges.iterrows():
            quadratic_terms[row["node1"], row["node2"]] = (
                quadratic_terms.get((row["node1"], row["node2"]), 0) + A * row["weight"]
            )

        for i, row in nodes.iterrows():
            quadratic_terms[(row["node"], row["node"])] = (
                quadratic_terms.get((row["node"], row["node"]), 0) + B
            )

        return lin_terms, quadratic_terms, offset

    def gen_quadratic_program(self) -> QuadraticProgram:
        qubo = QuadraticProgram(name="sensor placement")
        n = self.params["n"]
        N = self.params["N"]

        qubo.binary_var_dict(n, key_format="v{}")
        for i in range(n):
            qubo.binary_var_dict(
                key_format="z" + str(i) + ",{}", keys=list(range(N + 1))
            )

        qubo.minimize(
            linear=self.lin_terms,
            quadratic=self.quadratic_terms,
            constant=self.offset,
        )
        return qubo


class SensorPlacementResults:
    def __init__(self, results: Any) -> None:
        self._results: Any = results

    @property
    def results(self) -> Any:
        return self._results

    def _run_gate_based_opt(
        self,
        quantum_instance: Optional[QuantumInstance] = None,
        label: str = "qaoa",
        opt_type: str = "qaoa",
    ) -> SensorPlacementResults:
        opt_types = {"qaoa": QAOA, "vqe": VQE}
        quantum_algo = opt_types[opt_type]

        if quantum_instance is None:
            backend = Aer.get_backend("qasm_simulator")
            quantum_instance = QuantumInstance(
                backend=backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        quadprog = self.quadratic_program

        _eval_count = 0

        def callback(eval_count, parameters, mean, std):
            nonlocal _eval_count
            _eval_count = eval_count

        solver = quantum_algo(
            quantum_instance=quantum_instance,
            callback=callback,
        )

        optimizer = MinimumEigenOptimizer(solver)

        result = optimizer.solve(quadprog)

        self.results[label] = SensorPlacementResults(
            result, {"eval_count": _eval_count}
        )
        return self.results[label]

    def run_qaoa(
        self, quantum_instance: Optional[QuantumInstance] = None, label: str = "qaoa"
    ) -> SensorPlacementResults:
        return self._run_gate_based_opt(
            quantum_instance=quantum_instance,
            label=label,
            opt_type="qaoa",
        )

    def run_vqe(
        self,
        quantum_instance: Optional[QuantumInstance] = None,
        label: str = "vqe",
    ) -> SensorPlacementResults:
        return self._run_gate_based_opt(
            quantum_instance=quantum_instance,
            label=label,
            opt_type="vqe",
        )

    def run_classical(self, label: str = "classical") -> SensorPlacementResults:
        solver = NumPyMinimumEigensolver()

        optimizer = MinimumEigenOptimizer(solver)

        result = optimizer.solve(self.quadratic_program)

        self.results[label] = SensorPlacementResults(result)
        return self.results[label]

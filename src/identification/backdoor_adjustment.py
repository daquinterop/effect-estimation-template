import json
import logging
from pathlib import Path

from dowhy.causal_identifier import AutoIdentifier, BackdoorAdjustment, EstimandType
import networkx as nx

# Monkey patch for networkx d_separated, needed by some dowhy versions
nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
if not hasattr(nx, "d_separated"):
    nx.d_separated = nx.algorithms.d_separation.is_d_separator

logger = logging.getLogger("causal_pipeline")


def identify_adjustment_set(
    global_config: dict, exp_config: dict
) -> list[list[str]]:
    """
    Read a DAG specification from a JSON file and identify valid adjustment sets.

    Parameters
    ----------
    global_config : dict
        Global configuration dictionary containing keys such as 'treatment_node',
        'outcome_node', and 'conditional_nodes'.
    exp_config : dict
        Experiment configuration dictionary containing keys such as 'dag'
        (name of the DAG file without extension) and 'adjustment_set'
        (type of backdoor adjustment, e.g., 'BACKDOOR_EFFICIENT').

    Returns
    -------
    list of list of str
        A list of identified adjustment sets, where each set is a sorted list
        of node names.

    Raises
    ------
    ValueError
        If `adjustment_set` type is invalid or if no valid adjustment set is found.
    """
    conditional_nodes = global_config.get("conditional_nodes")
    treatment_name = global_config.get("treatment_node")
    outcome_name = global_config.get("outcome_node")
    adjustment_set_type = exp_config['adjustment_set']
    dag_path = Path("config/") / f'{exp_config["dag"]}.json'
    with open(dag_path, "rb") as f:
        edges_list = json.load(f)
    causal_graph = nx.DiGraph(edges_list)
    observed_nodes = list(causal_graph.nodes())
    try:
        bd_adj = getattr(BackdoorAdjustment, adjustment_set_type)
    except AttributeError:
        valid_options = [attr for attr in dir(BackdoorAdjustment) if not attr.startswith('_')]
        raise ValueError(f"'{adjustment_set_type}' is not a valid BackdoorAdjustment. Valid options are: {valid_options}")
        
    identifier = AutoIdentifier(
        estimand_type=EstimandType.NONPARAMETRIC_ATE,
        backdoor_adjustment=bd_adj,
    )
    
    identified_estimand = identifier.identify_effect(
        graph=causal_graph,
        action_nodes=treatment_name,
        outcome_nodes=outcome_name,
        observed_nodes=observed_nodes,
        conditional_node_names=conditional_nodes
    )
    
    if adjustment_set_type == "BACKDOOR_EXHAUSTIVE":
        # Get descendants of the treatment to exclude them from the required conditional nodes
        descendants = nx.descendants(causal_graph, treatment_name)
        filtered_conditional_nodes = [node for node in conditional_nodes if node not in descendants]
        
        all_sets = list(identified_estimand.backdoor_variables.values())
        valid_sets = []
        for s in all_sets:
            if s is not None and all(c in s for c in filtered_conditional_nodes):
                sorted_s = sorted(s)
                if sorted_s not in valid_sets:
                    valid_sets.append(sorted_s)
    else:
        valid_sets = [sorted(identified_estimand.get_backdoor_variables())]
    
    if len(valid_sets[0]) == 0:
        raise ValueError(f"No valid adjustment set found for experiment {exp_config['name']}")
    else:
        return valid_sets
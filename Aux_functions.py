import pandas as pd
import gudhi as gd
from typing import List

#############################################################################################################
#
# Auxiliary functions used in the Jupyter notebooks.
#
#############################################################################################################

def prepare_data(players_df: pd.DataFrame) -> List:
    players = players_df.values.tolist()
    
    for i in range(len(players)):
        del players[i][:9]

    return players

def sum_of_longest_distances(players_data: List, team_name: str = '') -> int:
    rc = gd.RipsComplex(points=players_data, max_edge_length=100)
    st = rc.create_simplex_tree(max_dimension=2)
    diag = st.persistence()
    
    h0 = [p for p in diag if p[0] == 0 and p[1][1] != float('inf')]
    
    sum_longest_distances = sum(p[1][1] for p in h0)

    if not team_name:
        return sum_longest_distances
    else:
        return (team_name, sum_longest_distances)

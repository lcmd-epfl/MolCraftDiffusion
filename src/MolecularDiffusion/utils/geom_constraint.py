import networkx as nx
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_networkx
from typing import List, Optional
from ase.data import covalent_radii

EDGE_THRESHOLD = 2
WEIGHT_FACTOR = 2

def find_close_points_torch(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    threshold: float = 1.0,
    push_distance: float = None
):
    """
    Find indices of rows in 'tgt' that lie within 'threshold' (in Angstroms)
    of any row in 'ref', and optionally translate those too-close points 
    away by 'push_distance' along the vector from the closest reference.

    Parameters
    ----------
    ref : torch.Tensor
        Reference structure of shape (N, 3).
    tgt : torch.Tensor
        Target structure of shape (Nt, 3).
    threshold : float, optional
        Distance threshold (default = 1.0 Å).
    push_distance : float or None, optional
        If not None, each target point within threshold will be translated
        by this distance away from its *closest* reference point.

    Returns
    -------
    close_indices : torch.Tensor
        1D tensor of indices in 'tgt' whose distance to any point in 'ref'
        is below 'threshold'.
    updated_tgt : torch.Tensor
        A new tensor (Nt, 3). If 'push_distance' is not None, the points 
        that were too close are translated away by 'push_distance'.
        Otherwise, it's the same as 'tgt'.
    """


    distances = torch.cdist(ref, tgt, p=2)  
    min_dist, closest_ref_idx = torch.min(distances, dim=0)  # shape => (Nt,), (Nt,)


    close_mask = min_dist < threshold
    close_indices = torch.where(close_mask)[0]

    if push_distance is not None:
        # Create a copy so we don't modify 'tgt' in-place
        updated_tgt = tgt.clone()

        # --- 4) Compute the centroid of the entire *original* target set ---
        centroid = tgt.mean(dim=0)  # shape => (3,)

        # --- 5) For each target point, if it is too close, push it toward the centroid ---
        # Direction from the point to the centroid
        direction = centroid - updated_tgt  # shape => (Nt, 3)

        # Normalize each direction vector
        norms = direction.norm(dim=1, keepdim=True) + 1e-12
        direction_normalized = direction / norms

        # Only apply shift to the too-close points
        # (Nt, 1) so we can broadcast multiply by the direction vectors
        shift_factor = close_mask.float().unsqueeze(1) * push_distance

        # Apply the translation
        updated_tgt = updated_tgt + direction_normalized * shift_factor
    else:
        updated_tgt = tgt

    return close_indices, updated_tgt

def find_close_points_torch_and_push(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    d_threshold_f: float = 1.0,
    d_threshold_c: float = 5.0,
    alpha_max: float = 10.0,
    steps: int = 40,
    centroid_mol: torch.Tensor = None,
    centroid_masked = None,
    b_weight=2
):
    """
    Identify target points that are too close to reference points and push them away.
    Args:
        ref (torch.Tensor): Reference points tensor of shape (N, D).
        tgt (torch.Tensor): Target points tensor of shape (Nt, D).
        d_threshold_f (float, optional): Minimum allowed distance between target nodes and frozen nodes. Default is 1.0.
        d_threshold_c (float, optional): Maximum allowed distance between target points and centroid point. Default is 5.0.
        alpha_max (float, optional): Maximum step size for binary search. Default is 10.0.
        steps (int, optional): Number of steps for binary search. Default is 40.
        centroid_mol (torch.Tensor, optional): Centroid of the molecule. Default is None.
        centroid_masked (torch.Tensor or list of torch.Tensor, optional): Masked centroid(s) for pushing direction. Default is None.
        b_weight (float, optional): Weight for the b vector in the combined push direction. Default is 2. 
            b vector is the vector from the target point to the centroid of the molecule.
    Returns:
        tuple: A tuple containing:
            - close_indices (torch.Tensor): Indices of target points that were too close to reference points.
            - updated_tgt (torch.Tensor): Updated target points tensor with pushed points.    

    """

    # Ensure both tensors are on the same device
    device = ref.device
    tgt = tgt.to(device)

    def min_dist_to_any_ref_and_closest_node(point, ref):
        dists = (ref - point).norm(dim=1)
        min_dist, closest_idx = dists.min(dim=0)
        return min_dist, ref[closest_idx]


    def push_point_with_weighted_combined_vector(
        point, 
        ref, 
        threshold, 
        centroid_mol, 
        centroid_masked,
        b_weight=1):
        """
        Push a point using the vector `a + b_weight * b`, where:
        - a = (centroid - point)
        - b = (point - closest_reference_node)
        """
        min_dist, _ = min_dist_to_any_ref_and_closest_node(point, ref)

        # If there are multiple centroids, find the closest one
        if len(centroid_masked) > 1:
            distances_to_centroids = [torch.cdist(point.unsqueeze(0), centroid_masked_o.unsqueeze(0), p=2).squeeze(0) 
                                      for centroid_masked_o in centroid_masked]
            
            closest_centroid_idx = torch.argmin(torch.stack(distances_to_centroids))
            centroid_masked_pt = centroid_masked[closest_centroid_idx]
        else:
   
            centroid_masked_pt = centroid_masked[0]    
        # If the point is already sufficiently far from all reference points, no push needed
        if min_dist >= threshold:
            return point

        # Compute the vector components
        # a = centroid - point  # Vector from point to centroid
        # b = point -closest_ref_node   # Vector from point to closest reference
        a = centroid_masked_pt - point
        b = point - centroid_mol
        
        # Determine if a and b are in the same direction
        # dot_product = torch.dot(a, b)
        # sign = -1 if dot_product > 0 else 1
        sign = 1
        combined_vector = a + sign* b_weight * b  # Emphasized combined direction
        norm_combined = combined_vector.norm()

        # Avoid division by zero in normalization
        if norm_combined < 1e-12:
            return point  # No meaningful push possible

        combined_unit_vector = combined_vector / norm_combined

        # Perform binary search along the `combined_unit_vector` direction to ensure threshold
        low, high = 0.0, alpha_max
        best_alpha = alpha_max

        for _ in range(steps):
            mid = 0.5 * (low + high)
            candidate = point + mid * combined_unit_vector
            candidate_min_dist, _ = min_dist_to_any_ref_and_closest_node(candidate, ref)

            if candidate_min_dist >= threshold:
                best_alpha = mid
                high = mid
            else:
                low = mid

        return point + best_alpha * combined_unit_vector
 
    # 1) Detect too-close target points to reference points
    distances = torch.cdist(ref, tgt, p=2)  # (N, Nt)
    min_dists, _ = torch.min(distances, dim=0)  # (Nt,)
    close_mask = min_dists < d_threshold_f
    close_indices = torch.where(close_mask)[0]
    
    if close_indices.size(0) > 0:
        print(f"Found {close_indices.size(0)} too-close points.")

    # 2) Push too-close points away from reference points
    updated_tgt = tgt.clone()
    if centroid_masked is None:
        centroid_masked = tgt.mean(dim=0)

    def dynamic_b_weight(updated_tgt, centroid):
        """
        Dynamically calculate the b_weight based on the variance of the distances
        of all target nodes from the centroid. Higher variance reduces b_weight
        to avoid over-spreading, and lower variance increases b_weight to spread
        nodes more evenly.
        """
        distances = (updated_tgt - centroid).norm(dim=1)
        variance = distances.var()
        # Dynamically adjust b_weight: lower variance increases weight
        b_weight = 1.0 / (variance + 1e-6)  # Add small epsilon to avoid division by zero
        return torch.clamp(b_weight, min=0.0, max=3.0)  # Limit b_weight to a reasonable range

    
    for i in close_indices:
        # b_weight = dynamic_b_weight(updated_tgt, centroid)  
        # print(b_weight)
        updated_tgt[i] = push_point_with_weighted_combined_vector(
            updated_tgt[i], ref, d_threshold_f, centroid_mol, centroid_masked, b_weight
        )

    # 3) Ensure all target nodes are within threshold of their closest centroid in centroid_masked
    # Convert centroid_masked to a tensor for distance calculations
    centroids_tensor = torch.stack(centroid_masked)  # (M, D)

    # Compute distances from each node to all centroids
    distances_to_centroids = torch.cdist(updated_tgt, centroids_tensor, p=2)  # (Nt, M)
    min_dists_centroid, closest_centroid_indices = torch.min(distances_to_centroids, dim=1)
    too_far_mask = min_dists_centroid > d_threshold_c
    too_far_indices = torch.where(too_far_mask)[0]

    if too_far_indices.size(0) > 0:
        print(f"Found {too_far_indices.size(0)} nodes too far from their closest centroid. Adjusting.")

    # Move each too-far node towards its closest centroid to be within threshold
    for idx in too_far_indices:
        node_pos = updated_tgt[idx]
        closest_centroid = centroids_tensor[closest_centroid_indices[idx]]
        D = min_dists_centroid[idx]

        # Compute the direction from node to centroid and move the node
        direction = closest_centroid - node_pos
        direction_normalized = direction / D  # Normalize by distance
        new_pos = node_pos + direction_normalized * (D - d_threshold_c)
        updated_tgt[idx] = new_pos

    return close_indices, updated_tgt


def find_connected_components(edge_index, num_nodes):
    """
    Find connected components in a graph represented by edge_index.

    Parameters
    ----------
    edge_index : torch.Tensor
        The edge indices of the graph, shape (2, num_edges).
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    components : list of torch.Tensor
        A list where each element is a tensor containing the node indices
        of a connected component.
    """
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    components = []

    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(current)
                neighbors = edge_index[1, edge_index[0] == current]
                stack.extend(neighbors.tolist())

    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(node, component)
            components.append(torch.tensor(component))

    return components

def find_subgraph_centroids(coords, distance_threshold=2.0):
    """
    Determine the centroids of all subgraphs (connected components) in a radius graph.

    Parameters
    ----------
    coords : torch.Tensor
        Cartesian coordinates of nodes, shape (N, 3).
    distance_threshold : float
        Maximum distance for an edge in the graph.

    Returns
    -------
    centroids : list of torch.Tensor
        A list of centroid coordinates for each connected component.
    """
    # Step 1: Create the radius graph (edge indices)
    edge_index = radius_graph(coords, r=distance_threshold, loop=False)

    num_nodes = coords.size(0)  
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    components = [list(c) for c in nx.connected_components(G)]
    
    
    # Step 3: Compute centroid for each subgraph
    centroids = []
    for component in components:
        # Collect the coordinates of all nodes in this component
        subgraph_coords = coords[component]
        # Compute the centroid as the mean position
        centroid = subgraph_coords.mean(dim=0)
        centroids.append(centroid)

    return centroids


def find_close_points_torch_and_push_op_v0(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor = None,
    d_threshold_f: float = 1.0,
    d_threshold_c: float = 1.0,
    alpha_max: float = 10.0,
    steps: int = 40,
):
    """
    For out-painting, we need to push the nodes away from the reference nodes
    Identify target points (tgt) that are too close to any reference points (ref).

    The reference tensor `ref` has some nodes potentially serving as connectors (given by
    `connector_indices`) and the rest are implicitly treated as frozen.

    If `connector_indices` is specified and non-empty, for each violating node in `tgt`:
      1. Identify the closest connector node in `ref`.
      2. Push the violating `tgt` node away from that connector node so that
         it is at least `d_threshold_f` away from all *frozen* nodes (i.e., those not in `connector_indices`)
         and at least `d_threshold_c` away from the chosen connector node.

    If `connector_indices` is None or empty, revert to the old logic:
      - We find all `tgt` points too close to any node in `ref` (threshold = `d_threshold_f`).
      - We push them away from the closest reference node, ensuring they end up
        at least `d_threshold_f` away from all of `ref`.

    Args:
        ref (torch.Tensor): Reference points of shape (Nr, D).
        tgt (torch.Tensor): Target points of shape (Nt, D).
        connector_indices (torch.Tensor, optional): Indices of connector nodes in `ref`.
        d_threshold_f (float, optional): Required min distance from frozen nodes.
        d_threshold_c (float, optional): Required min distance from the chosen connector node.
        alpha_max (float, optional): Max step size for binary search. Default 10.
        steps (int, optional): Number of binary search steps. Default 40.

    Returns:
        tuple:
            - violating_indices (torch.Tensor): Indices of `tgt` points that were too close.
            - updated_tgt (torch.Tensor): Updated target points tensor with pushed nodes.
    """

    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()

    # 1) If connector_indices is None or empty, we do the fallback: all references are obstacles.
    if connector_indices is None or connector_indices.numel() == 0:
        dist_all = torch.cdist(tgt, ref, p=2)  # (Nt, Nr)
        min_dist_all, idx_closest = dist_all.min(dim=1)
        violate_mask = min_dist_all < d_threshold_f
        violating_indices = torch.where(violate_mask)[0]

        def constraints_satisfied_fallback(candidate_pt: torch.Tensor) -> bool:
            dists = torch.norm(candidate_pt - ref, dim=1)
            return dists.min() >= d_threshold_f

        for idx in violating_indices:
            point = updated_tgt[idx]
            closest_ref_pt = ref[idx_closest[idx]]

            direction = point - closest_ref_pt
            dist_dir = direction.norm()
            if dist_dir < 1e-12:
                direction = torch.randn_like(point)
                dist_dir = direction.norm()
            direction_unit = direction / dist_dir

            low, high = 0.0, alpha_max
            best_alpha = alpha_max

            for _ in range(steps):
                mid = 0.5 * (low + high)
                candidate_pt = point + mid * direction_unit
                if constraints_satisfied_fallback(candidate_pt):
                    best_alpha = mid
                    high = mid
                else:
                    low = mid

            updated_tgt[idx] = point + best_alpha * direction_unit

        return violating_indices, updated_tgt

    # 2) Otherwise, infer the frozen indices as the complement of connector_indices.
    
    all_indices = torch.arange(ref.size(0), device=device)
    frozen_mask = torch.ones(ref.size(0), dtype=torch.bool, device=device)
    frozen_mask[connector_indices] = False
    
    frozen_indices = all_indices[frozen_mask]

    ref_frozen = ref[frozen_indices]
    ref_conn = ref[connector_indices]

    # (a) Distances from tgt to frozen & connector nodes
    if ref_frozen.numel() > 0:
        dist_frozen = torch.cdist(tgt, ref_frozen, p=2)
        min_dist_frozen, _ = dist_frozen.min(dim=1)
    else:
        min_dist_frozen = torch.full((tgt.size(0),), float('inf'), device=device)

    dist_conn = torch.cdist(tgt, ref_conn, p=2)
    min_dist_conn, idx_conn = dist_conn.min(dim=1)

    # (b) A node is violating if it is < d_threshold_f from any frozen node
    #     or < d_threshold_c2 from any connector node.
    violate_mask = (min_dist_frozen < d_threshold_f) | (min_dist_conn < d_threshold_c)
    violating_indices = torch.where(violate_mask)[0]

    def constraints_satisfied(candidate, chosen_conn):
        # Must be >= d_threshold_f away from all frozen nodes
        if ref_frozen.numel() > 0:
            dist_fz = torch.norm(candidate - ref_frozen, dim=1)
            if dist_fz.min() < d_threshold_f:
                return False
        # Must be >= d_threshold_c2 away from the chosen connector node
        if torch.norm(candidate - chosen_conn) < d_threshold_c:
            return False
        return True

    # (c) For each violating node, push it away from its closest connector
    for idx in violating_indices:
        point = updated_tgt[idx]
        c_idx = idx_conn[idx]
        chosen_conn_pt = ref_conn[c_idx]

        direction = point - chosen_conn_pt
        dist_dir = direction.norm()
        if dist_dir < 1e-12:
            direction = torch.randn_like(point)
            dist_dir = direction.norm()
        direction_unit = direction / dist_dir

        # Binary search
        low, high = 0.0, alpha_max
        best_alpha = alpha_max
        for _ in range(steps):
            mid = 0.5 * (low + high)
            candidate_pt = point + mid * direction_unit
            if constraints_satisfied(candidate_pt, chosen_conn_pt):
                best_alpha = mid
                high = mid
            else:
                low = mid

        updated_tgt[idx] = point + best_alpha * direction_unit


    return violating_indices, updated_tgt


def find_close_points_torch_and_push_op(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor,
    d_threshold_c: float = 1.0,
    d_fixed: Optional[float] = None,
    d_threshold_f: float = 1.8,
    w_b: float = 0.5,
    d_max: float = 10.0,
    steps: int = 40,
    tol: float = 1e-6,
) -> torch.Tensor:
    r"""
    For each target node, first assign it to the closest connector node in ref (as defined
    by the provided connector_indices). Then, for each group (i.e. all target nodes that share
    the same connector), do the following:

      1. Compute the group centroid c_group from the target nodes in the group.
      2. Compute two vectors:
           a = (connector node of the group) - (c_group)
           b = c_group - (global centroid of ref)
      3. Define the push (or pull) vector as:
           v = a + w_b * b
         and normalize it.
      4. Update every target node p in the group by translating it along v:
           p_new = p + d * v
         where the translation distance d is either:
            - Fixed (if d_fixed is provided) or
            - Determined automatically (via binary search) so that the point in the group
              that is closest to c_group after translation is exactly d_threshold_c away.
    
    Parameters:
      ref: Tensor of shape (N_ref, D).
      tgt: Tensor of shape (N_tgt, D).
      connector_indices: 1D Tensor containing indices of ref that are “connector” nodes.
      d_threshold_c: In auto mode, the desired minimum distance from the group centroid (c_group)
                     after translation.
      d_fixed: Optional fixed translation distance. If provided (not None), every group uses
               this distance along the computed push vector.
      d_threshold_f: Minimum allowed distance between target nodes and frozen nodes.
      w_b: Weight applied to vector b (b = c_group - global centroid of ref) when forming the push vector.
      d_max: Maximum (absolute) allowed translation distance (used in auto mode).
      steps: Number of binary search iterations (auto mode).
      tol: Tolerance for binary search convergence (auto mode).
      
    Returns:
      updated_tgt: The target nodes after translation.
    """
    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()

    # Global centroid of the reference nodes.
    c_ref = ref.mean(dim=0)

    # Extract connector nodes from ref.
    connectors = ref[connector_indices]

    # For each target, find the index (within connectors) of the closest connector.
    dist_conn = torch.cdist(tgt, connectors, p=2)
    _, idx_conn = dist_conn.min(dim=1)  # For each target: index in connectors

    # Group target indices by their assigned connector.
    groups = {}
    for i, c_idx in enumerate(idx_conn.tolist()):
        groups.setdefault(c_idx, []).append(i)

    # Process each group.
    for conn_grp, tgt_inds in groups.items():
        # Get the connector node for this group.
        connector_node = connectors[conn_grp]

        # Compute the group centroid from the target nodes.
        pts = updated_tgt[tgt_inds]  # shape (N_grp, D)
        c_group = pts.mean(dim=0)

        # Compute the two vectors.
        a = connector_node - c_group
        b = c_group - c_ref

        # Form the push vector.
        push_vec = a + w_b * b
        norm_push = torch.norm(push_vec)
        if norm_push < 1e-12:
            # If the push vector is nearly zero, choose an arbitrary direction.
            push_vec = torch.randn_like(push_vec)
            norm_push = torch.norm(push_vec)
        v = push_vec / norm_push

        # Determine translation distance.
        if d_fixed is not None:
            d_grp = d_fixed
        else:
            # In auto mode, we want the worst–off node (relative to c_group) after translation
            # to lie exactly d_threshold_c away from c_group.
            # For each point, decompose its offset from c_group along v and perpendicular to v.
            offs = pts - c_group  # shape (N_grp, D)
            A = (offs * v).sum(dim=1)  # projection lengths (N_grp,)
            B = (offs ** 2).sum(dim=1)
            C = torch.sqrt(torch.clamp(B - A**2, min=0.0))

            # Define F(d) = minimum over group of sqrt((A + d)^2 + C^2)
            def F(d_val: float) -> float:
                new_dists = torch.sqrt((A + d_val)**2 + C**2)
                return new_dists.min().item()

            f0 = F(0.0)
            # If f0 is already within tolerance, no translation is needed.
            if abs(f0 - d_threshold_c) < tol:
                d_grp = 0.0
            elif f0 < d_threshold_c:
                # Group is too close; push outward (d > 0).
                low = 0.0
                high = d_max
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    if F(mid) >= d_threshold_c:
                        high = mid
                    else:
                        low = mid
                d_grp = high
            else:
                # f0 > d_threshold_c: group is too far; pull inward (d < 0).
                # Ensure that the parallel component remains nonnegative.
                d_low_bound = -float(A.min().item())
                low = d_low_bound
                high = 0.0
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    if F(mid) < d_threshold_c:
                        low = mid
                    else:
                        high = mid
                d_grp = high

            d_grp = max(min(d_grp, d_max), -d_max)

        # Update all nodes in this group.
        updated_tgt[tgt_inds] = pts + d_grp * v

        dists_to_frozen = torch.cdist(pts, ref[~connector_indices], p=2)
        sorted_indices = torch.argsort(dists_to_frozen, dim=0)
        # print(sorted_indices)
        for idx in sorted_indices:
            point = updated_tgt[idx]
            dist_to_frozen = torch.norm(point - ref[~connector_indices], dim=1).min()
            # print(dist_to_frozen)
            if dist_to_frozen < d_threshold_f:
                print("Violating distance constraint", dist_to_frozen, idx)
                a = c_group - point
                b = point - c_ref
                move_vec = (a + b)
                move_vec = move_vec / move_vec.norm()
                move_dist = d_threshold_f - dist_to_frozen
                updated_tgt[idx] = point + move_dist * move_vec
            else:
                break

    return updated_tgt

#%% For OP
def find_close_points_torch_and_push_op2(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor = None,
    d_threshold_f: float = 1.0,
    d_threshold_c: float = 1.0,
    d_fixed_move: Optional[float] = None,
    alpha_max: float = 10.0,
    steps: int = 40,
    w_b: float = 0.5,
    search_method: str = "binary",
    all_frozen: bool = False,
    z_ref: torch.Tensor = None,
    z_tgt: torch.Tensor = None,
    scale_factor: float = 1.1,
    debug: bool = False ,
):  
    """
    Identifies points in the target tensor that are too close to the reference tensor and adjusts their positions.
    Also detects outliers based on subgraph connectivity using a distance threshold.
    
    Order of operations:
      1. Detect and move outliers to merge with their closest target subgraph, ensuring a minimum distance of d_threshold_c.
      2. Recalculate distances and detect violating nodes (relative to frozen points) and adjust them.
      3. Additionally, detect target nodes that are closer to any frozen nodes than the minimal distance of their 
         closest connector node to any frozen node.
    
    Args:
        ref (torch.Tensor): Reference tensor of shape (N, D).
        tgt (torch.Tensor): Target tensor of shape (M, D).
        connector_indices (torch.Tensor, optional): Indices of connector points in the reference tensor.
        d_threshold_f (float, optional): Distance threshold for frozen points.
        d_threshold_c (float, optional): Distance threshold for connector points (and outlier merging).
        d_fixed_move (float, optional): Fixed distance for moving points.
        alpha_max (float, optional): Maximum step size for adjustments.
        steps (int, optional): Number of steps for binary/adaptive search.
        w_b (float, optional): Weight for the secondary direction in adjustment.
        search_method (str, optional): Search method for finding the best adjustment ("binary", "adaptive", or "log").
        all_frozen (bool, optional): Whether all points in the reference tensor are frozen. Default is False.
        
        Toggle this if you want few bonds with the connectors.
        This is useful when there is too few frozen nodes to push the target nodes away to form valid subgraphs to the connector nodes,
        resulting in these target nodes developing in the middle of rerfernce structures, forming too many bonds with the connector nodes.
        
        
        z_ref (torch.Tensor, optional): Atomic numbers of the reference tensor. Default is None.
        z_tgt (torch.Tensor, optional): Atomic numbers of the target tensor. Default is None.
        scale_factor (float, optional): Scale factor for the covalent radii. Default is 1.1.
        debug (bool, optional): Whether to print debug information. Default is False.
        
        If these are provided, the function will consider covanlent radii of the atoms to adjust the distances
        instead of d_threshold_f.
        
        For example, see 60.xyz
        NOTE: When this is toggled, recommend to decrease the d_threshold_c to around 1.4
        and set t_critical to 0.8
        
        debug (bool, optional): Whether to print debug information. Default is False.
        
    Returns:
        torch.Tensor: Indices of the points that were adjusted.
        torch.Tensor: Updated target tensor with adjusted points.
    """
    device = ref.device
    tgt = tgt.to(device)
    COV_R = torch.tensor(covalent_radii, dtype=torch.float32, device=device)
    updated_tgt = tgt.clone()
    
    if (z_ref is not None and z_tgt is None) or (z_ref is None and z_tgt is not None):
        raise ValueError("Both z_ref and z_tgt must be specified if either is provided.")

    
    if connector_indices is None or connector_indices.numel() == 0:
        return torch.tensor([]), updated_tgt
    
    # Set up frozen and connector points.
    all_indices = torch.arange(ref.size(0), device=device)
    frozen_mask = torch.ones(ref.size(0), dtype=torch.bool, device=device)
    frozen_mask[connector_indices] = False
    if len(connector_indices) == ref.size(0):
        frozen_mask[connector_indices] = True
    frozen_indices = all_indices[frozen_mask]
    if all_frozen:
        ref_frozen = ref
    else:
        ref_frozen = ref[frozen_indices]
    ref_conn = ref[connector_indices]
    centroid = ref.mean(dim=0)  # Centroid of ref
    
    # Compute KDTree for connector points.
    tree_conn = KDTree(ref_conn.cpu().numpy())
    min_dist_conn, idx_conn = tree_conn.query(tgt.cpu().numpy())
    min_dist_conn = torch.tensor(min_dist_conn, device=device)
    idx_conn = torch.tensor(idx_conn, device=device)

    adjusted_indices = []  # Record indices that have been moved.


    # --- Step 2: Detect and Handle Violating Nodes ---
    # Recompute distances using the updated target positions.
    dist_frozen = torch.cdist(updated_tgt, ref_frozen, p=2) if ref_frozen.numel() > 0 else torch.full((tgt.size(0), 1), float('inf'), device=device)
    dist_centroid = torch.norm(updated_tgt - centroid, dim=1)
    dist_connector_to_centroid = torch.norm(ref_conn[idx_conn] - centroid, dim=1)
    dist_to_connector = torch.norm(updated_tgt - ref_conn[idx_conn], dim=1)

    if z_ref is not None and z_tgt is not None:
        r1 = COV_R[z_tgt] / scale_factor  # (M,)
        r2 = COV_R[z_ref] / scale_factor  # (N,)
        min_bond_length = r1[:, None] + r2[None, :]  # (M+N, N)
        min_bond_length = min_bond_length[:-z_ref.size(0)]
        violating_indices = torch.where(
            (dist_frozen.min(dim=1).values < min_bond_length.min(dim=1).values) & 
            ((dist_centroid <= dist_connector_to_centroid) | (dist_to_connector >= d_threshold_c))
        )[0].tolist()
    else:
    
        violating_indices = torch.where(
            (dist_frozen.min(dim=1).values < d_threshold_f) & 
            ((dist_centroid <= dist_connector_to_centroid) | (dist_to_connector >= d_threshold_c))
        )[0].tolist()
    
    # --- Additional Detection: Find target nodes that are closer to frozen nodes than
    # the minimal distance of their closest connector node to any frozen node.
    # For each target, compute its minimum distance to any frozen node.
    d_frozen_target = dist_frozen.min(dim=1).values  # Shape: (M,)
    # For each connector, compute its minimum distance to any frozen node.
    if ref_frozen.numel() > 0:
        dist_conn_frozen = torch.cdist(ref_conn, ref_frozen, p=2)  # shape: (number_of_connectors, num_frozen)
        d_min_conn = dist_conn_frozen.min(dim=1).values  # minimal distance for each connector
        # For each target, get the minimal frozen distance of its closest connector.
        d_threshold_target = d_min_conn[idx_conn]
        # Find targets where the target's minimum distance to frozen is less than that threshold.
        additional_violation_indices = torch.where(d_frozen_target < d_threshold_target)[0].tolist()
    else:
        additional_violation_indices = []
    
    # Combine both violation lists (avoiding duplicates)
    all_violation_indices = list(set(violating_indices +  additional_violation_indices))
    
    # Move each violating node to satisfy the frozen distance constraint.
    for idx in all_violation_indices:
        point = updated_tgt[idx]
        c_idx = idx_conn[idx]
        c2 = ref_conn[c_idx]
        
        a = c2 - point
        b = c2 - centroid
        direction = a + w_b * b
        dist_dir = direction.norm()
        if dist_dir < 1e-12:
            direction = torch.randn_like(point)
            dist_dir = direction.norm()
        direction_unit = direction / dist_dir
        
        if d_fixed_move is not None:
            best_alpha = d_fixed_move
        else:
            low, high = 0.0, alpha_max
            best_alpha = alpha_max
            if search_method == "adaptive":
                alpha = best_alpha
                for _ in range(steps):
                    candidate_pt = point + alpha * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = alpha
                        alpha *= 0.5
                    else:
                        alpha *= 0.7
            elif search_method == "binary":
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    candidate_pt = point + mid * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = mid
                        high = mid
                    else:
                        low = mid
            elif search_method == "log":
                for _ in range(steps):
                    alpha = alpha_max / (1 + torch.exp(torch.tensor(-0.1 * (_ - steps / 2))))
                    candidate_pt = point + alpha * direction_unit
                    if torch.norm(candidate_pt - ref_frozen, dim=1).min() >= d_threshold_f:
                        best_alpha = alpha  
        if debug:
            print("Mover 2: Moving point", idx, "by", best_alpha, "to satisfy frozen distance constraint.")
        updated_tgt[idx] = point + best_alpha * direction_unit
        adjusted_indices.append(idx)

 
    return torch.tensor(adjusted_indices, device=device), updated_tgt


def ensure_intact(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor = None,
    d_threshold: float = 1.0,
    d_threshold_e: float = 1.8, 
    d_fixed_move: Optional[float] = None,
    steps: int = 40,
    debug: bool = False,
):  
    """
    Identifies subgraphs of target nodes, detects outlier subgraphs based on their distance to connector nodes,
    and moves them toward the closest non-outlier subgraph.
    
    Args:
        ref (torch.Tensor): Reference tensor of shape (N, D).
        tgt (torch.Tensor): Target tensor of shape (M, D).
        connector_indices (torch.Tensor, optional): Indices of connector points in the reference tensor.
        d_threshold (float, optional): Distance threshold to determine outlier subgraphs.
        d_threshold_e (float, optional): Distance threshold for edge formation in the graph.
        d_fixed_move (float, optional): Fixed distance for moving points.   
        steps (int, optional): Number of steps for binary search while moving outlier subgraphs.
        debug (bool, optional): Whether to print debug information.
        
    Returns:
        torch.Tensor: Indices of the points that were adjusted.
        torch.Tensor: Updated target tensor with adjusted points.
    """
    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()
    
    if connector_indices is None or connector_indices.numel() == 0:
        return torch.tensor([]), updated_tgt
    
    # Compute KDTree for connector points.
    ref_conn = ref[connector_indices]
    tree_conn = KDTree(ref_conn.cpu().numpy())
    
    # Compute connected components on the target using KDTree.
    tree_tgt = KDTree(tgt.cpu().numpy())
    adjacency_matrix = tree_tgt.sparse_distance_matrix(tree_tgt, max_distance=d_threshold_e, output_type='coo_matrix')
    sparse_graph = csr_matrix(
        (adjacency_matrix.data, (adjacency_matrix.row, adjacency_matrix.col)),
        shape=(tgt.size(0), tgt.size(0))
    )
    _, labels = connected_components(sparse_graph)
    
    # Identify subgraphs
    unique_labels = torch.tensor(labels, device=device).unique()
    subgraphs = {label.item(): (labels == label.item()).nonzero()[0] for label in unique_labels}
    
    outlier_subgraphs = []
    for label, indices in subgraphs.items():
        subgraph_points = tgt[indices]
        min_dist, _ = tree_conn.query(subgraph_points.cpu().numpy())
        min_dist = torch.tensor(min_dist, device=device).min()
        
        if min_dist > d_threshold:
            outlier_subgraphs.append(indices)
    
    # Move outlier subgraphs toward the closest non-outlier subgraph
    non_outlier_indices = torch.tensor([i for i in range(tgt.size(0)) if not any(i in o for o in outlier_subgraphs)], device=device)
    
    adjusted_indices = []
    if non_outlier_indices.numel() > 0 and len(outlier_subgraphs) > 0:
        for indices in outlier_subgraphs:
            subgraph_points = updated_tgt[indices]
            
            dists = torch.cdist(subgraph_points, updated_tgt[non_outlier_indices])
            closest_idx = dists.argmin(dim=1)
            closest_points = updated_tgt[non_outlier_indices][closest_idx]
            
            direction = closest_points.mean(dim=0) - subgraph_points.mean(dim=0)
            dist_dir = direction.norm()
            if dist_dir < 1e-12:
                direction = torch.randn_like(direction)
                dist_dir = direction.norm()
            direction_unit = direction / dist_dir
            
            if d_fixed_move:
                best_alpha = d_fixed_move
            else:
            # Binary search to find minimal movement
                low, high = 0.0, dist_dir
                best_alpha = high
                for _ in range(steps):
                    mid = 0.5 * (low + high)
                    candidate_pts = subgraph_points + mid * direction_unit
                    candidate_dists = torch.norm(candidate_pts - closest_points.mean(dim=0), dim=1)
                    
                    if candidate_dists.max() > d_threshold:
                        low = mid
                    else:
                        best_alpha = mid
                        high = mid
                
                updated_tgt[indices] += best_alpha * direction_unit
                adjusted_indices.extend(indices.tolist())
            
            if debug:
                print(f"Moving outlier subgraph {indices.tolist()} toward closest non-outlier subgraph by {best_alpha}")
    
    return torch.tensor(adjusted_indices, device=device), updated_tgt


def enforce_min_nodes_per_connector(
    ref: torch.Tensor,
    tgt: torch.Tensor,
    connector_indices: torch.Tensor,
    N: List[int],
    d_threshold_c: List[float],
    debug: bool = False,
) -> torch.Tensor:
    device = ref.device
    tgt = tgt.to(device)
    updated_tgt = tgt.clone()
    moved_nodes = set()  # track nodes already moved as part of a subgraph

    # Extract connector nodes from ref
    connectors = ref[connector_indices]
    num_connectors = connectors.shape[0]
    
    if len(N) == 1:
        N = N * num_connectors
    if len(d_threshold_c) == 1:
        d_threshold_c = d_threshold_c * num_connectors

    if len(N) != num_connectors or len(d_threshold_c) != num_connectors:
        raise ValueError("Number of N and d_threshold_c must match number of connector nodes.")

    # Assign each target to its closest connector
    ref_conn = ref[connector_indices]
    tree_conn = KDTree(ref_conn.cpu().numpy())
    min_dist_conn, idx_conn = tree_conn.query(tgt.cpu().numpy())
    min_dist_conn = torch.tensor(min_dist_conn, device=device)
    idx_conn = torch.tensor(idx_conn, device=device)
    

    # Group target indices by their assigned connector
    groups = {i: [] for i in range(num_connectors)}
    for i, c_idx in enumerate(idx_conn.tolist()):
        groups[c_idx].append(i)

    # Process each group
    for i, tgt_inds in groups.items():
        connector_node = connectors[i]
        # Set a maximum number of iterations to avoid infinite loops.
        iteration = 0
        max_iter = len(tgt_inds)  # adjust as needed
        while iteration < max_iter:
            iteration += 1
            pts = updated_tgt[tgt_inds]  # positions for nodes in this group
            dists = torch.norm(pts - connector_node, dim=1)
            close_mask = dists <= d_threshold_c[i]
            num_close = close_mask.sum().item()
            # if debug:
            #     print(f"Group {i}: {num_close} nodes within threshold") # for debugging
            #     print(f"Group {i}: Close distances: {dists[close_mask].tolist()}")
            if num_close >= N[i]:
                break

            sorted_indices = torch.argsort(dists)
            candidate_found = False

            for j in sorted_indices:
                if close_mask[j]:
                    continue  # skip nodes already within threshold
                global_idx = tgt_inds[j]
                # if debug:
                #     print(f"Group {i}: Trying to move node {global_idx}")  # for debugging
                if global_idx in moved_nodes:
                    continue

                candidate_found = True
                chosen_global = global_idx
                chosen_pos = updated_tgt[chosen_global]

                # Compute unit direction from candidate toward the connector
                a_vec = connector_node - chosen_pos

                # # Find the target node in the group that is closest to the connector.
                # dists_group = torch.norm(updated_tgt[tgt_inds] - connector_node, dim=1)
                # min_idx = torch.argmin(dists_group)
                # closest_global = tgt_inds[min_idx]
                # b = (chosen_pos) - (position of the closest node)
                close_indices = [idx for idx in tgt_inds if torch.norm(updated_tgt[idx] - connector_node) <= d_threshold_c[i]]
                if close_indices:
                    close_positions = torch.stack([updated_tgt[idx] for idx in close_indices], dim=0)
                    centroid = close_positions.mean(dim=0)
                    b_vec = chosen_pos - centroid
                else:
                    b_vec = torch.zeros_like(a_vec)
                    
                move_vector = a_vec + WEIGHT_FACTOR * b_vec

                # Build connectivity graph using radius_graph from PyTorch Geometric.
                group_graph = nx.Graph()
                for idx in tgt_inds:
                    group_graph.add_node(idx)
                pts_group = updated_tgt[tgt_inds]
                edge_index = radius_graph(pts_group, r=EDGE_THRESHOLD, loop=False)
                # Convert local indices (relative to pts_group) to global indices.
                global_indices = torch.tensor(tgt_inds, device=device)
                global_edge_index = edge_index.clone()
                global_edge_index[0] = global_indices[edge_index[0]]
                global_edge_index[1] = global_indices[edge_index[1]]
                for m, n in global_edge_index.t().tolist():
                    group_graph.add_edge(m, n)
                
                # Extract the connected subgraph (component) containing the candidate.
                sub_nodes = list(nx.node_connected_component(group_graph, chosen_global))
                # Filter out nodes that are already within threshold.
                sub_nodes_to_move = [n for n in sub_nodes if torch.norm(updated_tgt[n] - connector_node) > d_threshold_c[i]]
                # If no nodes need to be moved, skip this candidate.
                if not sub_nodes_to_move:
                    continue

                # Determine maximum allowed translation so that none overshoot the connector.
                safe_factors = []
                mv_dot = torch.dot(move_vector, move_vector)
                if mv_dot == 0:
                    continue  # Cannot move if move_vector is zero.
                for n_idx in sub_nodes_to_move:
                    pos_n = updated_tgt[n_idx]
                    # Maximum scalar such that pos_n + s*move_vector does not overshoot.
                    s_max_n = torch.dot(connector_node - pos_n, move_vector) / mv_dot
                    safe_factors.append(s_max_n)
                safe_max = min(safe_factors)

                # Compute the distance the candidate must move to reach the threshold.
                d_chosen = torch.norm(connector_node - chosen_pos)
                desired_alpha = d_chosen - d_threshold_c[i]
                if desired_alpha < 0:
                    continue  # candidate is already within threshold (should not occur)
                desired_alpha = min(desired_alpha, safe_max)

                # Use binary search to refine the minimal translation needed.
                low = 0.0
                high = desired_alpha
                tol = 1e-5
                while high - low > tol:
                    mid = (low + high) / 2.0
                    new_distance = torch.norm((chosen_pos + mid * move_vector) - connector_node)
                    if new_distance > d_threshold_c[i]:
                        low = mid
                    else:
                        high = mid
                alpha_move = high

                # Move every node in the filtered subgraph.
                for n_idx in sub_nodes_to_move:
                    if debug:
                        print(f"Mover 3: Group {i}: Moving node {n_idx} by {alpha_move}")
                    updated_tgt[n_idx] = updated_tgt[n_idx] + alpha_move * move_vector
                    moved_nodes.add(n_idx)
                break  # one subgraph move; re-evaluate the group

            if not candidate_found:
                break

    return updated_tgt


#%% For OP-FT

def align_target_with_reference(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Aligns the target coordinates to the reference coordinates by applying
    a pure translation. The first Nr rows of 'target' correspond to
    the same reference nodes in 'reference', but in a different coordinate
    system.

    Parameters:
    -----------
    reference : torch.Tensor
        Shape (Nr, 3). Reference coordinates of Nr nodes.
    target : torch.Tensor
        Shape (N, 3). Coordinates of N nodes, with the first Nr rows
        corresponding to the reference nodes, in a different coordinate system.

    Returns:
    --------
    torch.Tensor
        A translated version of 'target' whose first Nr rows align with 'reference'.
    """

    # Number of reference nodes
    Nr = reference.shape[0]

    # Compute the centroid of the reference nodes
    ref_centroid = reference.mean(dim=0)  # shape (3,)

    # Compute the centroid of the corresponding nodes in the target
    target_ref_centroid = target[:Nr].mean(dim=0)  # shape (3,)

    # The translation needed to align the target's reference nodes with reference
    translation = ref_centroid - target_ref_centroid

    # Translate all target nodes
    aligned_target = target + translation

    return aligned_target


def translate_to_origine(coords, node_mask):
    centroid = coords.mean(dim=1, keepdim=True)  
    translation_vector = -centroid
    translated_coords = coords + translation_vector * node_mask
    return translated_coords




#!/usr/bin/env python3
"""Generate a SUS2 LK-style untrained .mtp basis file.

This generator keeps the important mixed-mu detail explicit:
when two moments with different mu channels are lifted to a higher-rank object,
the intermediate tensor is not assumed to be fully symmetric across the two
factor blocks. Instead, the script tracks a block-symmetric tensor basis:

- each primitive moment contributes one symmetric block;
- contractions only remove slots between existing open blocks and the new leaf;
- symmetry is preserved inside each block, but different blocks remain ordered.

This is a broader and safer construction than the original hand-written
`gen-LK.py`, and it enforces one hard pruning rule during generation:
no intermediate coupling rank may exceed the requested `L`.

The default truncation is `max_degree = L + 1`, which corresponds to
`max_body_order = L + 2`. Override it explicitly if you want a different cutoff.
"""

import argparse
import itertools
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

AxisCounts = Tuple[int, int, int]
LeafKey = Tuple[int, int]  # (mu, rank)
FINGERPRINT_MODULUS = 2147483647
FINGERPRINT_SEEDS = (11, 29, 47, 83, 131, 197, 251, 337)


@lru_cache(maxsize=None)
def exponent_tuples(rank):
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    out = []
    for ax in range(rank, -1, -1):
        for ay in range(rank - ax, -1, -1):
            out.append((ax, ay, rank - ax - ay))
    return tuple(out)


@lru_cache(maxsize=None)
def exponent_to_index(rank):
    return {exp: idx for idx, exp in enumerate(exponent_tuples(rank))}


@lru_cache(maxsize=None)
def rank_gram_block_basis(rank):
    return tuple(
        (exp_idx, multinomial(exp))
        for exp_idx, exp in enumerate(exponent_tuples(rank))
    )


def add_counts(lhs, rhs):
    return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def multinomial(counts):
    total = counts[0] + counts[1] + counts[2]
    out = math.factorial(total)
    out //= math.factorial(counts[0])
    out //= math.factorial(counts[1])
    out //= math.factorial(counts[2])
    return out


class Layout:
    def __init__(self, open_counts, basis_sizes, strides, index_tuples):
        self.open_counts = open_counts
        self.basis_sizes = basis_sizes
        self.strides = strides
        self.index_tuples = index_tuples

    @property
    def total_size(self):
        if not self.basis_sizes:
            return 1
        out = 1
        for size in self.basis_sizes:
            out *= size
        return out

    def flatten(self, index_tuple):
        return sum(idx * stride for idx, stride in zip(index_tuple, self.strides))


@lru_cache(maxsize=None)
def block_layout(open_counts):
    basis_sizes = tuple(len(exponent_tuples(rank)) for rank in open_counts)
    if not basis_sizes:
        return Layout(open_counts, (), (), ((),))

    strides = [1] * len(basis_sizes)
    for i in range(len(basis_sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * basis_sizes[i + 1]

    index_tuples = tuple(itertools.product(*(range(size) for size in basis_sizes)))
    return Layout(open_counts, basis_sizes, tuple(strides), index_tuples)


@lru_cache(maxsize=None)
def contraction_patterns(open_counts, leaf_rank, max_rank):
    out = []

    def rec(pos, contracted, prefix):
        if pos == len(open_counts):
            new_rank = sum(open_counts) + leaf_rank - 2 * contracted
            if new_rank <= max_rank:
                out.append(tuple(prefix))
            return

        max_here = min(open_counts[pos], leaf_rank - contracted)
        for val in range(max_here + 1):
            prefix.append(val)
            rec(pos + 1, contracted + val, prefix)
            prefix.pop()

    rec(0, 0, [])
    return tuple(out)


def flatten_edges(edges):
    flat = []
    for i, row in enumerate(edges):
        for j in range(i + 1, len(row)):
            flat.append(row[j])
    return tuple(flat)


def topologically_renumber(basic, times, mapping, node_count):
    basic_count = len(basic)
    preds = defaultdict(list)
    succs = defaultdict(list)
    indeg = {}

    for node in range(basic_count, node_count):
        indeg[node] = 0

    for src0, src1, coeff, dst in times:
        preds[dst].append((src0, src1, coeff))
        for src in (src0, src1):
            if src >= basic_count:
                succs[src].append(dst)
                indeg[dst] += 1

    ready = sorted([node for node in range(basic_count, node_count) if indeg[node] == 0])
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for dst in succs.get(node, []):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort()

    if len(order) != node_count - basic_count:
        raise RuntimeError("internal graph is not a DAG")

    remap = {}
    for idx in range(basic_count):
        remap[idx] = idx
    next_id = basic_count
    for node in order:
        remap[node] = next_id
        next_id += 1

    new_times = []
    for node in order:
        for src0, src1, coeff in preds.get(node, []):
            new_times.append((remap[src0], remap[src1], coeff, remap[node]))
    new_mapping = [remap[node] for node in mapping]

    return basic, new_times, new_mapping, node_count


class TensorState:
    def __init__(self, leaves, open_counts, edges, components, recipe=None):
        self.leaves = leaves
        self.open_counts = open_counts
        self.edges = edges
        self.components = components
        self.recipe = recipe

    @property
    def degree(self):
        return len(self.leaves)

    @property
    def rank(self):
        return sum(self.open_counts)

    @property
    def signature(self):
        return (self.leaves, self.open_counts, flatten_edges(self.edges))


class Generator:
    def __init__(self, args):
        self.args = args
        self.next_node_id = 0
        self.alpha_index_basic = []
        self.alpha_index_times = []
        self.basic_node_ids = {}
        self.primitive_states = []
        self.scalar_node_ids = []
        self.states_by_degree = defaultdict(list)
        self.seen_states = {}
        self.primitive_leaves = []
        self.states_per_degree = {}
        self._scalar_state_cache = None
        self._raw_node_poly_cache = None
        self._compact_candidate_cache = {}
        self._compact_state_cost_cache = {}
        self._compact_state_choice_cache = {}
        self._compact_basis_cache = None
        self._compact_projection_cost_cache = {}
        self._compact_projection_choice_cache = {}
        self._pure_cut_state_cost_cache = {}
        self._pure_cut_state_choice_cache = {}
        self._pure_cut_basis_cache = None
        self._family_basis_cache = None
        self._staged_family_basis_cache = None
        self._staged_family_matches_cache = None
        self._staged_family_is_better_cache = None
        self._staged_block_family_key_cache = {}
        self._same_rank_degree2_scalar_signature_map_cache = None
        self._degree2_block_signatures_cache = None
        self._exact_scalar_block_lift_parts_cache = {}
        self._degree3_block_lift_parts_cache = {}
        self._selected_basis_cache = {}
        self._resolved_emit_mode_cache = {}
        self._legacy_candidate_cache = {}
        self._product_candidate_cache = {}
        self._projection_dependency_cache = {}
        self._projection_dependency_weight_cache = {}
        self._projection_dependency_summary_cache = {}
        self._selected_scalar_states_cache = None
        self._reachable_level_cap_cache = {}

    def effective_lmax(self, degree):
        return self.args.degree_l_max.get(degree, self.args.l)

    def effective_lmax_summary(self):
        return ", ".join(
            f"d{degree}:{self.effective_lmax(degree)}"
            for degree in range(1, self.args.max_degree + 1)
        )

    def effective_level_max(self, degree):
        if degree in self.args.degree_level_max:
            return self.args.degree_level_max[degree]
        return self.args.max_level

    def effective_level_max_summary(self):
        pieces = []
        for degree in range(1, self.args.max_degree + 1):
            level_max = self.effective_level_max(degree)
            if level_max is None:
                pieces.append(f"d{degree}:inf")
            else:
                pieces.append(f"d{degree}:{level_max}")
        return ", ".join(pieces)

    @staticmethod
    def state_level(state):
        return sum(leaf[1] for leaf in state.leaves)

    def include_state_by_level(self, state):
        level_max = self.effective_level_max(state.degree)
        if level_max is None:
            return True
        return self.state_level(state) <= level_max

    def max_reachable_level_from_degree(self, degree):
        if degree in self._reachable_level_cap_cache:
            return self._reachable_level_cap_cache[degree]

        caps = []
        for output_degree in range(degree, self.args.max_degree + 1):
            level_max = self.effective_level_max(output_degree)
            if level_max is None:
                self._reachable_level_cap_cache[degree] = None
                return None
            caps.append(level_max)

        value = max(caps) if caps else None
        self._reachable_level_cap_cache[degree] = value
        return value

    def level_can_reach_output_degree(self, degree, level):
        max_level = self.max_reachable_level_from_degree(degree)
        if max_level is None:
            return True
        return level <= max_level

    def max_future_contractible_rank(self, degree):
        return sum(
            self.effective_lmax(next_degree)
            for next_degree in range(degree + 1, self.args.max_degree + 1)
        )

    def state_can_reach_scalar(self, degree, total_open_rank):
        return total_open_rank <= self.max_future_contractible_rank(degree)

    def alloc_node(self):
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    @staticmethod
    def reorder_edges(edges, order):
        return tuple(
            tuple(edges[order[i]][order[j]] for j in range(len(order)))
            for i in range(len(order))
        )

    @staticmethod
    def reorder_components(open_counts, components, order):
        old_layout = block_layout(open_counts)
        new_open_counts = tuple(open_counts[idx] for idx in order)
        new_layout = block_layout(new_open_counts)
        new_components = [-1] * old_layout.total_size

        for old_index_tuple in old_layout.index_tuples:
            old_flat = old_layout.flatten(old_index_tuple)
            new_index_tuple = tuple(old_index_tuple[idx] for idx in order)
            new_flat = new_layout.flatten(new_index_tuple)
            new_components[new_flat] = components[old_flat]

        return new_components

    @staticmethod
    def canonical_order(leaves, open_counts, edges):
        groups = defaultdict(list)
        for idx, leaf in enumerate(leaves):
            groups[leaf].append(idx)

        permutable_groups = [
            tuple(indices)
            for _leaf, indices in sorted(groups.items())
            if len(indices) > 1
        ]
        if not permutable_groups:
            return tuple(range(len(leaves)))

        best_order = None
        best_key = None

        group_permutations = [
            tuple(itertools.permutations(indices))
            for indices in permutable_groups
        ]

        for combo in itertools.product(*group_permutations):
            order = list(range(len(leaves)))
            for indices, perm in zip(permutable_groups, combo):
                for position, src_idx in zip(indices, perm):
                    order[position] = src_idx

            order_tuple = tuple(order)
            permuted_open_counts = tuple(open_counts[idx] for idx in order_tuple)
            permuted_edges = Generator.reorder_edges(edges, order_tuple)
            key = (permuted_open_counts, flatten_edges(permuted_edges))

            if best_key is None or key < best_key:
                best_key = key
                best_order = order_tuple

        if best_order is None:
            raise RuntimeError("failed to compute canonical order")
        return best_order

    def canonicalize_structure(self, leaves, open_counts, edges):
        order = self.canonical_order(leaves, open_counts, edges)
        canonical_leaves = tuple(leaves[idx] for idx in order)
        canonical_open_counts = tuple(open_counts[idx] for idx in order)
        canonical_edges = self.reorder_edges(edges, order)
        return canonical_leaves, canonical_open_counts, canonical_edges, order

    def init_primitives(self):
        for k in range(self.args.k):
            for rank in range(self.args.l + 1):
                mu = k * (self.args.l + 1) + rank
                leaf = (mu, rank)
                self.primitive_leaves.append(leaf)

                component_ids = []
                for exp_idx, exp in enumerate(exponent_tuples(rank)):
                    node_id = self.alloc_node()
                    self.alpha_index_basic.append((mu, exp[0], exp[1], exp[2]))
                    self.basic_node_ids[(leaf, exp_idx)] = node_id
                    component_ids.append(node_id)

                state = TensorState(
                    leaves=(leaf,),
                    open_counts=(rank,),
                    edges=((0,),),
                    components=component_ids,
                    recipe=None,
                )
                self.primitive_states.append(state)
                if self.level_can_reach_output_degree(1, rank):
                    self.seen_states[state.signature] = state
                    self.states_by_degree[1].append(state)
                    if rank == 0:
                        self.scalar_node_ids.append(component_ids[0])

    def build_edges(self, state, contractions):
        n = len(state.leaves)
        edges = [list(row) + [0] for row in state.edges]
        edges.append([0] * (n + 1))
        for i, val in enumerate(contractions):
            edges[i][n] = val
            edges[n][i] = val
        return tuple(tuple(row) for row in edges)

    def build_components(self, state, leaf, contractions, new_open_counts):
        old_layout = block_layout(state.open_counts)
        new_layout = block_layout(new_open_counts)
        gamma_lists = [exponent_tuples(val) for val in contractions]
        leaf_rank = leaf[1]
        new_components = [-1] * new_layout.total_size

        for out_index_tuple in new_layout.index_tuples:
            out_flat = new_layout.flatten(out_index_tuple)
            dst = self.alloc_node()
            new_components[out_flat] = dst

            out_exps = [
                exponent_tuples(rank)[idx]
                for rank, idx in zip(new_open_counts, out_index_tuple)
            ]

            new_leaf_open_exp = out_exps[-1]
            accum = defaultdict(int)

            for gamma_tuple in itertools.product(*gamma_lists):
                leaf_exp = new_leaf_open_exp
                coeff = 1
                input_indices = []

                for block_idx, gamma in enumerate(gamma_tuple):
                    coeff *= multinomial(gamma)
                    input_exp = add_counts(out_exps[block_idx], gamma)
                    input_indices.append(
                        exponent_to_index(state.open_counts[block_idx])[input_exp]
                    )
                    leaf_exp = add_counts(leaf_exp, gamma)

                input_flat = old_layout.flatten(tuple(input_indices))
                src_state = state.components[input_flat]
                src_leaf = self.basic_node_ids[
                    (leaf, exponent_to_index(leaf_rank)[leaf_exp])
                ]
                accum[(src_state, src_leaf)] += coeff

            for (src_state, src_leaf), coeff in sorted(accum.items()):
                self.alpha_index_times.append((src_state, src_leaf, coeff, dst))

        return new_components

    def generate(self):
        self.init_primitives()

        for degree in range(1, self.args.max_degree + 1):
            self.states_per_degree[degree] = len(self.states_by_degree[degree])

        for degree in range(1, self.args.max_degree):
            current_states = list(self.states_by_degree[degree])
            for state in current_states:
                last_leaf_mu = state.leaves[-1][0]
                target_degree = degree + 1
                target_lmax = self.effective_lmax(target_degree)

                for leaf in self.primitive_leaves:
                    # Canonical non-decreasing mu order removes direct K-swap duplicates.
                    if leaf[0] < last_leaf_mu:
                        continue
                    if leaf[1] > target_lmax:
                        continue
                    target_level = self.state_level(state) + leaf[1]
                    if not self.level_can_reach_output_degree(
                        target_degree,
                        target_level,
                    ):
                        continue

                    for contractions in contraction_patterns(
                        state.open_counts,
                        leaf[1],
                        self.args.l,
                    ):
                        contracted = sum(contractions)
                        new_open_counts = list(state.open_counts)
                        for i, val in enumerate(contractions):
                            new_open_counts[i] -= val
                        new_open_counts.append(leaf[1] - contracted)
                        new_open_counts_tuple = tuple(new_open_counts)
                        total_open_rank = sum(new_open_counts_tuple)
                        if total_open_rank > target_lmax:
                            continue
                        if not self.state_can_reach_scalar(
                            target_degree,
                            total_open_rank,
                        ):
                            continue
                        new_edges = self.build_edges(state, contractions)
                        new_leaves = state.leaves + (leaf,)
                        (
                            canonical_leaves,
                            canonical_open_counts,
                            canonical_edges,
                            canonical_order,
                        ) = self.canonicalize_structure(
                            new_leaves,
                            new_open_counts_tuple,
                            new_edges,
                        )
                        signature = (
                            canonical_leaves,
                            canonical_open_counts,
                            flatten_edges(canonical_edges),
                        )

                        if signature in self.seen_states:
                            continue

                        raw_components = self.build_components(
                            state,
                            leaf,
                            contractions,
                            new_open_counts_tuple,
                        )
                        new_components = self.reorder_components(
                            new_open_counts_tuple,
                            raw_components,
                            canonical_order,
                        )
                        new_state = TensorState(
                            leaves=canonical_leaves,
                            open_counts=canonical_open_counts,
                            edges=canonical_edges,
                            components=new_components,
                            recipe={
                                "parent_signature": state.signature,
                                "leaf": leaf,
                                "contractions": tuple(contractions),
                                "new_open_counts": new_open_counts_tuple,
                                "canonical_order": tuple(canonical_order),
                            },
                        )
                        self.seen_states[signature] = new_state
                        self.states_by_degree[degree + 1].append(new_state)
                        self.states_per_degree[degree + 1] = len(
                            self.states_by_degree[degree + 1]
                        )
                        if new_state.rank == 0:
                            self.scalar_node_ids.append(new_components[0])

    def pruned_basis(self):
        scalar_ids = sorted(set(self.scalar_node_ids))
        incoming = defaultdict(list)
        for src0, src1, coeff, dst in self.alpha_index_times:
            incoming[dst].append((src0, src1, coeff, dst))

        used = set(scalar_ids)
        stack = list(scalar_ids)
        while stack:
            node = stack.pop()
            for src0, src1, _coeff, _dst in incoming.get(node, []):
                if src0 not in used:
                    used.add(src0)
                    stack.append(src0)
                if src1 not in used:
                    used.add(src1)
                    stack.append(src1)

        remap = {}
        next_id = 0
        for old_id in range(self.next_node_id):
            if old_id in used:
                remap[old_id] = next_id
                next_id += 1

        pruned_basic = []
        for old_id, entry in enumerate(self.alpha_index_basic):
            if old_id in used:
                pruned_basic.append(entry)

        pruned_times = []
        for src0, src1, coeff, dst in self.alpha_index_times:
            if dst not in used:
                continue
            pruned_times.append((remap[src0], remap[src1], coeff, remap[dst]))

        pruned_mapping = [remap[node] for node in scalar_ids]
        return pruned_basic, pruned_times, pruned_mapping, next_id

    def fingerprint_groups(self):
        node_count = self.next_node_id
        basic_count = len(self.alpha_index_basic)
        incoming = defaultdict(list)
        for src0, src1, coeff, dst in self.alpha_index_times:
            incoming[dst].append((src0, src1, coeff))

        signatures = {}
        for node_id in range(node_count):
            signatures[node_id] = []

        for seed in FINGERPRINT_SEEDS:
            rng = __import__("random").Random(seed)
            values = [0] * node_count
            for node_id in range(basic_count):
                values[node_id] = rng.randrange(1, FINGERPRINT_MODULUS)
            for node_id in range(basic_count, node_count):
                acc = 0
                for src0, src1, coeff in incoming.get(node_id, []):
                    acc = (acc + coeff * values[src0] * values[src1]) % FINGERPRINT_MODULUS
                values[node_id] = acc
            for node_id, value in enumerate(values):
                signatures[node_id].append(value)

        groups = defaultdict(list)
        for node_id, signature in signatures.items():
            groups[tuple(signature)].append(node_id)
        return groups

    def scalar_states(self):
        if self._scalar_state_cache is None:
            states = []
            for degree in sorted(self.states_by_degree):
                for state in self.states_by_degree[degree]:
                    if state.rank == 0:
                        states.append(state)
            self._scalar_state_cache = states
        return list(self._scalar_state_cache)

    def scalar_state_signature_map(self):
        mapping = {}
        for state in self.scalar_states():
            mapping[state.signature] = state.components[0]
        return mapping

    def include_scalar_state(self, state):
        if self.args.include_direct_scalar_products:
            return True
        return self.connected_component_count(state) != 2

    def selected_scalar_states(self):
        if self._selected_scalar_states_cache is not None:
            return list(self._selected_scalar_states_cache)

        states = [
            state
            for state in self.scalar_states()
            if self.include_scalar_state(state) and self.include_state_by_level(state)
        ]
        if self.args.include_k_relabel_equivalent_scalars:
            self._selected_scalar_states_cache = tuple(states)
            return list(self._selected_scalar_states_cache)

        passthrough_states = []
        grouped_states = defaultdict(list)
        for state in states:
            if state.degree < self.args.strong_k_family_from_degree:
                passthrough_states.append(state)
                continue
            family_key, canonical_to_actual = self.scalar_k_relabel_family_info(state)
            grouped_states[family_key].append((state, canonical_to_actual))

        selected_states = list(passthrough_states)
        used_dependencies = set()
        ordered_groups = sorted(
            grouped_states.values(),
            key=lambda group: min(state.components[0] for state, _canonical_to_actual in group),
        )
        for group in ordered_groups:
            if len(group) == 1:
                best_state = group[0][0]
                selected_states.append(best_state)
                used_dependencies.update(
                    self.projection_dependency_keys(best_state.signature, 0)
                )
                continue

            anchor_state, anchor_canonical_to_actual = min(
                group,
                key=lambda item: item[0].components[0],
            )
            anchor_dep_keys, anchor_dep_items, anchor_total = self.projection_dependency_summary(
                anchor_state.signature,
                0,
            )
            anchor_actual_to_canonical = {
                actual_channel: canonical_channel
                for canonical_channel, actual_channel in enumerate(anchor_canonical_to_actual)
            }

            best_state = None
            best_score = None
            best_dep_keys = None
            for state, state_canonical_to_actual in sorted(
                group,
                key=lambda item: item[0].components[0],
            ):
                if state.signature == anchor_state.signature:
                    dep_keys = anchor_dep_keys
                    dep_items = anchor_dep_items
                    total = anchor_total
                else:
                    channel_remap = {
                        actual_channel: state_canonical_to_actual[
                            anchor_actual_to_canonical[actual_channel]
                        ]
                        for actual_channel in anchor_actual_to_canonical
                    }
                    dep_keys = tuple(
                        self.relabel_projection_key(dep_key, channel_remap)
                        for dep_key in anchor_dep_keys
                    )
                    dep_items = tuple(
                        (
                            self.relabel_projection_key(dep_key, channel_remap),
                            weight,
                        )
                        for dep_key, weight in anchor_dep_items
                    )
                    total = anchor_total

                marginal = sum(
                    weight for dep_key, weight in dep_items if dep_key not in used_dependencies
                )
                score = (marginal, total, state.components[0])
                if best_score is None or score < best_score:
                    best_score = score
                    best_state = state
                    best_dep_keys = dep_keys

            selected_states.append(best_state)
            used_dependencies.update(best_dep_keys)

        selected_states.sort(key=lambda state: state.components[0])
        self._selected_scalar_states_cache = tuple(selected_states)
        return list(self._selected_scalar_states_cache)

    def selected_scalar_node_ids(self):
        return [state.components[0] for state in self.selected_scalar_states()]

    def selected_scalar_degree_hist(self):
        hist = defaultdict(int)
        for state in self.selected_scalar_states():
            hist[state.degree] += 1
        return dict(sorted(hist.items()))

    def selected_scalar_level_hist(self):
        hist = defaultdict(int)
        for state in self.selected_scalar_states():
            hist[self.state_level(state)] += 1
        return dict(sorted(hist.items()))

    @staticmethod
    def connected_vertex_sets(state):
        n = len(state.leaves)
        seen = [False] * n
        components = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            vertices = []
            while stack:
                cur = stack.pop()
                vertices.append(cur)
                for nxt in range(n):
                    if cur != nxt and not seen[nxt] and state.edges[cur][nxt] > 0:
                        seen[nxt] = True
                        stack.append(nxt)
            components.append(tuple(sorted(vertices)))
        return tuple(components)

    def scalar_component_signatures(self, state):
        signatures = []
        for vertices in self.connected_vertex_sets(state):
            leaves = tuple(state.leaves[idx] for idx in vertices)
            open_counts = tuple(0 for _ in vertices)
            edges = tuple(
                tuple(state.edges[vertices[i]][vertices[j]] for j in range(len(vertices)))
                for i in range(len(vertices))
            )
            canonical_leaves, canonical_open_counts, canonical_edges, _order = self.canonicalize_structure(
                leaves,
                open_counts,
                edges,
            )
            signatures.append(
                (
                    canonical_leaves,
                    canonical_open_counts,
                    flatten_edges(canonical_edges),
                )
            )
        return tuple(signatures)

    @staticmethod
    def degree2_block_rank(signature):
        leaves, _open_counts, flat_edges = signature
        if len(leaves) != 2:
            return None
        if not any(flat_edges):
            return None
        return 2

    @staticmethod
    def same_rank_degree2_block_rank(signature):
        if Generator.degree2_block_rank(signature) is None:
            return None
        leaves, _open_counts, _flat_edges = signature
        rank = leaves[0][1]
        if leaves[1][1] != rank:
            return None
        return rank

    def is_degree2_block_signature(self, signature):
        return self.degree2_block_rank(signature) is not None

    def is_same_rank_degree2_block_signature(self, signature):
        return self.same_rank_degree2_block_rank(signature) is not None

    def staged_block_family_key(self, signature):
        if signature in self._staged_block_family_key_cache:
            return self._staged_block_family_key_cache[signature]

        if self.degree2_block_rank(signature) is None:
            self._staged_block_family_key_cache[signature] = None
            return None

        state = self.seen_states.get(signature)
        if state is None:
            self._staged_block_family_key_cache[signature] = None
            return None

        raw_polys, _labels = self.raw_node_polynomials()
        component_polys = tuple(
            self.canonicalize_raw_polynomial(raw_polys[node])[0]
            for node in state.components
        )
        same_rank = self.same_rank_degree2_block_rank(signature)
        key = (
            "degree2-block",
            same_rank,
            state.open_counts,
            component_polys,
        )
        self._staged_block_family_key_cache[signature] = key
        return key

    def same_rank_degree2_scalar_signature_map(self):
        if self._same_rank_degree2_scalar_signature_map_cache is not None:
            return {
                rank: dict(items)
                for rank, items in self._same_rank_degree2_scalar_signature_map_cache.items()
            }

        out = defaultdict(dict)
        for state in self.states_by_degree[2]:
            if state.rank != 0:
                continue
            rank = self.same_rank_degree2_block_rank(state.signature)
            if rank is None:
                continue
            out[rank][tuple(state.leaves)] = state.signature

        self._same_rank_degree2_scalar_signature_map_cache = {
            rank: dict(items) for rank, items in out.items()
        }
        return {
            rank: dict(items)
            for rank, items in self._same_rank_degree2_scalar_signature_map_cache.items()
        }

    def same_rank_degree2_scalar_signatures(self):
        signatures = set()
        for items in self.same_rank_degree2_scalar_signature_map().values():
            signatures.update(items.values())
        return signatures

    def degree2_block_signatures(self):
        if self._degree2_block_signatures_cache is not None:
            return set(self._degree2_block_signatures_cache)

        signatures = {
            signature
            for signature in self.seen_states
            if self.is_degree2_block_signature(signature)
        }
        self._degree2_block_signatures_cache = frozenset(signatures)
        return set(self._degree2_block_signatures_cache)

    def exact_scalar_block_lift_parts(self, signature):
        if signature in self._exact_scalar_block_lift_parts_cache:
            return self._exact_scalar_block_lift_parts_cache[signature]

        state = self.seen_states.get(signature)
        if state is None or state.degree != 3 or state.rank != 0:
            self._exact_scalar_block_lift_parts_cache[signature] = None
            return None

        degree2_blocks = self.degree2_block_signatures()
        primitive_scalar_leaves = {
            leaf for leaf in self.primitive_leaves if leaf[1] == 0
        }
        for index, leaf in enumerate(state.leaves):
            if leaf not in primitive_scalar_leaves:
                continue

            other_vertices = tuple(
                vertex for vertex in range(len(state.leaves)) if vertex != index
            )
            other_leaves = tuple(state.leaves[vertex] for vertex in other_vertices)
            other_open_counts = tuple(0 for _ in other_vertices)
            other_edges = tuple(
                tuple(
                    state.edges[other_vertices[row]][other_vertices[col]]
                    for col in range(len(other_vertices))
                )
                for row in range(len(other_vertices))
            )
            (
                canonical_leaves,
                canonical_open_counts,
                canonical_edges,
                _order,
            ) = self.canonicalize_structure(
                other_leaves,
                other_open_counts,
                other_edges,
            )
            other_signature = (
                canonical_leaves,
                canonical_open_counts,
                flatten_edges(canonical_edges),
            )
            if other_signature in degree2_blocks:
                self._exact_scalar_block_lift_parts_cache[signature] = (
                    leaf,
                    other_signature,
                )
                return self._exact_scalar_block_lift_parts_cache[signature]

        self._exact_scalar_block_lift_parts_cache[signature] = None
        return None

    def exact_scalar_block_lift_signatures(self):
        return {
            signature
            for signature in self.seen_states
            if self.exact_scalar_block_lift_parts(signature) is not None
        }

    def degree3_block_lift_parts(self, signature):
        if signature in self._degree3_block_lift_parts_cache:
            return self._degree3_block_lift_parts_cache[signature]

        state = self.seen_states.get(signature)
        if state is None or state.degree != 3:
            self._degree3_block_lift_parts_cache[signature] = None
            return None

        degree2_blocks = self.degree2_block_signatures()
        for index, leaf in enumerate(state.leaves):
            other_vertices = tuple(
                vertex for vertex in range(len(state.leaves)) if vertex != index
            )
            other_leaves = tuple(state.leaves[vertex] for vertex in other_vertices)
            other_open_counts = tuple(0 for _ in other_vertices)
            other_edges = tuple(
                tuple(
                    state.edges[other_vertices[row]][other_vertices[col]]
                    for col in range(len(other_vertices))
                )
                for row in range(len(other_vertices))
            )
            (
                canonical_leaves,
                canonical_open_counts,
                canonical_edges,
                _order,
            ) = self.canonicalize_structure(
                other_leaves,
                other_open_counts,
                other_edges,
            )
            other_signature = (
                canonical_leaves,
                canonical_open_counts,
                flatten_edges(canonical_edges),
            )
            if other_signature in degree2_blocks:
                self._degree3_block_lift_parts_cache[signature] = (
                    leaf,
                    other_signature,
                )
                return self._degree3_block_lift_parts_cache[signature]

        self._degree3_block_lift_parts_cache[signature] = None
        return None

    def degree3_block_lift_signatures(self):
        return {
            signature
            for signature in self.seen_states
            if self.degree3_block_lift_parts(signature) is not None
        }

    def dedup_pruned_basis(self):
        groups = self.fingerprint_groups()
        representative = {}
        for nodes in groups.values():
            rep = min(nodes)
            for node in nodes:
                representative[node] = rep

        dedup_times = defaultdict(int)
        for src0, src1, coeff, dst in self.alpha_index_times:
            rep_dst = representative[dst]
            if rep_dst != dst:
                continue
            rep_src0 = representative[src0]
            rep_src1 = representative[src1]
            dedup_times[(rep_src0, rep_src1, rep_dst)] += coeff

        incoming = defaultdict(list)
        for (src0, src1, dst), coeff in dedup_times.items():
            incoming[dst].append((src0, src1, coeff))

        signature_to_scalar_node = self.scalar_state_signature_map()
        rewrite = {}
        for state in self.scalar_states():
            state_node = state.components[0]
            rep_node = representative[state_node]
            if rep_node != state_node:
                continue
            component_signatures = self.scalar_component_signatures(state)
            if len(component_signatures) <= 1:
                continue
            component_nodes = []
            missing_component = False
            for signature in component_signatures:
                component_node = signature_to_scalar_node.get(signature)
                if component_node is None:
                    missing_component = True
                    break
                component_nodes.append(representative[component_node])
            if missing_component:
                continue
            rewrite[rep_node] = tuple(component_nodes)

        basic_count = len(self.alpha_index_basic)
        emitted_basic = list(self.alpha_index_basic)
        emitted_times_dict = defaultdict(int)
        emitted_map = {}
        product_cache = {}
        next_id = basic_count

        def emit_product(factors):
            nonlocal next_id
            if len(factors) == 1:
                return factors[0]
            key = tuple(sorted(factors))
            if key in product_cache:
                return product_cache[key]
            split = len(key) // 2
            left = emit_product(key[:split])
            right = emit_product(key[split:])
            dst = next_id
            next_id += 1
            emitted_times_dict[(left, right, dst)] += 1
            product_cache[key] = dst
            return dst

        def emit_node(node):
            nonlocal next_id
            if node < basic_count:
                return node
            if node in emitted_map:
                return emitted_map[node]
            if node in rewrite:
                emitted = emit_product(tuple(emit_node(child) for child in rewrite[node]))
                emitted_map[node] = emitted
                return emitted
            dst = next_id
            next_id += 1
            emitted_map[node] = dst
            for src0, src1, coeff in incoming.get(node, []):
                out0 = emit_node(src0)
                out1 = emit_node(src1)
                emitted_times_dict[(out0, out1, dst)] += coeff
            return dst

        scalar_ids = sorted(set(representative[node] for node in self.selected_scalar_node_ids()))
        pruned_mapping = [emit_node(node) for node in scalar_ids]

        pruned_times = []
        for (src0, src1, dst), coeff in sorted(emitted_times_dict.items()):
            pruned_times.append((src0, src1, coeff, dst))

        return emitted_basic, pruned_times, pruned_mapping, next_id

    def scalar_polynomial(self, state):
        edge_list = []
        for i in range(len(state.leaves)):
            for j in range(i + 1, len(state.leaves)):
                for _ in range(state.edges[i][j]):
                    edge_list.append((i, j))

        zero_counts = tuple((0, 0, 0) for _ in state.leaves)
        dp = {zero_counts: 1}

        for i, j in edge_list:
            new_dp = defaultdict(int)
            for counts, weight in dp.items():
                for axis in range(3):
                    next_counts = [list(item) for item in counts]
                    next_counts[i][axis] += 1
                    next_counts[j][axis] += 1
                    next_key = tuple(tuple(item) for item in next_counts)
                    new_dp[next_key] += weight
            dp = new_dp

        poly = defaultdict(int)
        for counts, weight in dp.items():
            factors = []
            for leaf, exp in zip(state.leaves, counts):
                exp_idx = exponent_to_index(leaf[1])[exp]
                factors.append(self.basic_node_ids[(leaf, exp_idx)])
            poly[tuple(sorted(factors))] += weight

        return poly

    @staticmethod
    def signature_edges(signature):
        leaves, _open_counts, flat_edges = signature
        n = len(leaves)
        edges = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges[i][j] = flat_edges[idx]
                edges[j][i] = flat_edges[idx]
                idx += 1
        return tuple(tuple(row) for row in edges)

    @staticmethod
    def graph_cost_key(nodes, edges):
        return (nodes + edges, edges, nodes)

    @staticmethod
    def edge_first_cost_key(nodes, edges):
        return (edges, nodes)

    def basis_evaluation_labels(self, basic, times, node_count):
        by_dst = defaultdict(list)
        for src0, src1, coeff, dst in times:
            by_dst[dst].append((src0, src1, coeff))

        basic_tokens = self.basic_tokens(basic)
        labels = [None] * node_count

        for idx in range(len(basic_tokens)):
            labels[idx] = tuple(sorted((basic_tokens[idx][0],)))

        for dst in range(len(basic_tokens), node_count):
            label_set = set()
            for src0, src1, _coeff in by_dst.get(dst, []):
                label_set.update(labels[src0])
                label_set.update(labels[src1])
            labels[dst] = tuple(sorted(label_set))

        return labels

    def basis_fingerprint_groups(
        self,
        basic,
        times,
        node_count,
        seeds=FINGERPRINT_SEEDS,
        modulus=FINGERPRINT_MODULUS,
    ):
        by_dst = defaultdict(list)
        for src0, src1, coeff, dst in times:
            by_dst[dst].append((src0, src1, coeff))

        signatures = defaultdict(list)
        labels = self.basis_evaluation_labels(basic, times, node_count)

        for seed in seeds:
            rng = __import__("random").Random(seed)
            values = [0] * node_count
            for node_id in range(len(basic)):
                values[node_id] = rng.randrange(1, modulus)
            for node_id in range(len(basic), node_count):
                acc = 0
                for src0, src1, coeff in by_dst.get(node_id, []):
                    acc = (acc + coeff * values[src0] * values[src1]) % modulus
                values[node_id] = acc
            for node_id, value in enumerate(values):
                signatures[(tuple(labels[node_id]), node_id)].append(value)

        groups = defaultdict(list)
        for (_labels, node_id), signature in signatures.items():
            groups[(tuple(labels[node_id]), tuple(signature))].append(node_id)
        return groups

    def basis_fingerprint_set(
        self,
        basic,
        times,
        mapping,
        node_count,
        seeds=FINGERPRINT_SEEDS,
        modulus=FINGERPRINT_MODULUS,
    ):
        by_dst = defaultdict(list)
        for src0, src1, coeff, dst in times:
            by_dst[dst].append((src0, src1, coeff))

        values_by_node = defaultdict(list)
        for seed in seeds:
            rng = __import__("random").Random(seed)
            values = [0] * node_count
            for node_id in range(len(basic)):
                values[node_id] = rng.randrange(1, modulus)
            for node_id in range(len(basic), node_count):
                acc = 0
                for src0, src1, coeff in by_dst.get(node_id, []):
                    acc = (acc + coeff * values[src0] * values[src1]) % modulus
                values[node_id] = acc
            for node_id in mapping:
                values_by_node[node_id].append(values[node_id])
        return {tuple(items) for items in values_by_node.values()}

    def dedup_emitted_basis(self, basic, times, mapping, node_count):
        groups = self.basis_fingerprint_groups(basic, times, node_count)
        representative = {}
        for nodes in groups.values():
            rep = min(nodes)
            for node in nodes:
                representative[node] = rep

        merged_times = defaultdict(int)
        for src0, src1, coeff, dst in times:
            rep_src0 = representative[src0]
            rep_src1 = representative[src1]
            rep_dst = representative[dst]
            if rep_dst != dst:
                continue
            key_src = tuple(sorted((rep_src0, rep_src1)))
            merged_times[(key_src[0], key_src[1], rep_dst)] += coeff

        used = set(representative[node] for node in mapping)
        incoming = defaultdict(list)
        for (src0, src1, dst), coeff in merged_times.items():
            incoming[dst].append((src0, src1, coeff))

        stack = list(used)
        while stack:
            node = stack.pop()
            for src0, src1, _coeff in incoming.get(node, []):
                if src0 not in used:
                    used.add(src0)
                    stack.append(src0)
                if src1 not in used:
                    used.add(src1)
                    stack.append(src1)

        remap = {}
        next_id = 0
        for old_id in range(node_count):
            rep_id = representative[old_id]
            if rep_id != old_id or old_id not in used:
                continue
            remap[old_id] = next_id
            next_id += 1

        pruned_basic = [basic[idx] for idx in range(len(basic)) if idx in remap]
        pruned_times = []
        for (src0, src1, dst), coeff in sorted(merged_times.items()):
            if dst not in remap:
                continue
            pruned_times.append((remap[src0], remap[src1], coeff, remap[dst]))
        pruned_mapping = [remap[representative[node]] for node in mapping]
        return pruned_basic, pruned_times, pruned_mapping, next_id

    def dedup_emitted_basis_fixed_point(self, basic, times, mapping, node_count):
        previous_signature = None
        current = (basic, times, mapping, node_count)
        while True:
            deduped = self.dedup_emitted_basis(*current)
            deduped = topologically_renumber(*deduped)
            signature = (deduped[3], len(deduped[1]))
            if signature == previous_signature:
                return deduped
            previous_signature = signature
            current = deduped

    def basis_node_degrees(self, basic, times, node_count):
        incoming = defaultdict(list)
        for src0, src1, coeff, dst in times:
            incoming[dst].append((src0, src1, coeff))
        degrees = [None] * node_count
        for idx in range(len(basic)):
            degrees[idx] = 1
        for dst in range(len(basic), node_count):
            entries = incoming.get(dst, [])
            if not entries:
                degrees[dst] = 1
                continue
            degrees[dst] = degrees[entries[0][0]] + degrees[entries[0][1]]
        return degrees

    def prune_emitted_basis_targets(self, basic, times, mapping, node_count):
        incoming = defaultdict(list)
        for src0, src1, coeff, dst in times:
            incoming[dst].append((src0, src1, coeff))

        used = set(mapping)
        stack = list(mapping)
        while stack:
            node = stack.pop()
            for src0, src1, _coeff in incoming.get(node, []):
                if src0 not in used:
                    used.add(src0)
                    stack.append(src0)
                if src1 not in used:
                    used.add(src1)
                    stack.append(src1)

        remap = {}
        next_id = 0
        for old_id in range(node_count):
            if old_id in used:
                remap[old_id] = next_id
                next_id += 1

        pruned_basic = [basic[idx] for idx in range(len(basic)) if idx in used]
        pruned_times = []
        for src0, src1, coeff, dst in times:
            if dst not in used:
                continue
            pruned_times.append((remap[src0], remap[src1], coeff, remap[dst]))
        pruned_mapping = [remap[node] for node in mapping]
        return pruned_basic, pruned_times, pruned_mapping, next_id

    def dedup_k_relabel_scalar_outputs(self, basic, times, mapping, node_count):
        if self.args.include_k_relabel_equivalent_scalars:
            return basic, times, mapping, node_count

        polys, _labels, basic_tokens = self.emitted_node_polynomials(
            basic, times, node_count
        )
        degrees = self.basis_node_degrees(basic, times, node_count)
        dependency_cache = {}

        def closure(node):
            if node in dependency_cache:
                return dependency_cache[node]
            used = {node}
            incoming = defaultdict(list)
            for src0, src1, coeff, dst in times:
                incoming[dst].append((src0, src1))
            stack = [node]
            while stack:
                current = stack.pop()
                for src0, src1 in incoming.get(current, []):
                    if src0 not in used:
                        used.add(src0)
                        stack.append(src0)
                    if src1 not in used:
                        used.add(src1)
                        stack.append(src1)
            dependency_cache[node] = used
            return used

        groups = defaultdict(list)
        passthrough_nodes = []
        for node in mapping:
            if degrees[node] < self.args.strong_k_family_from_degree:
                passthrough_nodes.append(node)
                continue
            canonical_poly, _labels = self.canonicalize_emitted_polynomial(
                polys[node],
                basic_tokens,
            )
            groups[canonical_poly].append(node)

        selected = list(passthrough_nodes)
        used_dependencies = set()
        for nodes in sorted(groups.values(), key=lambda group: min(group)):
            best_node = None
            best_score = None
            for node in sorted(nodes):
                deps = closure(node)
                marginal = len(deps - used_dependencies)
                score = (marginal, len(deps), node)
                if best_score is None or score < best_score:
                    best_score = score
                    best_node = node
            selected.append(best_node)
            used_dependencies.update(closure(best_node))

        selected.sort()
        return self.prune_emitted_basis_targets(basic, times, selected, node_count)

    @staticmethod
    def merge_component_terms(component_terms, dst_count):
        merged = [defaultdict(int) for _ in range(dst_count)]
        for terms in component_terms:
            for dst_index, term_entries in enumerate(terms):
                for src0, src1, coeff in term_entries:
                    merged[dst_index][(src0, src1)] += coeff
        output_terms = []
        edge_count = 0
        for accum in merged:
            flat_terms = []
            for (src0, src1), coeff in sorted(accum.items()):
                flat_terms.append((src0, src1, coeff))
            output_terms.append(tuple(flat_terms))
            edge_count += len(flat_terms)
        return tuple(output_terms), edge_count

    def cut_left_sizes(self, degree):
        if degree == 3:
            return (1,)
        if degree == 4:
            return (1, 2)
        if degree == 5:
            return (1, 2)
        return ()

    def legacy_candidate(self, signature):
        if signature in self._legacy_candidate_cache:
            return self._legacy_candidate_cache[signature]

        state = self.seen_states.get(signature)
        if state is None or state.recipe is None:
            self._legacy_candidate_cache[signature] = None
            return None

        recipe = state.recipe
        candidate = self.legacy_candidate_from_recipe(
            recipe["parent_signature"],
            recipe["leaf"],
            recipe["contractions"],
            recipe["new_open_counts"],
            recipe["canonical_order"],
            state.open_counts,
        )
        self._legacy_candidate_cache[signature] = candidate
        return candidate

    def legacy_candidate_from_recipe(
        self,
        parent_signature,
        leaf,
        contractions,
        new_open_counts,
        canonical_order,
        canonical_open_counts,
    ):
        _parent_leaves, parent_open_counts, _parent_flat_edges = parent_signature
        old_layout = block_layout(parent_open_counts)
        pre_layout = block_layout(new_open_counts)
        canonical_layout = block_layout(canonical_open_counts)
        gamma_lists = [exponent_tuples(val) for val in contractions]
        leaf_rank = leaf[1]
        leaf_basis_nodes = tuple(
            self.basic_node_ids[(leaf, exp_idx)]
            for exp_idx in range(len(exponent_tuples(leaf_rank)))
        )
        output_terms = [tuple() for _ in range(canonical_layout.total_size)]

        for out_index_tuple in pre_layout.index_tuples:
            out_exps = [
                exponent_tuples(rank)[idx]
                for rank, idx in zip(new_open_counts, out_index_tuple)
            ]
            new_leaf_open_exp = out_exps[-1]
            accum = defaultdict(int)

            for gamma_tuple in itertools.product(*gamma_lists):
                leaf_exp = new_leaf_open_exp
                coeff = 1
                input_indices = []

                for block_idx, gamma in enumerate(gamma_tuple):
                    coeff *= multinomial(gamma)
                    input_exp = add_counts(out_exps[block_idx], gamma)
                    input_indices.append(
                        exponent_to_index(parent_open_counts[block_idx])[input_exp]
                    )
                    leaf_exp = add_counts(leaf_exp, gamma)

                input_flat = old_layout.flatten(tuple(input_indices))
                leaf_node = leaf_basis_nodes[exponent_to_index(leaf_rank)[leaf_exp]]
                accum[(input_flat, leaf_node)] += coeff

            canonical_index_tuple = tuple(
                out_index_tuple[idx] for idx in canonical_order
            )
            canonical_flat = canonical_layout.flatten(canonical_index_tuple)
            output_terms[canonical_flat] = tuple(
                (input_flat, leaf_node, coeff)
                for (input_flat, leaf_node), coeff in sorted(accum.items())
            )

        edge_count = sum(len(terms) for terms in output_terms)
        candidate = {
            "kind": "legacy",
            "child_signatures": (parent_signature,),
            "parent_signature": parent_signature,
            "leaf_basis_nodes": leaf_basis_nodes,
            "output_terms": tuple(output_terms),
            "node_count": canonical_layout.total_size,
            "edge_count": edge_count,
        }
        return candidate

    def direct_step_candidates(self, signature):
        leaves, open_counts, flat_edges = signature
        if len(leaves) <= 1:
            return tuple()

        edge_matrix = self.signature_edges(signature)
        output_candidates = []
        seen_keys = set()

        for leaf_index in range(len(leaves)):
            parent_indices = tuple(
                idx for idx in range(len(leaves)) if idx != leaf_index
            )
            leaf = leaves[leaf_index]
            contractions = tuple(
                edge_matrix[parent_idx][leaf_index]
                for parent_idx in parent_indices
            )
            parent_leaves_raw = tuple(leaves[parent_idx] for parent_idx in parent_indices)
            parent_open_raw = tuple(
                open_counts[parent_idx] + edge_matrix[parent_idx][leaf_index]
                for parent_idx in parent_indices
            )
            parent_edges_raw = tuple(
                tuple(edge_matrix[i][j] for j in parent_indices)
                for i in parent_indices
            )
            (
                parent_leaves,
                parent_open_counts,
                parent_edges,
                _parent_order,
            ) = self.canonicalize_structure(
                parent_leaves_raw,
                parent_open_raw,
                parent_edges_raw,
            )
            parent_signature = (
                parent_leaves,
                parent_open_counts,
                flatten_edges(parent_edges),
            )

            new_open_counts_raw = tuple(
                open_counts[parent_idx] for parent_idx in parent_indices
            ) + (open_counts[leaf_index],)
            new_edges_raw = [list(row) for row in parent_edges_raw]
            for row, contraction in zip(new_edges_raw, contractions):
                row.append(contraction)
            new_edges_raw.append(list(contractions) + [0])
            new_edges_raw = tuple(tuple(row) for row in new_edges_raw)
            (
                current_leaves,
                current_open_counts,
                current_edges,
                canonical_order,
            ) = self.canonicalize_structure(
                parent_leaves_raw + (leaf,),
                new_open_counts_raw,
                new_edges_raw,
            )
            current_signature = (
                current_leaves,
                current_open_counts,
                flatten_edges(current_edges),
            )
            if current_signature != signature:
                continue

            try:
                candidate = self.legacy_candidate_from_recipe(
                    parent_signature,
                    leaf,
                    contractions,
                    new_open_counts_raw,
                    canonical_order,
                    open_counts,
                )
            except (KeyError, IndexError):
                continue
            dedup_key = (
                candidate["parent_signature"],
                candidate["leaf_basis_nodes"],
                candidate["output_terms"],
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            output_candidates.append(candidate)

        return tuple(output_candidates)

    def signature_connected_vertex_sets(self, signature):
        leaves, _open_counts, flat_edges = signature
        edges = self.signature_edges(signature)
        n = len(leaves)
        seen = [False] * n
        components = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            vertices = []
            while stack:
                cur = stack.pop()
                vertices.append(cur)
                for nxt in range(n):
                    if cur != nxt and not seen[nxt] and edges[cur][nxt] > 0:
                        seen[nxt] = True
                        stack.append(nxt)
            components.append(tuple(sorted(vertices)))
        return tuple(components)

    def disconnected_scalar_candidate(self, signature):
        if signature in self._product_candidate_cache:
            return self._product_candidate_cache[signature]

        leaves, open_counts, _flat_edges = signature
        if sum(open_counts) != 0:
            self._product_candidate_cache[signature] = None
            return None

        component_signatures = []
        for vertices in self.signature_connected_vertex_sets(signature):
            component_leaves = tuple(leaves[idx] for idx in vertices)
            component_edges = tuple(
                tuple(self.signature_edges(signature)[vertices[i]][vertices[j]] for j in range(len(vertices)))
                for i in range(len(vertices))
            )
            canonical_leaves, canonical_open_counts, canonical_edges, _order = self.canonicalize_structure(
                component_leaves,
                tuple(0 for _ in vertices),
                component_edges,
            )
            component_signatures.append(
                (
                    canonical_leaves,
                    canonical_open_counts,
                    flatten_edges(canonical_edges),
                )
            )
        if len(component_signatures) <= 1:
            self._product_candidate_cache[signature] = None
            return None

        candidate = {
            "kind": "product",
            "factor_signatures": tuple(sorted(component_signatures)),
            "node_count": len(component_signatures) - 1,
            "edge_count": len(component_signatures) - 1,
        }
        self._product_candidate_cache[signature] = candidate
        return candidate

    def partition_candidates(self, signature, allowed_left_sizes):
        cache_key = (
            signature,
            tuple(allowed_left_sizes) if allowed_left_sizes is not None else None,
        )
        if cache_key in self._compact_candidate_cache:
            return self._compact_candidate_cache[cache_key]

        leaves, open_counts, _flat_edges = signature
        edge_matrix = self.signature_edges(signature)
        n = len(leaves)
        if allowed_left_sizes is None:
            allowed_left_sizes = tuple(range(1, n))
        allowed_left_sizes = set(allowed_left_sizes)
        if not allowed_left_sizes:
            self._compact_candidate_cache[cache_key] = tuple()
            return self._compact_candidate_cache[cache_key]

        parent_layout = block_layout(open_counts)
        parent_exp_lists = [exponent_tuples(rank) for rank in open_counts]
        all_indices = tuple(range(n))
        candidates = []

        for left_size in range(1, n):
            if left_size not in allowed_left_sizes:
                continue
            for left in itertools.combinations(all_indices, left_size):
                if 0 not in left:
                    continue
                if left_size > n - left_size:
                    continue

                right = tuple(idx for idx in all_indices if idx not in left)
                left_leaves = tuple(leaves[idx] for idx in left)
                right_leaves = tuple(leaves[idx] for idx in right)
                left_open_raw = tuple(
                    open_counts[idx] + sum(edge_matrix[idx][other] for other in right)
                    for idx in left
                )
                right_open_raw = tuple(
                    open_counts[idx] + sum(edge_matrix[idx][other] for other in left)
                    for idx in right
                )
                left_edges = tuple(
                    tuple(edge_matrix[i][j] for j in left)
                    for i in left
                )
                right_edges = tuple(
                    tuple(edge_matrix[i][j] for j in right)
                    for i in right
                )

                (
                    canonical_left_leaves,
                    canonical_left_open,
                    canonical_left_edges,
                    left_order,
                ) = self.canonicalize_structure(
                    left_leaves,
                    left_open_raw,
                    left_edges,
                )
                (
                    canonical_right_leaves,
                    canonical_right_open,
                    canonical_right_edges,
                    right_order,
                ) = self.canonicalize_structure(
                    right_leaves,
                    right_open_raw,
                    right_edges,
                )

                left_signature = (
                    canonical_left_leaves,
                    canonical_left_open,
                    flatten_edges(canonical_left_edges),
                )
                right_signature = (
                    canonical_right_leaves,
                    canonical_right_open,
                    flatten_edges(canonical_right_edges),
                )

                left_layout = block_layout(canonical_left_open)
                right_layout = block_layout(canonical_right_open)
                left_exp_lists = [exponent_tuples(rank) for rank in left_open_raw]
                right_exp_lists = [exponent_tuples(rank) for rank in right_open_raw]

                left_position = {state_idx: pos for pos, state_idx in enumerate(left)}
                right_position = {state_idx: pos for pos, state_idx in enumerate(right)}
                cross_pairs = []
                gamma_choices = []
                for left_idx in left:
                    for right_idx in right:
                        count = edge_matrix[left_idx][right_idx]
                        cross_pairs.append(
                            (
                                left_position[left_idx],
                                right_position[right_idx],
                                count,
                            )
                        )
                        gamma_choices.append(exponent_tuples(count))

                output_terms = []
                edge_count = 0
                for parent_index_tuple in parent_layout.index_tuples:
                    parent_exps = [
                        parent_exp_lists[pos][comp_idx]
                        for pos, comp_idx in enumerate(parent_index_tuple)
                    ]
                    accum = defaultdict(int)

                    for gamma_tuple in itertools.product(*gamma_choices):
                        left_counts = [
                            list(parent_exps[state_idx]) for state_idx in left
                        ]
                        right_counts = [
                            list(parent_exps[state_idx]) for state_idx in right
                        ]
                        coeff = 1

                        for (left_pos, right_pos, _count), gamma in zip(
                            cross_pairs,
                            gamma_tuple,
                        ):
                            coeff *= multinomial(gamma)
                            for axis in range(3):
                                left_counts[left_pos][axis] += gamma[axis]
                                right_counts[right_pos][axis] += gamma[axis]

                        left_index_raw = tuple(
                            exponent_to_index(left_open_raw[pos])[tuple(left_counts[pos])]
                            for pos in range(len(left))
                        )
                        right_index_raw = tuple(
                            exponent_to_index(right_open_raw[pos])[tuple(right_counts[pos])]
                            for pos in range(len(right))
                        )
                        left_index_canonical = tuple(
                            left_index_raw[pos] for pos in left_order
                        )
                        right_index_canonical = tuple(
                            right_index_raw[pos] for pos in right_order
                        )
                        accum[(left_index_canonical, right_index_canonical)] += coeff

                    flat_terms = []
                    for (left_index, right_index), coeff in sorted(accum.items()):
                        flat_terms.append((left_index, right_index, coeff))
                    output_terms.append(tuple(flat_terms))
                    edge_count += len(flat_terms)

                candidates.append(
                    {
                        "kind": "cut",
                        "left_signature": left_signature,
                        "right_signature": right_signature,
                        "left_layout": left_layout,
                        "right_layout": right_layout,
                        "output_terms": tuple(output_terms),
                        "node_count": parent_layout.total_size,
                        "edge_count": edge_count,
                    }
                )

        self._compact_candidate_cache[cache_key] = tuple(candidates)
        return self._compact_candidate_cache[cache_key]

    def cut_candidates(self, signature):
        return self.partition_candidates(
            signature,
            self.cut_left_sizes(len(signature[0])),
        )

    def state_candidates(self, signature):
        leaves, open_counts, _flat_edges = signature
        if len(leaves) == 1:
            return tuple()

        candidates = []
        legacy = self.legacy_candidate(signature)
        if legacy is not None:
            candidates.append(legacy)
        else:
            candidates.extend(self.direct_step_candidates(signature))

        product = self.disconnected_scalar_candidate(signature)
        if product is not None:
            candidates.append(product)

        candidates.extend(self.cut_candidates(signature))
        return tuple(candidates)

    def projection_candidates(self, signature, component_index):
        leaves, open_counts, _flat_edges = signature
        if len(leaves) == 1:
            return tuple()
        candidates = []
        for candidate in self.state_candidates(signature):
            if candidate["kind"] == "product":
                if component_index == 0:
                    candidates.append(candidate)
                continue
            if component_index < len(candidate["output_terms"]):
                candidates.append(candidate)
        return tuple(candidates)

    def compact_projection_cost(self, signature, component_index):
        cache_key = (signature, component_index)
        if cache_key in self._compact_projection_cost_cache:
            return self._compact_projection_cost_cache[cache_key]

        leaves, open_counts, _flat_edges = signature
        if len(leaves) == 1:
            self._compact_projection_cost_cache[cache_key] = (0, 0)
            self._compact_projection_choice_cache[cache_key] = None
            return self._compact_projection_cost_cache[cache_key]

        best_cost = None
        best_choice = None

        for candidate in self.projection_candidates(signature, component_index):
            if candidate["kind"] == "legacy":
                child_signature = candidate["child_signatures"][0]
                terms = candidate["output_terms"][component_index]
                unique_child_indices = sorted({child_index for child_index, _leaf_node, _coeff in terms})
                child_costs = [
                    self.compact_projection_cost(child_signature, child_index)
                    for child_index in unique_child_indices
                ]
                total_cost = (
                    1 + sum(nodes for nodes, _edges in child_costs),
                    len(terms) + sum(edges for _nodes, edges in child_costs),
                )
            elif candidate["kind"] == "product":
                factor_costs = [
                    self.compact_projection_cost(factor_signature, 0)
                    for factor_signature in candidate["factor_signatures"]
                ]
                extra = max(0, len(candidate["factor_signatures"]) - 1)
                total_cost = (
                    extra + sum(nodes for nodes, _edges in factor_costs),
                    extra + sum(edges for _nodes, edges in factor_costs),
                )
            elif candidate["kind"] == "cut":
                terms = candidate["output_terms"][component_index]
                left_signature = candidate["left_signature"]
                right_signature = candidate["right_signature"]
                left_indices = sorted(
                    {
                        candidate["left_layout"].flatten(left_index)
                        for left_index, _right_index, _coeff in terms
                    }
                )
                right_indices = sorted(
                    {
                        candidate["right_layout"].flatten(right_index)
                        for _left_index, right_index, _coeff in terms
                    }
                )
                left_costs = [
                    self.compact_projection_cost(left_signature, child_component_index)
                    for child_component_index in left_indices
                ]
                right_costs = [
                    self.compact_projection_cost(right_signature, child_component_index)
                    for child_component_index in right_indices
                ]
                total_cost = (
                    1
                    + sum(nodes for nodes, _edges in left_costs)
                    + sum(nodes for nodes, _edges in right_costs),
                    len(terms)
                    + sum(edges for _nodes, edges in left_costs)
                    + sum(edges for _nodes, edges in right_costs),
                )
            else:
                raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

            if best_cost is None or self.graph_cost_key(*total_cost) < self.graph_cost_key(
                *best_cost
            ):
                best_cost = total_cost
                best_choice = candidate

        if best_cost is None:
            raise RuntimeError(
                f"failed to build compact projection plan for state {signature}, component {component_index}"
            )

        self._compact_projection_cost_cache[cache_key] = best_cost
        self._compact_projection_choice_cache[cache_key] = best_choice
        return best_cost

    def compact_projection_choice(self, signature, component_index):
        self.compact_projection_cost(signature, component_index)
        return self._compact_projection_choice_cache[(signature, component_index)]

    def projection_candidate_total_cost(self, candidate, component_index):
        if candidate["kind"] == "legacy":
            child_signature = candidate["child_signatures"][0]
            terms = candidate["output_terms"][component_index]
            unique_child_indices = sorted(
                {child_index for child_index, _leaf_node, _coeff in terms}
            )
            child_costs = [
                self.compact_projection_cost(child_signature, child_index)
                for child_index in unique_child_indices
            ]
            return (
                1 + sum(nodes for nodes, _edges in child_costs),
                len(terms) + sum(edges for _nodes, edges in child_costs),
            )

        if candidate["kind"] == "product":
            factor_costs = [
                self.compact_projection_cost(factor_signature, 0)
                for factor_signature in candidate["factor_signatures"]
            ]
            extra = max(0, len(candidate["factor_signatures"]) - 1)
            return (
                extra + sum(nodes for nodes, _edges in factor_costs),
                extra + sum(edges for _nodes, edges in factor_costs),
            )

        if candidate["kind"] == "cut":
            terms = candidate["output_terms"][component_index]
            left_signature = candidate["left_signature"]
            right_signature = candidate["right_signature"]
            left_indices = sorted(
                {
                    candidate["left_layout"].flatten(left_index)
                    for left_index, _right_index, _coeff in terms
                }
            )
            right_indices = sorted(
                {
                    candidate["right_layout"].flatten(right_index)
                    for _left_index, right_index, _coeff in terms
                }
            )
            left_costs = [
                self.compact_projection_cost(left_signature, child_component_index)
                for child_component_index in left_indices
            ]
            right_costs = [
                self.compact_projection_cost(right_signature, child_component_index)
                for child_component_index in right_indices
            ]
            return (
                1
                + sum(nodes for nodes, _edges in left_costs)
                + sum(nodes for nodes, _edges in right_costs),
                len(terms)
                + sum(edges for _nodes, edges in left_costs)
                + sum(edges for _nodes, edges in right_costs),
            )

        raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

    def staged_projection_candidate_total_cost(
        self,
        candidate,
        component_index,
        block_signatures,
    ):
        if candidate["kind"] == "legacy":
            child_signature = candidate["child_signatures"][0]
            terms = candidate["output_terms"][component_index]
            unique_child_indices = sorted(
                {child_index for child_index, _leaf_node, _coeff in terms}
            )
            if child_signature in block_signatures:
                child_costs = [(0, 0) for _child_index in unique_child_indices]
            else:
                child_costs = [
                    self.compact_projection_cost(child_signature, child_index)
                    for child_index in unique_child_indices
                ]
            return (
                1 + sum(nodes for nodes, _edges in child_costs),
                len(terms) + sum(edges for _nodes, edges in child_costs),
            )

        if candidate["kind"] == "product":
            factor_costs = []
            for factor_signature in candidate["factor_signatures"]:
                if factor_signature in block_signatures:
                    factor_costs.append((0, 0))
                else:
                    factor_costs.append(self.compact_projection_cost(factor_signature, 0))
            extra = max(0, len(candidate["factor_signatures"]) - 1)
            return (
                extra + sum(nodes for nodes, _edges in factor_costs),
                extra + sum(edges for _nodes, edges in factor_costs),
            )

        if candidate["kind"] == "cut":
            terms = candidate["output_terms"][component_index]
            left_signature = candidate["left_signature"]
            right_signature = candidate["right_signature"]
            left_indices = sorted(
                {
                    candidate["left_layout"].flatten(left_index)
                    for left_index, _right_index, _coeff in terms
                }
            )
            right_indices = sorted(
                {
                    candidate["right_layout"].flatten(right_index)
                    for _left_index, right_index, _coeff in terms
                }
            )
            if left_signature in block_signatures:
                left_costs = [(0, 0) for _child_component_index in left_indices]
            else:
                left_costs = [
                    self.compact_projection_cost(left_signature, child_component_index)
                    for child_component_index in left_indices
                ]
            if right_signature in block_signatures:
                right_costs = [(0, 0) for _child_component_index in right_indices]
            else:
                right_costs = [
                    self.compact_projection_cost(right_signature, child_component_index)
                    for child_component_index in right_indices
                ]
            return (
                1
                + sum(nodes for nodes, _edges in left_costs)
                + sum(nodes for nodes, _edges in right_costs),
                len(terms)
                + sum(edges for _nodes, edges in left_costs)
                + sum(edges for _nodes, edges in right_costs),
            )

        raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

    def compact_projection_choice_with_overrides(
        self,
        signature,
        component_index,
        overrides,
    ):
        return overrides.get(
            (signature, component_index),
            self.compact_projection_choice(signature, component_index),
        )

    def projection_dependency_keys(self, signature, component_index):
        cache_key = (signature, component_index)
        if cache_key in self._projection_dependency_cache:
            return self._projection_dependency_cache[cache_key]

        leaves, open_counts, _flat_edges = signature
        deps = {cache_key}
        if len(leaves) == 1:
            self._projection_dependency_cache[cache_key] = deps
            return deps

        candidate = self.compact_projection_choice(signature, component_index)
        if candidate["kind"] == "legacy":
            terms = candidate["output_terms"][component_index]
            child_signature = candidate["child_signatures"][0]
            for child_index, _leaf_node, _coeff in terms:
                deps.update(self.projection_dependency_keys(child_signature, child_index))
        elif candidate["kind"] == "product":
            for factor_signature in candidate["factor_signatures"]:
                deps.update(self.projection_dependency_keys(factor_signature, 0))
        elif candidate["kind"] == "cut":
            terms = candidate["output_terms"][component_index]
            for left_index, right_index, _coeff in terms:
                deps.update(
                    self.projection_dependency_keys(
                        candidate["left_signature"],
                        candidate["left_layout"].flatten(left_index),
                    )
                )
                deps.update(
                    self.projection_dependency_keys(
                        candidate["right_signature"],
                        candidate["right_layout"].flatten(right_index),
                    )
                )
        else:
            raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

        self._projection_dependency_cache[cache_key] = deps
        return deps

    def compact_projection_local_weight(self, signature, component_index):
        candidate = self.compact_projection_choice(signature, component_index)
        if candidate is None:
            return 0
        if candidate["kind"] == "legacy":
            return 1 + len(candidate["output_terms"][component_index])
        if candidate["kind"] == "product":
            return max(0, len(candidate["factor_signatures"]) - 1) * 2
        if candidate["kind"] == "cut":
            return 1 + len(candidate["output_terms"][component_index])
        raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

    def projection_dependency_weights(self, signature, component_index):
        dep_keys, dep_items, _total = self.projection_dependency_summary(
            signature,
            component_index,
        )
        return dict(dep_items)

    def projection_dependency_summary(self, signature, component_index):
        cache_key = (signature, component_index)
        if cache_key in self._projection_dependency_summary_cache:
            return self._projection_dependency_summary_cache[cache_key]

        dep_keys = tuple(self.projection_dependency_keys(signature, component_index))
        dep_items = tuple(
            (dep_key, self.compact_projection_local_weight(*dep_key))
            for dep_key in dep_keys
        )
        total = sum(weight for _dep_key, weight in dep_items)
        summary = (dep_keys, dep_items, total)
        self._projection_dependency_summary_cache[cache_key] = summary
        self._projection_dependency_weight_cache[cache_key] = dict(dep_items)
        return summary

    def scalar_k_relabel_family_info(self, state):
        channel_map = {}
        canonical_to_actual = []
        next_channel = 0
        canonical_leaves = []
        for mu, rank in state.leaves:
            actual_channel = mu // (self.args.l + 1)
            if actual_channel not in channel_map:
                channel_map[actual_channel] = next_channel
                canonical_to_actual.append(actual_channel)
                next_channel += 1
            canonical_leaves.append((channel_map[actual_channel], rank))
        family_key = (
            state.degree,
            tuple(canonical_leaves),
            state.open_counts,
            flatten_edges(state.edges),
        )
        return family_key, tuple(canonical_to_actual)

    def relabel_projection_key(self, projection_key, channel_remap):
        signature, component_index = projection_key
        leaves, open_counts, _flat_edges = signature
        edges = self.signature_edges(signature)
        relabeled_leaves = []
        for mu, rank in leaves:
            actual_channel = mu // (self.args.l + 1)
            new_channel = channel_remap.get(actual_channel, actual_channel)
            relabeled_leaves.append(((self.args.l + 1) * new_channel + rank, rank))

        (
            canonical_leaves,
            canonical_open_counts,
            canonical_edges,
            canonical_order,
        ) = self.canonicalize_structure(
            tuple(relabeled_leaves),
            open_counts,
            edges,
        )
        old_layout = block_layout(open_counts)
        old_index_tuple = old_layout.index_tuples[component_index]
        new_index_tuple = tuple(old_index_tuple[idx] for idx in canonical_order)
        new_layout = block_layout(canonical_open_counts)
        new_component_index = new_layout.flatten(new_index_tuple)
        return (
            (
                canonical_leaves,
                canonical_open_counts,
                flatten_edges(canonical_edges),
            ),
            new_component_index,
        )

    @staticmethod
    def normalized_linear_combo_key(terms):
        normalized = []
        for src0, src1, coeff in terms:
            if src0 <= src1:
                normalized.append((src0, src1, coeff))
            else:
                normalized.append((src1, src0, coeff))
        return tuple(sorted(normalized))

    def collect_projection_requests(self, signature, component_index, requests):
        requests[signature][component_index] += 1
        leaves, _open_counts, _flat_edges = signature
        if len(leaves) == 1:
            return

        candidate = self.compact_projection_choice(signature, component_index)
        if candidate["kind"] == "legacy":
            child_signature = candidate["child_signatures"][0]
            for child_index, _leaf_node, _coeff in candidate["output_terms"][component_index]:
                self.collect_projection_requests(child_signature, child_index, requests)
        elif candidate["kind"] == "product":
            for factor_signature in candidate["factor_signatures"]:
                self.collect_projection_requests(factor_signature, 0, requests)
        elif candidate["kind"] == "cut":
            for left_index, right_index, _coeff in candidate["output_terms"][component_index]:
                self.collect_projection_requests(
                    candidate["left_signature"],
                    candidate["left_layout"].flatten(left_index),
                    requests,
                )
                self.collect_projection_requests(
                    candidate["right_signature"],
                    candidate["right_layout"].flatten(right_index),
                    requests,
                )
        else:
            raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

    def collect_projection_requests_with_overrides(
        self,
        signature,
        component_index,
        requests,
        overrides,
    ):
        requests[signature][component_index] += 1
        leaves, _open_counts, _flat_edges = signature
        if len(leaves) == 1:
            return

        candidate = self.compact_projection_choice_with_overrides(
            signature,
            component_index,
            overrides,
        )
        if candidate["kind"] == "legacy":
            child_signature = candidate["child_signatures"][0]
            for child_index, _leaf_node, _coeff in candidate["output_terms"][
                component_index
            ]:
                self.collect_projection_requests_with_overrides(
                    child_signature,
                    child_index,
                    requests,
                    overrides,
                )
        elif candidate["kind"] == "product":
            for factor_signature in candidate["factor_signatures"]:
                self.collect_projection_requests_with_overrides(
                    factor_signature,
                    0,
                    requests,
                    overrides,
                )
        elif candidate["kind"] == "cut":
            for left_index, right_index, _coeff in candidate["output_terms"][
                component_index
            ]:
                self.collect_projection_requests_with_overrides(
                    candidate["left_signature"],
                    candidate["left_layout"].flatten(left_index),
                    requests,
                    overrides,
                )
                self.collect_projection_requests_with_overrides(
                    candidate["right_signature"],
                    candidate["right_layout"].flatten(right_index),
                    requests,
                    overrides,
                )
        else:
            raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

    def whole_state_projection_cutover(self, signature, requested_components):
        if len(signature[0]) < 2:
            return False
        distinct_components = len(requested_components)
        total_requests = len(requested_components)
        total_size = block_layout(signature[1]).total_size
        if distinct_components <= 1:
            return False
        if distinct_components >= min(6, total_size):
            return True
        if distinct_components >= max(3, total_size // 3):
            return True
        whole_cost = self.compact_state_cost(signature)
        projection_cost = (
            sum(self.compact_projection_cost(signature, component_index)[0] for component_index in requested_components),
            sum(self.compact_projection_cost(signature, component_index)[1] for component_index in requested_components),
        )
        return self.graph_cost_key(*whole_cost) <= self.graph_cost_key(*projection_cost)

    def whole_state_projection_cutover_with_cost_key(
        self,
        signature,
        requested_components,
        cost_key,
    ):
        if len(signature[0]) < 2:
            return False
        distinct_components = len(requested_components)
        total_size = block_layout(signature[1]).total_size
        if distinct_components <= 1:
            return False
        if distinct_components >= min(6, total_size):
            return True
        if distinct_components >= max(3, total_size // 3):
            return True
        whole_cost = self.compact_state_cost(signature)
        projection_cost = (
            sum(
                self.compact_projection_cost(signature, component_index)[0]
                for component_index in requested_components
            ),
            sum(
                self.compact_projection_cost(signature, component_index)[1]
                for component_index in requested_components
            ),
        )
        return cost_key(*whole_cost) <= cost_key(*projection_cost)

    def compact_state_cost(self, signature):
        if signature in self._compact_state_cost_cache:
            return self._compact_state_cost_cache[signature]

        leaves, open_counts, _flat_edges = signature
        if len(leaves) == 1:
            self._compact_state_cost_cache[signature] = (0, 0)
            self._compact_state_choice_cache[signature] = None
            return self._compact_state_cost_cache[signature]

        best_cost = None
        best_choice = None

        for candidate in self.state_candidates(signature):
            if candidate["kind"] == "legacy":
                child_nodes, child_edges = self.compact_state_cost(
                    candidate["child_signatures"][0]
                )
                total_cost = (
                    child_nodes + candidate["node_count"],
                    child_edges + candidate["edge_count"],
                )
            elif candidate["kind"] == "product":
                factor_nodes = 0
                factor_edges = 0
                for factor_signature in candidate["factor_signatures"]:
                    child_nodes, child_edges = self.compact_state_cost(factor_signature)
                    factor_nodes += child_nodes
                    factor_edges += child_edges
                total_cost = (
                    factor_nodes + candidate["node_count"],
                    factor_edges + candidate["edge_count"],
                )
            elif candidate["kind"] == "cut":
                left_nodes, left_edges = self.compact_state_cost(
                    candidate["left_signature"]
                )
                right_nodes, right_edges = self.compact_state_cost(
                    candidate["right_signature"]
                )
                total_cost = (
                    left_nodes + right_nodes + candidate["node_count"],
                    left_edges + right_edges + candidate["edge_count"],
                )
            else:
                raise RuntimeError(f"unknown candidate kind {candidate['kind']}")

            if best_cost is None or self.graph_cost_key(*total_cost) < self.graph_cost_key(
                *best_cost
            ):
                best_cost = total_cost
                best_choice = candidate

        if best_cost is None:
            raise RuntimeError(f"failed to build compact plan for state {signature}")

        self._compact_state_cost_cache[signature] = best_cost
        self._compact_state_choice_cache[signature] = best_choice
        return best_cost

    def compact_state_choice(self, signature):
        self.compact_state_cost(signature)
        return self._compact_state_choice_cache[signature]

    def compact_partition_basis(self):
        if self._compact_basis_cache is not None:
            return self._compact_basis_cache

        emitted_basic = list(self.alpha_index_basic)
        emitted_times = []
        next_id = len(emitted_basic)
        emitted_projections = {}
        emitted_states = {}
        product_cache = {}
        linear_combo_cache = {}

        projection_requests = defaultdict(Counter)
        for state in self.selected_scalar_states():
            self.collect_projection_requests(state.signature, 0, projection_requests)

        full_state_signatures = {
            signature
            for signature, component_counts in projection_requests.items()
            if self.whole_state_projection_cutover(
                signature,
                tuple(sorted(component_counts)),
            )
        }

        def emit_scalar_product(factor_nodes):
            nonlocal next_id
            if len(factor_nodes) == 1:
                return factor_nodes[0]
            key = tuple(sorted(factor_nodes))
            if key in product_cache:
                return product_cache[key]
            split = len(key) // 2
            left = emit_scalar_product(key[:split])
            right = emit_scalar_product(key[split:])
            dst = next_id
            next_id += 1
            emitted_times.append((left, right, 1, dst))
            product_cache[key] = dst
            return dst

        def emit_linear_combo(terms):
            nonlocal next_id
            if len(terms) == 1:
                src0, src1, coeff = terms[0]
                if coeff == 1:
                    return emit_scalar_product((src0, src1))
            term_key = self.normalized_linear_combo_key(terms)
            if term_key not in linear_combo_cache:
                dst = next_id
                next_id += 1
                linear_combo_cache[term_key] = dst
                for src0, src1, coeff in terms:
                    emitted_times.append((src0, src1, coeff, dst))
            return linear_combo_cache[term_key]

        def emit_state(signature):
            nonlocal next_id
            if signature in emitted_states:
                return emitted_states[signature]

            leaves, open_counts, _flat_edges = signature
            if len(leaves) == 1:
                leaf = leaves[0]
                rank = open_counts[0]
                components = tuple(
                    self.basic_node_ids[(leaf, exp_idx)]
                    for exp_idx in range(len(exponent_tuples(rank)))
                )
                emitted_states[signature] = components
                return components

            candidate = self.compact_state_choice(signature)
            if candidate["kind"] == "legacy":
                child_signature = candidate["child_signatures"][0]
                child_components = (
                    emit_state(child_signature)
                    if child_signature in full_state_signatures
                    else {
                        child_index: emit_projection(child_signature, child_index)
                        for child_index in sorted(
                            {
                                child_index
                                for terms in candidate["output_terms"]
                                for child_index, _leaf_node, _coeff in terms
                            }
                        )
                    }
                )
                components = []
                for terms in candidate["output_terms"]:
                    dst = next_id
                    next_id += 1
                    components.append(dst)
                    for child_index, leaf_node, coeff in terms:
                        src0 = (
                            child_components[child_index]
                            if isinstance(child_components, tuple)
                            else child_components[child_index]
                        )
                        emitted_times.append((src0, leaf_node, coeff, dst))
                emitted_states[signature] = tuple(components)
                return emitted_states[signature]

            if candidate["kind"] == "product":
                factor_nodes = [
                    emit_projection(factor_signature, 0)
                    for factor_signature in candidate["factor_signatures"]
                ]
                emitted_states[signature] = (emit_scalar_product(tuple(factor_nodes)),)
                return emitted_states[signature]

            left_components = (
                emit_state(candidate["left_signature"])
                if candidate["left_signature"] in full_state_signatures
                else {
                    candidate["left_layout"].flatten(left_index): emit_projection(
                        candidate["left_signature"],
                        candidate["left_layout"].flatten(left_index),
                    )
                    for terms in candidate["output_terms"]
                    for left_index, _right_index, _coeff in terms
                }
            )
            right_components = (
                emit_state(candidate["right_signature"])
                if candidate["right_signature"] in full_state_signatures
                else {
                    candidate["right_layout"].flatten(right_index): emit_projection(
                        candidate["right_signature"],
                        candidate["right_layout"].flatten(right_index),
                    )
                    for terms in candidate["output_terms"]
                    for _left_index, right_index, _coeff in terms
                }
            )

            components = []
            for terms in candidate["output_terms"]:
                dst = next_id
                next_id += 1
                components.append(dst)
                for left_index, right_index, coeff in terms:
                    left_component_index = candidate["left_layout"].flatten(left_index)
                    right_component_index = candidate["right_layout"].flatten(right_index)
                    src0 = (
                        left_components[left_component_index]
                        if isinstance(left_components, tuple)
                        else left_components[left_component_index]
                    )
                    src1 = (
                        right_components[right_component_index]
                        if isinstance(right_components, tuple)
                        else right_components[right_component_index]
                    )
                    emitted_times.append((src0, src1, coeff, dst))

            emitted_states[signature] = tuple(components)
            return emitted_states[signature]

        def emit_projection(signature, component_index):
            nonlocal next_id
            cache_key = (signature, component_index)
            if cache_key in emitted_projections:
                return emitted_projections[cache_key]
            leaves, open_counts, _flat_edges = signature
            if len(leaves) == 1:
                leaf = leaves[0]
                emitted_projections[cache_key] = self.basic_node_ids[(leaf, component_index)]
                return emitted_projections[cache_key]
            if signature in full_state_signatures:
                emitted_projections[cache_key] = emit_state(signature)[component_index]
                return emitted_projections[cache_key]

            candidate = self.compact_projection_choice(signature, component_index)
            if candidate["kind"] == "legacy":
                terms = candidate["output_terms"][component_index]
                normalized_terms = []
                child_signature = candidate["child_signatures"][0]
                for child_index, leaf_node, coeff in terms:
                    src0 = emit_projection(child_signature, child_index)
                    normalized_terms.append((src0, leaf_node, coeff))
                emitted_projections[cache_key] = emit_linear_combo(normalized_terms)
                return emitted_projections[cache_key]

            if candidate["kind"] == "product":
                factor_nodes = [
                    emit_projection(factor_signature, 0)
                    for factor_signature in candidate["factor_signatures"]
                ]
                emitted_projections[cache_key] = emit_scalar_product(tuple(factor_nodes))
                return emitted_projections[cache_key]

            terms = candidate["output_terms"][component_index]
            normalized_terms = []
            for left_index, right_index, coeff in terms:
                src0 = emit_projection(candidate["left_signature"], candidate["left_layout"].flatten(left_index))
                src1 = emit_projection(candidate["right_signature"], candidate["right_layout"].flatten(right_index))
                normalized_terms.append((src0, src1, coeff))
            emitted_projections[cache_key] = emit_linear_combo(normalized_terms)
            return emitted_projections[cache_key]

        mapping = []
        for state in self.selected_scalar_states():
            mapping.append(emit_projection(state.signature, 0))

        self._compact_basis_cache = (
            emitted_basic,
            emitted_times,
            mapping,
            next_id,
        )
        return self._compact_basis_cache

    def best_staged_family_overrides(self, preemitted_signatures):
        overrides = {}
        lift_signatures = self.degree3_block_lift_signatures()
        for state in self.selected_scalar_states():
            cache_key = (state.signature, 0)
            chosen = self.compact_projection_choice(*cache_key)
            if chosen is None:
                continue
            best_candidate = chosen
            best_key = self.edge_first_cost_key(
                *self.projection_candidate_total_cost(chosen, 0)
            )

            for candidate in self.projection_candidates(*cache_key):
                if candidate == chosen:
                    continue

                if state.degree == 4:
                    if (
                        candidate["kind"] == "cut"
                        and len(candidate["left_signature"][0]) == 2
                        and len(candidate["right_signature"][0]) == 2
                    ):
                        if (
                            candidate["left_signature"] in preemitted_signatures
                            and candidate["right_signature"] in preemitted_signatures
                        ):
                            candidate_key = self.edge_first_cost_key(
                                *self.staged_projection_candidate_total_cost(
                                    candidate,
                                    0,
                                    preemitted_signatures,
                                )
                            )
                        else:
                            candidate_key = self.edge_first_cost_key(
                                *self.projection_candidate_total_cost(candidate, 0)
                            )
                    elif candidate["kind"] == "cut":
                        left_leaf_count = len(candidate["left_signature"][0])
                        right_leaf_count = len(candidate["right_signature"][0])
                        if tuple(sorted((left_leaf_count, right_leaf_count))) != (1, 3):
                            continue
                        lift_signature = (
                            candidate["left_signature"]
                            if left_leaf_count == 3
                            else candidate["right_signature"]
                        )
                        if lift_signature not in lift_signatures:
                            continue
                        candidate_key = self.edge_first_cost_key(
                            *self.staged_projection_candidate_total_cost(
                                candidate,
                                0,
                                preemitted_signatures,
                            )
                        )
                    else:
                        continue
                elif state.degree == 3:
                    preemitted_signature = None
                    if candidate["kind"] == "cut":
                        left_leaf_count = len(candidate["left_signature"][0])
                        right_leaf_count = len(candidate["right_signature"][0])
                        if tuple(sorted((left_leaf_count, right_leaf_count))) == (1, 2):
                            preemitted_signature = (
                                candidate["left_signature"]
                                if left_leaf_count == 2
                                else candidate["right_signature"]
                            )
                    if preemitted_signature not in preemitted_signatures:
                        continue
                    candidate_key = self.edge_first_cost_key(
                        *self.staged_projection_candidate_total_cost(
                            candidate,
                            0,
                            preemitted_signatures,
                        )
                    )
                else:
                    continue

                if candidate_key < best_key:
                    best_candidate = candidate
                    best_key = candidate_key

            if best_candidate != chosen:
                overrides[cache_key] = best_candidate
        return overrides

    def legacy_staged_family_overrides(self):
        overrides = {}
        for state in self.selected_scalar_states():
            cache_key = (state.signature, 0)
            chosen = self.compact_projection_choice(*cache_key)
            if chosen is None or state.degree != 4 or chosen["kind"] != "legacy":
                continue

            candidates = [
                candidate
                for candidate in self.projection_candidates(*cache_key)
                if candidate["kind"] == "cut"
                and len(candidate["left_signature"][0]) == 2
                and len(candidate["right_signature"][0]) == 2
            ]
            if not candidates:
                continue

            best_candidate = min(
                candidates,
                key=lambda item: self.edge_first_cost_key(
                    *self.projection_candidate_total_cost(item, 0)
                ),
            )
            overrides[cache_key] = best_candidate
        return overrides

    def scalar_gram_staged_overrides(self, preemitted_signatures):
        overrides = {}
        if not preemitted_signatures:
            return overrides

        for state in self.selected_scalar_states():
            cache_key = (state.signature, 0)
            chosen = self.compact_projection_choice(*cache_key)
            if chosen is None:
                continue

            best_candidate = chosen
            best_key = self.edge_first_cost_key(
                *self.projection_candidate_total_cost(chosen, 0)
            )

            for candidate in self.projection_candidates(*cache_key):
                if candidate == chosen:
                    continue
                candidate_key = self.edge_first_cost_key(
                    *self.staged_projection_candidate_total_cost(
                        candidate,
                        0,
                        preemitted_signatures,
                    )
                )
                if candidate_key < best_key:
                    best_candidate = candidate
                    best_key = candidate_key

            if best_candidate != chosen:
                overrides[cache_key] = best_candidate
        return overrides

    def staged_family_partition_basis(self, overrides, preemitted_signatures):
        emitted_basic = list(self.alpha_index_basic)
        emitted_times = []
        next_id = len(emitted_basic)
        emitted_projections = {}
        emitted_states = {}
        product_cache = {}
        linear_combo_cache = {}
        emitted_gram_blocks = {}
        emitted_scalar_block_lifts = {}

        projection_requests = defaultdict(Counter)
        for state in self.selected_scalar_states():
            self.collect_projection_requests_with_overrides(
                state.signature,
                0,
                projection_requests,
                overrides,
            )

        requested_block_signatures = {
            signature
            for signature in projection_requests
            if signature in preemitted_signatures
        }
        full_state_signatures = {
            signature
            for signature, component_counts in projection_requests.items()
            if self.whole_state_projection_cutover_with_cost_key(
                signature,
                tuple(sorted(component_counts)),
                self.edge_first_cost_key,
            )
        }
        full_state_signatures.update(requested_block_signatures)

        scalar_signature_map = self.same_rank_degree2_scalar_signature_map()

        def emit_same_rank_gram_block(rank, primitive_leaf_group):
            nonlocal next_id
            cache_key = (rank, tuple(leaf[0] for leaf in primitive_leaf_group))
            if cache_key in emitted_gram_blocks:
                return emitted_gram_blocks[cache_key]

            basis_terms = rank_gram_block_basis(rank)
            pair_nodes = {}
            for left_index, left_leaf in enumerate(primitive_leaf_group):
                for right_leaf in primitive_leaf_group[left_index:]:
                    dst = next_id
                    next_id += 1
                    pair_nodes[(left_leaf, right_leaf)] = dst
                    for exp_index, coeff in basis_terms:
                        src0 = self.basic_node_ids[(left_leaf, exp_index)]
                        src1 = self.basic_node_ids[(right_leaf, exp_index)]
                        emitted_times.append((src0, src1, coeff, dst))
                    signature = scalar_signature_map.get(rank, {}).get(
                        (left_leaf, right_leaf)
                    )
                    if signature is not None:
                        emitted_projections[(signature, 0)] = dst
                        emitted_states[signature] = (dst,)

            emitted_gram_blocks[cache_key] = pair_nodes
            return pair_nodes

        def emit_exact_scalar_block_lift(signature):
            nonlocal next_id
            if signature in emitted_scalar_block_lifts:
                return emitted_scalar_block_lifts[signature]

            parts = self.exact_scalar_block_lift_parts(signature)
            if parts is None:
                raise RuntimeError(f"{signature} is not an exact scalar block lift")
            scalar_leaf, degree2_signature = parts
            scalar_node = self.basic_node_ids[(scalar_leaf, 0)]
            block_node = emit_projection(degree2_signature, 0)
            dst = emit_scalar_product((scalar_node, block_node))
            emitted_projections[(signature, 0)] = dst
            emitted_states[signature] = (dst,)
            emitted_scalar_block_lifts[signature] = dst
            return dst

        def emit_degree3_block_lift_state(signature):
            if signature in emitted_states:
                return emitted_states[signature]

            legacy = self.legacy_candidate(signature)
            if (
                legacy is None
                or legacy["child_signatures"][0] not in preemitted_signatures
            ):
                return emit_state(signature)

            child_signature = legacy["child_signatures"][0]
            child_components = (
                emit_state(child_signature)
                if child_signature in full_state_signatures
                else {
                    child_index: emit_projection(child_signature, child_index)
                    for child_index in sorted(
                        {
                            child_index
                            for terms in legacy["output_terms"]
                            for child_index, _leaf_node, _coeff in terms
                        }
                    )
                }
            )
            components = []
            nonlocal next_id
            for terms in legacy["output_terms"]:
                dst = next_id
                next_id += 1
                components.append(dst)
                for child_index, leaf_node, coeff in terms:
                    src0 = (
                        child_components[child_index]
                        if isinstance(child_components, tuple)
                        else child_components[child_index]
                    )
                    emitted_times.append((src0, leaf_node, coeff, dst))
            emitted_states[signature] = tuple(components)
            return emitted_states[signature]

        def emit_scalar_product(factor_nodes):
            nonlocal next_id
            if len(factor_nodes) == 1:
                return factor_nodes[0]
            key = tuple(sorted(factor_nodes))
            if key in product_cache:
                return product_cache[key]
            split = len(key) // 2
            left = emit_scalar_product(key[:split])
            right = emit_scalar_product(key[split:])
            dst = next_id
            next_id += 1
            emitted_times.append((left, right, 1, dst))
            product_cache[key] = dst
            return dst

        def emit_linear_combo(terms):
            nonlocal next_id
            if len(terms) == 1:
                src0, src1, coeff = terms[0]
                if coeff == 1:
                    return emit_scalar_product((src0, src1))
            term_key = self.normalized_linear_combo_key(terms)
            if term_key not in linear_combo_cache:
                dst = next_id
                next_id += 1
                linear_combo_cache[term_key] = dst
                for src0, src1, coeff in terms:
                    emitted_times.append((src0, src1, coeff, dst))
            return linear_combo_cache[term_key]

        def emit_state(signature):
            nonlocal next_id
            if signature in emitted_states:
                return emitted_states[signature]

            leaves, open_counts, _flat_edges = signature
            if len(leaves) == 1:
                leaf = leaves[0]
                rank = open_counts[0]
                components = tuple(
                    self.basic_node_ids[(leaf, exp_idx)]
                    for exp_idx in range(len(exponent_tuples(rank)))
                )
                emitted_states[signature] = components
                return components

            candidate = self.compact_state_choice(signature)
            if candidate["kind"] == "legacy":
                child_signature = candidate["child_signatures"][0]
                child_components = (
                    emit_state(child_signature)
                    if child_signature in full_state_signatures
                    else {
                        child_index: emit_projection(child_signature, child_index)
                        for child_index in sorted(
                            {
                                child_index
                                for terms in candidate["output_terms"]
                                for child_index, _leaf_node, _coeff in terms
                            }
                        )
                    }
                )
                components = []
                for terms in candidate["output_terms"]:
                    dst = next_id
                    next_id += 1
                    components.append(dst)
                    for child_index, leaf_node, coeff in terms:
                        src0 = (
                            child_components[child_index]
                            if isinstance(child_components, tuple)
                            else child_components[child_index]
                        )
                        emitted_times.append((src0, leaf_node, coeff, dst))
                emitted_states[signature] = tuple(components)
                return emitted_states[signature]

            if candidate["kind"] == "product":
                factor_nodes = [
                    emit_projection(factor_signature, 0)
                    for factor_signature in candidate["factor_signatures"]
                ]
                emitted_states[signature] = (emit_scalar_product(tuple(factor_nodes)),)
                return emitted_states[signature]

            left_components = (
                emit_state(candidate["left_signature"])
                if candidate["left_signature"] in full_state_signatures
                else {
                    candidate["left_layout"].flatten(left_index): emit_projection(
                        candidate["left_signature"],
                        candidate["left_layout"].flatten(left_index),
                    )
                    for terms in candidate["output_terms"]
                    for left_index, _right_index, _coeff in terms
                }
            )
            right_components = (
                emit_state(candidate["right_signature"])
                if candidate["right_signature"] in full_state_signatures
                else {
                    candidate["right_layout"].flatten(right_index): emit_projection(
                        candidate["right_signature"],
                        candidate["right_layout"].flatten(right_index),
                    )
                    for terms in candidate["output_terms"]
                    for _left_index, right_index, _coeff in terms
                }
            )

            components = []
            for terms in candidate["output_terms"]:
                dst = next_id
                next_id += 1
                components.append(dst)
                for left_index, right_index, coeff in terms:
                    left_component_index = candidate["left_layout"].flatten(left_index)
                    right_component_index = candidate["right_layout"].flatten(right_index)
                    src0 = (
                        left_components[left_component_index]
                        if isinstance(left_components, tuple)
                        else left_components[left_component_index]
                    )
                    src1 = (
                        right_components[right_component_index]
                        if isinstance(right_components, tuple)
                        else right_components[right_component_index]
                    )
                    emitted_times.append((src0, src1, coeff, dst))

            emitted_states[signature] = tuple(components)
            return emitted_states[signature]

        def emit_projection(signature, component_index):
            nonlocal next_id
            cache_key = (signature, component_index)
            if cache_key in emitted_projections:
                return emitted_projections[cache_key]
            leaves, open_counts, _flat_edges = signature
            if len(leaves) == 1:
                leaf = leaves[0]
                emitted_projections[cache_key] = self.basic_node_ids[
                    (leaf, component_index)
                ]
                return emitted_projections[cache_key]
            if signature in requested_block_signatures:
                if self.exact_scalar_block_lift_parts(signature) is not None:
                    return emit_exact_scalar_block_lift(signature)
                if self.degree3_block_lift_parts(signature) is not None:
                    emitted_projections[cache_key] = emit_degree3_block_lift_state(
                        signature
                    )[component_index]
                    return emitted_projections[cache_key]
            gram_rank = self.same_rank_degree2_block_rank(signature)
            if (
                gram_rank is not None
                and open_counts == (0, 0)
                and signature in requested_block_signatures
            ):
                primitive_leaf_group = tuple(
                    leaf for leaf in self.primitive_leaves if leaf[1] == gram_rank
                )
                emit_same_rank_gram_block(gram_rank, primitive_leaf_group)
                if cache_key in emitted_projections:
                    return emitted_projections[cache_key]
            if signature in full_state_signatures:
                emitted_projections[cache_key] = emit_state(signature)[component_index]
                return emitted_projections[cache_key]

            candidate = self.compact_projection_choice_with_overrides(
                signature,
                component_index,
                overrides,
            )
            if candidate["kind"] == "legacy":
                terms = candidate["output_terms"][component_index]
                normalized_terms = []
                child_signature = candidate["child_signatures"][0]
                for child_index, leaf_node, coeff in terms:
                    src0 = emit_projection(child_signature, child_index)
                    normalized_terms.append((src0, leaf_node, coeff))
                emitted_projections[cache_key] = emit_linear_combo(normalized_terms)
                return emitted_projections[cache_key]

            if candidate["kind"] == "product":
                factor_nodes = [
                    emit_projection(factor_signature, 0)
                    for factor_signature in candidate["factor_signatures"]
                ]
                emitted_projections[cache_key] = emit_scalar_product(tuple(factor_nodes))
                return emitted_projections[cache_key]

            terms = candidate["output_terms"][component_index]
            normalized_terms = []
            for left_index, right_index, coeff in terms:
                src0 = emit_projection(
                    candidate["left_signature"],
                    candidate["left_layout"].flatten(left_index),
                )
                src1 = emit_projection(
                    candidate["right_signature"],
                    candidate["right_layout"].flatten(right_index),
                )
                normalized_terms.append((src0, src1, coeff))
            emitted_projections[cache_key] = emit_linear_combo(normalized_terms)
            return emitted_projections[cache_key]

        mapping = []
        for state in self.selected_scalar_states():
            mapping.append(emit_projection(state.signature, 0))

        return emitted_basic, emitted_times, mapping, next_id

    def staged_family_basis(self):
        if self._staged_family_basis_cache is not None:
            return self._staged_family_basis_cache

        baseline_basis = self.basis_for_mode("degree-aware")
        candidate_bases = []

        legacy_overrides = self.legacy_staged_family_overrides()
        legacy_basis = self.staged_family_partition_basis(legacy_overrides, set())
        legacy_basis = self.dedup_emitted_basis_fixed_point(
            *topologically_renumber(*legacy_basis)
        )
        legacy_basis = self.dedup_k_relabel_scalar_outputs(*legacy_basis)
        legacy_basis = topologically_renumber(*legacy_basis)
        candidate_bases.append(legacy_basis)

        gram_scalar_signatures = self.same_rank_degree2_scalar_signatures()
        gram_overrides = self.scalar_gram_staged_overrides(gram_scalar_signatures)
        gram_basis = self.staged_family_partition_basis(
            gram_overrides,
            gram_scalar_signatures,
        )
        gram_basis = self.dedup_emitted_basis_fixed_point(
            *topologically_renumber(*gram_basis)
        )
        gram_basis = self.dedup_k_relabel_scalar_outputs(*gram_basis)
        gram_basis = topologically_renumber(*gram_basis)
        candidate_bases.append(gram_basis)

        preemitted_signatures = self.degree2_block_signatures()
        preemitted_signatures.update(self.degree3_block_lift_signatures())
        preemitted_signatures.update(self.exact_scalar_block_lift_signatures())
        overrides = self.best_staged_family_overrides(preemitted_signatures)
        candidate_basis = self.staged_family_partition_basis(
            overrides,
            preemitted_signatures,
        )
        candidate_basis = self.dedup_emitted_basis_fixed_point(
            *topologically_renumber(*candidate_basis)
        )
        candidate_basis = self.dedup_k_relabel_scalar_outputs(*candidate_basis)
        candidate_basis = topologically_renumber(*candidate_basis)
        candidate_bases.append(candidate_basis)

        baseline_fp_set = self.basis_fingerprint_set(*baseline_basis)
        matching_candidates = [
            basis
            for basis in candidate_bases
            if self.basis_fingerprint_set(*basis) == baseline_fp_set
        ]
        self._staged_family_matches_cache = bool(matching_candidates)
        if not matching_candidates:
            self._staged_family_is_better_cache = False
            self._staged_family_basis_cache = baseline_basis
            return self._staged_family_basis_cache

        candidate_basis = min(
            matching_candidates,
            key=lambda basis: self.edge_first_cost_key(basis[3], len(basis[1])),
        )
        self._staged_family_is_better_cache = (
            len(candidate_basis[1]) < len(baseline_basis[1])
            and candidate_basis[3] <= baseline_basis[3]
        )
        self._staged_family_basis_cache = candidate_basis
        return self._staged_family_basis_cache

    def pure_cut_state_cost(self, signature):
        if signature in self._pure_cut_state_cost_cache:
            return self._pure_cut_state_cost_cache[signature]

        leaves, open_counts, _flat_edges = signature
        if len(leaves) == 1:
            self._pure_cut_state_cost_cache[signature] = (0, 0)
            self._pure_cut_state_choice_cache[signature] = None
            return self._pure_cut_state_cost_cache[signature]

        parent_nodes = block_layout(open_counts).total_size
        best_cost = None
        best_choice = None

        for candidate in self.partition_candidates(signature, None):
            left_nodes, left_edges = self.pure_cut_state_cost(
                candidate["left_signature"]
            )
            right_nodes, right_edges = self.pure_cut_state_cost(
                candidate["right_signature"]
            )
            total_cost = (
                left_nodes + right_nodes + parent_nodes,
                left_edges + right_edges + candidate["edge_count"],
            )
            if best_cost is None or self.graph_cost_key(*total_cost) < self.graph_cost_key(
                *best_cost
            ):
                best_cost = total_cost
                best_choice = candidate

        if best_cost is None:
            raise RuntimeError(f"failed to build pure-cut plan for state {signature}")

        self._pure_cut_state_cost_cache[signature] = best_cost
        self._pure_cut_state_choice_cache[signature] = best_choice
        return best_cost

    def pure_cut_state_choice(self, signature):
        self.pure_cut_state_cost(signature)
        return self._pure_cut_state_choice_cache[signature]

    def pure_cut_basis(self):
        if self._pure_cut_basis_cache is not None:
            return self._pure_cut_basis_cache

        emitted_basic = list(self.alpha_index_basic)
        emitted_times = []
        next_id = len(emitted_basic)
        emitted_states = {}

        def emit_state(signature):
            nonlocal next_id
            if signature in emitted_states:
                return emitted_states[signature]

            leaves, open_counts, _flat_edges = signature
            if len(leaves) == 1:
                leaf = leaves[0]
                rank = open_counts[0]
                components = tuple(
                    self.basic_node_ids[(leaf, exp_idx)]
                    for exp_idx in range(len(exponent_tuples(rank)))
                )
                emitted_states[signature] = components
                return components

            candidate = self.pure_cut_state_choice(signature)
            left_components = emit_state(candidate["left_signature"])
            right_components = emit_state(candidate["right_signature"])
            left_layout = candidate["left_layout"]
            right_layout = candidate["right_layout"]

            components = []
            for terms in candidate["output_terms"]:
                dst = next_id
                next_id += 1
                components.append(dst)
                for left_index, right_index, coeff in terms:
                    src0 = left_components[left_layout.flatten(left_index)]
                    src1 = right_components[right_layout.flatten(right_index)]
                    emitted_times.append((src0, src1, coeff, dst))

            emitted_states[signature] = tuple(components)
            return emitted_states[signature]

        mapping = []
        for state in self.selected_scalar_states():
            mapping.append(emit_state(state.signature)[0])

        self._pure_cut_basis_cache = (
            emitted_basic,
            emitted_times,
            mapping,
            next_id,
        )
        return self._pure_cut_basis_cache

    def non_family_best_basis(self):
        candidates = {}
        for candidate_mode in (
            "legacy-dedup",
            "compact-cut",
            "degree-aware",
            "direct-compiled",
        ):
            candidates[candidate_mode] = self.basis_for_mode(candidate_mode)
        _resolved_mode, basis = min(
            candidates.items(),
            key=lambda item: self.graph_cost_key(item[1][3], len(item[1][1])),
        )
        return basis

    def family_dag_basis(self):
        if self._family_basis_cache is not None:
            return self._family_basis_cache

        baseline_basic, baseline_times, baseline_mapping, baseline_node_count = self.basis_for_mode(
            "legacy-dedup"
        )
        try:
            import generate_lk_from_template as lktemplate
        except ModuleNotFoundError:
            self._family_basis_cache = (
                list(baseline_basic),
                list(baseline_times),
                list(baseline_mapping),
                baseline_node_count,
            )
            return self._family_basis_cache
        raw_template = {
            "potential_name": self.args.potential_name
            or f"sus2mlip_l{self.args.l}k{self.args.k}",
            "scaling": str(self.args.scaling),
            "L": self.args.l,
            "scaling_map": "LK",
            "species_count": self.args.species_count,
            "radial_basis_type": self.args.radial_basis_type,
            "min_dist": str(self.args.min_dist),
            "max_dist": str(self.args.max_dist),
            "radial_basis_size": str(self.args.radial_basis_size),
            "radial_funcs_count": self.args.k * (self.args.l + 1),
            "alpha_moments_count": self.next_node_id,
            "alpha_index_basic": list(self.alpha_index_basic),
            "alpha_index_times": list(self.alpha_index_times),
            "alpha_moment_mapping": self.selected_scalar_node_ids(),
        }
        family_template = lktemplate.build_family_template(raw_template)
        family_basic, family_times, family_mapping, family_node_count = (
            lktemplate.instantiate_template(raw_template, family_template, self.args.k)
        )
        family_basic = list(family_basic)
        family_times = list(family_times)
        family_mapping = list(family_mapping)
        if (
            len(family_basic) == len(self.alpha_index_basic)
            and set(family_basic) == set(self.alpha_index_basic)
        ):
            basic_position = {entry: idx for idx, entry in enumerate(family_basic)}
            basic_remap = {
                old_idx: basic_position[entry]
                for old_idx, entry in enumerate(self.alpha_index_basic)
            }
            reordered_times = []
            for src0, src1, coeff, dst in family_times:
                new_src0 = basic_remap.get(src0, src0)
                new_src1 = basic_remap.get(src1, src1)
                reordered_times.append((new_src0, new_src1, coeff, dst))
            family_times = reordered_times
            family_mapping = [basic_remap.get(node, node) for node in family_mapping]
            family_basic = list(self.alpha_index_basic)
        source_fp_set = self.basis_fingerprint_set(
            baseline_basic,
            baseline_times,
            baseline_mapping,
            baseline_node_count,
        )
        family_fp_set = self.basis_fingerprint_set(
            family_basic,
            family_times,
            family_mapping,
            family_node_count,
        )
        if family_fp_set != source_fp_set:
            self._family_basis_cache = (
                list(baseline_basic),
                list(baseline_times),
                list(baseline_mapping),
                baseline_node_count,
            )
            return self._family_basis_cache
        self._family_basis_cache = (
            family_basic,
            family_times,
            family_mapping,
            family_node_count,
        )
        return self._family_basis_cache

    def selected_basis(self):
        return self.basis_for_mode(self.args.emit_mode)

    def basis_for_mode(self, mode):
        if mode in self._selected_basis_cache:
            return self._selected_basis_cache[mode]

        if mode == "legacy-dedup":
            basis = self.dedup_emitted_basis_fixed_point(
                *topologically_renumber(*self.dedup_pruned_basis())
            )
            basis = self.dedup_k_relabel_scalar_outputs(*basis)
            basis = topologically_renumber(*basis)
            resolved_mode = mode
        elif mode == "direct-compiled":
            basis = self.dedup_emitted_basis_fixed_point(
                *topologically_renumber(*self.compiled_basis())
            )
            basis = self.dedup_k_relabel_scalar_outputs(*basis)
            basis = topologically_renumber(*basis)
            resolved_mode = mode
        elif mode == "compact-cut":
            basis = self.dedup_emitted_basis_fixed_point(
                *topologically_renumber(*self.pure_cut_basis())
            )
            basis = self.dedup_k_relabel_scalar_outputs(*basis)
            basis = topologically_renumber(*basis)
            resolved_mode = mode
        elif mode == "degree-aware":
            basis = self.dedup_emitted_basis_fixed_point(
                *topologically_renumber(*self.compact_partition_basis())
            )
            basis = self.dedup_k_relabel_scalar_outputs(*basis)
            basis = topologically_renumber(*basis)
            resolved_mode = mode
        elif mode == "staged-family":
            basis = self.staged_family_basis()
            resolved_mode = mode
            if self._staged_family_matches_cache is False:
                resolved_mode = "degree-aware"
        elif mode == "family-dag":
            basis = self.dedup_emitted_basis_fixed_point(
                *topologically_renumber(*self.family_dag_basis())
            )
            basis = self.dedup_k_relabel_scalar_outputs(*basis)
            basis = topologically_renumber(*basis)
            resolved_mode = mode
        elif mode == "auto":
            candidates = {}
            for candidate_mode in (
                "legacy-dedup",
                "compact-cut",
                "degree-aware",
                "family-dag",
                "direct-compiled",
            ):
                candidates[candidate_mode] = self.basis_for_mode(candidate_mode)
            staged_basis = self.staged_family_basis()
            if self._staged_family_matches_cache and self._staged_family_is_better_cache:
                candidates["staged-family"] = staged_basis
            resolved_mode, basis = min(
                candidates.items(),
                key=lambda item: self.graph_cost_key(item[1][3], len(item[1][1])),
            )
        else:
            raise ValueError(f"unknown emit mode {mode}")

        self._selected_basis_cache[mode] = basis
        self._resolved_emit_mode_cache[mode] = resolved_mode
        return basis

    def resolved_emit_mode(self):
        self.basis_for_mode(self.args.emit_mode)
        return self._resolved_emit_mode_cache[self.args.emit_mode]

    def emitted_basis(self):
        return self.selected_basis()

    def raw_node_polynomials(self):
        if self._raw_node_poly_cache is not None:
            return self._raw_node_poly_cache

        by_dst = defaultdict(list)
        for src0, src1, coeff, dst in self.alpha_index_times:
            by_dst[dst].append((src0, src1, coeff))

        polys = [None] * self.next_node_id
        basic_tokens = self.basic_tokens(self.alpha_index_basic)
        labels = [None] * self.next_node_id

        for idx in range(len(self.alpha_index_basic)):
            polys[idx] = {(idx,): 1}
            labels[idx] = (basic_tokens[idx][0],)

        for dst in range(len(self.alpha_index_basic), self.next_node_id):
            poly = defaultdict(int)
            label_set = set()
            for src0, src1, coeff in by_dst.get(dst, []):
                label_set.update(labels[src0])
                label_set.update(labels[src1])
                for mono0, weight0 in polys[src0].items():
                    for mono1, weight1 in polys[src1].items():
                        mono = tuple(sorted(mono0 + mono1))
                        poly[mono] += coeff * weight0 * weight1
            polys[dst] = dict(poly)
            labels[dst] = tuple(sorted(label_set))

        self._raw_node_poly_cache = (polys, labels)
        return self._raw_node_poly_cache

    def basic_tokens(self, basic):
        tokens = []
        for mu, x, y, z in basic:
            tokens.append((mu // (self.args.l + 1), mu % (self.args.l + 1), (x, y, z)))
        return tokens

    def emitted_node_polynomials(self, basic, times, node_count):
        by_dst = defaultdict(list)
        for src0, src1, coeff, dst in times:
            by_dst[dst].append((src0, src1, coeff))

        polys = [None] * node_count
        labels = [None] * node_count
        basic_tokens = self.basic_tokens(basic)

        for idx in range(len(basic_tokens)):
            polys[idx] = {(idx,): 1}
            labels[idx] = (basic_tokens[idx][0],)

        for dst in range(len(basic_tokens), node_count):
            poly = defaultdict(int)
            label_set = set()
            for src0, src1, coeff in by_dst.get(dst, []):
                label_set.update(labels[src0])
                label_set.update(labels[src1])
                for mono0, weight0 in polys[src0].items():
                    for mono1, weight1 in polys[src1].items():
                        mono = tuple(sorted(mono0 + mono1))
                        poly[mono] += coeff * weight0 * weight1
            polys[dst] = dict(poly)
            labels[dst] = tuple(sorted(label_set))

        return polys, labels, basic_tokens

    def canonicalize_emitted_polynomial(self, poly, basic_tokens):
        labels = sorted({basic_tokens[idx][0] for mono in poly for idx in mono})
        label_map = dict((label, idx) for idx, label in enumerate(labels))
        items = []
        for mono, coeff in sorted(poly.items()):
            canon_mono = tuple(
                sorted(
                    (
                        label_map[basic_tokens[idx][0]],
                        basic_tokens[idx][1],
                        basic_tokens[idx][2],
                    )
                    for idx in mono
                )
            )
            items.append((canon_mono, coeff))
        return tuple(items), tuple(labels)

    def canonicalize_raw_polynomial(self, poly):
        basic_tokens = self.basic_tokens(self.alpha_index_basic)
        labels = sorted({basic_tokens[idx][0] for mono in poly for idx in mono})
        label_map = dict((label, idx) for idx, label in enumerate(labels))
        items = []
        for mono, coeff in sorted(poly.items()):
            canon_mono = tuple(
                sorted(
                    (
                        label_map[basic_tokens[idx][0]],
                        basic_tokens[idx][1],
                        basic_tokens[idx][2],
                    )
                    for idx in mono
                )
            )
            items.append((canon_mono, coeff))
        return tuple(items), tuple(labels)

    def connected_component_count(self, state):
        n = len(state.leaves)
        seen = [False] * n
        count = 0
        for start in range(n):
            if seen[start]:
                continue
            count += 1
            stack = [start]
            seen[start] = True
            while stack:
                cur = stack.pop()
                for nxt in range(n):
                    if cur != nxt and not seen[nxt] and state.edges[cur][nxt] > 0:
                        seen[nxt] = True
                        stack.append(nxt)
        return count

    def extract_connected_scalar_families(self):
        basic_tokens = self.basic_tokens(self.alpha_index_basic)
        families = {}
        by_degree = defaultdict(list)
        next_family = 0
        for state in self.scalar_states():
            if self.connected_component_count(state) != 1:
                continue
            canonical_poly, label_count = self.canonicalize_emitted_polynomial(
                self.scalar_polynomial(state),
                basic_tokens,
            )
            if canonical_poly not in families:
                families[canonical_poly] = {
                    "family_id": next_family,
                    "degree": len(state.leaves),
                    "label_count": label_count,
                    "example_leaves": state.leaves,
                    "canonical_poly": canonical_poly,
                }
                by_degree[len(state.leaves)].append(next_family)
                next_family += 1

        return {
            "families": families,
            "by_degree": dict(by_degree),
        }

    def extract_state_families(self):
        raw_polys, _labels = self.raw_node_polynomials()
        families = {}
        by_open_signature = defaultdict(list)
        next_family = 0

        for degree in sorted(self.states_by_degree):
            for state in self.states_by_degree[degree]:
                component_signatures = []
                for node in state.components:
                    component_signatures.append(self.canonicalize_raw_polynomial(raw_polys[node])[0])
                signature = (
                    state.open_counts,
                    tuple(component_signatures),
                )
                if signature not in families:
                    families[signature] = {
                        "family_id": next_family,
                        "degree": degree,
                        "open_counts": state.open_counts,
                        "component_count": len(state.components),
                        "example_leaves": state.leaves,
                    }
                    by_open_signature[state.open_counts].append(next_family)
                    next_family += 1

        return {
            "families": families,
            "by_open_counts": dict(by_open_signature),
        }

    def compiled_basis(self):
        basic_count = len(self.alpha_index_basic)
        times = []
        next_id = basic_count
        product_cache = {}
        mapping = []

        def get_product_node(factors):
            nonlocal next_id
            if len(factors) == 1:
                return factors[0]
            if factors in product_cache:
                return product_cache[factors]
            split = len(factors) // 2
            left = get_product_node(factors[:split])
            right = get_product_node(factors[split:])
            dst = next_id
            next_id += 1
            times.append((left, right, 1, dst))
            product_cache[factors] = dst
            return dst

        for state in self.selected_scalar_states():
            if len(state.leaves) == 1 and state.leaves[0][1] == 0:
                mapping.append(state.components[0])
                continue

            poly = self.scalar_polynomial(state)
            if len(poly) == 1:
                factors, coeff = next(iter(poly.items()))
                if len(factors) == 1 and coeff == 1:
                    mapping.append(factors[0])
                    continue

            dst = next_id
            next_id += 1
            for factors, coeff in sorted(poly.items()):
                if len(factors) < 2:
                    raise RuntimeError("unexpected degree-1 monomial in non-primitive scalar")
                split = len(factors) // 2
                left = get_product_node(factors[:split])
                right = get_product_node(factors[split:])
                times.append((left, right, coeff, dst))
            mapping.append(dst)

        return list(self.alpha_index_basic), times, mapping, next_id

    def render(self):
        potential_name = self.args.potential_name or f"sus2mlip_l{self.args.l}k{self.args.k}"
        (
            pruned_basic,
            pruned_times,
            alpha_moment_mapping,
            pruned_node_count,
        ) = self.selected_basis()

        def fmt_int_list(values):
            return "{" + ", ".join(str(v) for v in values) + "}"

        def fmt_tuple_list(values):
            return "{" + ",".join("{" + ", ".join(str(x) for x in item) + "}" for item in values) + "}"

        return (
            "MTP\n"
            "version = 1.1.0\n"
            f"potential_name = {potential_name}\n"
            f"scaling = {self.args.scaling}\n"
            f"L = {self.args.l}\n"
            "scaling_map = LK\n"
            f"species_count = {self.args.species_count}\n"
            "potential_tag =\n"
            f"radial_basis_type = {self.args.radial_basis_type}\n"
            f"        min_dist = {self.args.min_dist}\n"
            f"        max_dist = {self.args.max_dist}\n"
            f"        radial_basis_size = {self.args.radial_basis_size}\n"
            f"        radial_funcs_count = {self.args.k * (self.args.l + 1)}\n"
            f"alpha_moments_count = {pruned_node_count}\n"
            f"alpha_index_basic_count = {len(pruned_basic)}\n"
            f"alpha_index_basic = {fmt_tuple_list(pruned_basic)}\n"
            f"alpha_index_times_count = {len(pruned_times)}\n"
            f"alpha_index_times = {fmt_tuple_list(pruned_times)}\n"
            f"alpha_scalar_moments = {len(alpha_moment_mapping)}\n"
            f"alpha_moment_mapping = {fmt_int_list(alpha_moment_mapping)}\n"
        )

    def summary(self):
        (
            pruned_basic,
            pruned_times,
            pruned_mapping,
            pruned_node_count,
        ) = self.selected_basis()
        scalar_degrees = self.basis_node_degrees(
            pruned_basic,
            pruned_times,
            pruned_node_count,
        )
        scalar_degree_hist = defaultdict(int)
        for node in pruned_mapping:
            scalar_degree_hist[scalar_degrees[node]] += 1
        pieces = [
            f"K={self.args.k}",
            f"L={self.args.l}",
            f"max_degree={self.args.max_degree}",
            f"effective_lmax=({self.effective_lmax_summary()})",
            f"effective_level_max=({self.effective_level_max_summary()})",
            f"include_direct_scalar_products={self.args.include_direct_scalar_products}",
            f"include_k_relabel_equivalent_scalars={self.args.include_k_relabel_equivalent_scalars}",
            f"strong_k_family_from_degree={self.args.strong_k_family_from_degree}",
            f"emit_mode={self.args.emit_mode}",
            f"resolved_emit_mode={self.resolved_emit_mode()}",
            f"raw_basic={len(self.alpha_index_basic)}",
            f"raw_nodes={self.next_node_id}",
            f"raw_times={len(self.alpha_index_times)}",
            f"emitted_basic={len(pruned_basic)}",
            f"emitted_nodes={pruned_node_count}",
            f"emitted_times={len(pruned_times)}",
            f"emitted_scalars={len(pruned_mapping)}",
            f"scalar_degrees=({', '.join(f'd{degree}:{count}' for degree, count in sorted(scalar_degree_hist.items()))})",
            f"scalar_levels=({', '.join(f'l{level}:{count}' for level, count in self.selected_scalar_level_hist().items())})",
        ]
        degree_counts = ", ".join(
            f"d{degree}:{self.states_per_degree.get(degree, 0)}"
            for degree in range(1, self.args.max_degree + 1)
        )
        return " ".join(pieces) + f" states=({degree_counts})"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a general LK-style SUS2 untrained .mtp basis. "
            "The generator keeps mixed-mu uplift tensors block-ordered instead of "
            "forcing them into a fully symmetric tensor basis."
        )
    )
    parser.add_argument("--k", type=int, required=True, help="Number of K channels.")
    parser.add_argument("--l", type=int, required=True, help="Maximum coupling rank L.")
    parser.add_argument(
        "--max-degree",
        type=int,
        default=None,
        help=(
            "Maximum polynomial degree generated, optimized, and emitted. "
            "No intermediate or final moment will exceed this degree. "
            "Defaults to L + 1, i.e. max body order L + 2."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the generated .mtp to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--potential-name",
        type=str,
        default=None,
        help="Override the potential_name field.",
    )
    parser.add_argument("--scaling", type=float, default=0.01)
    parser.add_argument("--species-count", type=int, default=1)
    parser.add_argument("--radial-basis-type", type=str, default="RBChebyshev_sss")
    parser.add_argument("--min-dist", type=float, default=1.5)
    parser.add_argument("--max-dist", type=float, default=6.0)
    parser.add_argument("--radial-basis-size", type=int, default=8)
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help=(
            "Maximum allowed scalar path level in the final emitted basis, "
            "where level is the sum of primitive moment ranks along the path. "
            "Defaults to no additional level cutoff."
        ),
    )
    parser.add_argument(
        "--degree-l-max",
        action="append",
        default=[],
        metavar="ORDER:LMAX",
        help=(
            "Override the effective rank cap for one output degree. "
            "May be passed multiple times; unspecified degrees inherit --l."
        ),
    )
    parser.add_argument(
        "--degree-level-max",
        action="append",
        default=[],
        metavar="ORDER:LEVELMAX",
        help=(
            "Override the maximum allowed scalar path level for one output degree. "
            "May be passed multiple times; unspecified degrees inherit --max-level."
        ),
    )
    parser.add_argument(
        "--emit-mode",
        type=str,
        default="auto",
        choices=(
            "auto",
            "compact-cut",
            "degree-aware",
            "staged-family",
            "family-dag",
            "legacy-dedup",
            "direct-compiled",
        ),
        help=(
            "How to emit the final multiplication graph. "
            "'auto' picks the smallest emitted graph among the available backends, "
            "'compact-cut' uses pure balanced-cut decomposition, "
            "'degree-aware' compares legacy/product/cut candidates by state degree, "
            "'staged-family' forces manual-style staged cut templates for reusable scalar families, "
            "'family-dag' extracts a label-agnostic family DAG from the best non-family basis, "
            "'legacy-dedup' keeps the previous dedup/prune emitter, "
            "and 'direct-compiled' emits directly from scalar polynomials."
        ),
    )
    parser.add_argument(
        "--include-direct-scalar-products",
        action="store_true",
        help=(
            "Keep scalar outputs that are direct products of two scalar subgraphs. "
            "By default these are excluded from the final alpha_moment_mapping."
        ),
    )
    parser.add_argument(
        "--include-k-relabel-equivalent-scalars",
        action="store_true",
        help=(
            "Keep all scalar outputs that are equivalent under K-channel relabeling. "
            "By default only one representative from each K-relabel family is kept."
        ),
    )
    parser.add_argument(
        "--strong-k-family-from-degree",
        type=int,
        default=3,
        help=(
            "Apply strong K-channel relabel-family deduplication only to scalar outputs "
            "with degree >= this threshold. Defaults to 3."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the generation summary to stderr.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be positive")
    if args.l < 0:
        parser.error("--l must be non-negative")
    if args.max_degree is None:
        args.max_degree = args.l + 1
    if args.max_degree <= 0:
        parser.error("--max-degree must be positive")
    if args.species_count <= 0:
        parser.error("--species-count must be positive")
    if args.radial_basis_size <= 0:
        parser.error("--radial-basis-size must be positive")
    if args.strong_k_family_from_degree < 1:
        parser.error("--strong-k-family-from-degree must be positive")
    if args.max_level is not None and args.max_level < 0:
        parser.error("--max-level must be non-negative")

    parsed_degree_lmax = {}
    for raw_entry in args.degree_l_max:
        parts = raw_entry.split(":", 1)
        if len(parts) != 2:
            parser.error(
                f"--degree-l-max must be ORDER:LMAX, got {raw_entry!r}"
            )
        try:
            degree = int(parts[0])
            degree_lmax = int(parts[1])
        except ValueError as exc:
            parser.error(
                f"--degree-l-max must be ORDER:LMAX with integers, got {raw_entry!r}"
            )
        if degree < 1 or degree > args.max_degree:
            parser.error(
                f"--degree-l-max degree must satisfy 1 <= degree <= max_degree; got {raw_entry!r}"
            )
        if degree_lmax < 0 or degree_lmax > args.l:
            parser.error(
                f"--degree-l-max rank cap must satisfy 0 <= LMAX <= --l; got {raw_entry!r}"
            )
        if degree in parsed_degree_lmax:
            parser.error(
                f"--degree-l-max for degree {degree} was provided more than once"
            )
        parsed_degree_lmax[degree] = degree_lmax
    args.degree_l_max = parsed_degree_lmax

    parsed_degree_level_max = {}
    for raw_entry in args.degree_level_max:
        parts = raw_entry.split(":", 1)
        if len(parts) != 2:
            parser.error(
                f"--degree-level-max must be ORDER:LEVELMAX, got {raw_entry!r}"
            )
        try:
            degree = int(parts[0])
            degree_level_max = int(parts[1])
        except ValueError:
            parser.error(
                f"--degree-level-max must be ORDER:LEVELMAX with integers, got {raw_entry!r}"
            )
        if degree < 1 or degree > args.max_degree:
            parser.error(
                f"--degree-level-max degree must satisfy 1 <= degree <= max_degree; got {raw_entry!r}"
            )
        if degree_level_max < 0:
            parser.error(
                f"--degree-level-max cutoff must be non-negative; got {raw_entry!r}"
            )
        if degree in parsed_degree_level_max:
            parser.error(
                f"--degree-level-max for degree {degree} was provided more than once"
            )
        parsed_degree_level_max[degree] = degree_level_max
    args.degree_level_max = parsed_degree_level_max

    return args


def main():
    args = parse_args()
    generator = Generator(args)
    generator.generate()
    rendered = generator.render()

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")

    if not args.quiet:
        print(generator.summary(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

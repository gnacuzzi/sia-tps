import unittest

from sia_tp1.model import Direction, State, Transition
from sia_tp1.search import (
    CutoffReason,
    Node,
    SearchResult,
    SearchStatus,
    reconstruct_nodes,
)


class SearchModelTest(unittest.TestCase):
    def test_root_contract(self) -> None:
        state = State(player=(1, 1), boxes=frozenset({(1, 2)}))

        root = Node(
            state=state,
            parent=None,
            transition=None,
            depth=0,
            path_cost=0,
        )

        self.assertIs(root.state, state)
        self.assertIsNone(root.parent)
        self.assertEqual(root.depth, 0)
        self.assertEqual(root.path_cost, 0)

    def test_child_contract_and_path_reconstruction(self) -> None:
        initial = State(player=(1, 1), boxes=frozenset({(1, 3)}))
        child_state = State(player=(1, 2), boxes=initial.boxes)
        transition = Transition(
            state=child_state,
            direction=Direction.RIGHT,
            pushed=False,
        )
        root = Node(initial, None, None, 0, 0)
        child = Node(child_state, root, transition, 1, 1)

        self.assertEqual(reconstruct_nodes(child), (root, child))

    def test_failure_has_metrics_but_no_solution(self) -> None:
        result = SearchResult(
            status=SearchStatus.FAILURE,
            goal_node=None,
            expanded_nodes=4,
            frontier_size_at_end=0,
            max_frontier_size=2,
            elapsed_seconds=0.5,
        )

        self.assertEqual(result.expanded_nodes, 4)
        self.assertIsNone(result.solution_cost)
        self.assertIsNone(result.solution_nodes)
        self.assertIsNone(result.solution_transitions)
        self.assertIsNone(result.solution_moves)
        self.assertIsNone(result.solution_pushes)

    def test_cutoff_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a cutoff reason"):
            SearchResult(
                status=SearchStatus.CUTOFF,
                goal_node=None,
                expanded_nodes=1,
                frontier_size_at_end=1,
                max_frontier_size=1,
                elapsed_seconds=0.1,
            )

        result = SearchResult(
            status=SearchStatus.CUTOFF,
            goal_node=None,
            expanded_nodes=1,
            frontier_size_at_end=1,
            max_frontier_size=1,
            elapsed_seconds=0.1,
            cutoff_reason=CutoffReason.MAX_EXPANDED_NODES,
        )
        self.assertEqual(
            result.cutoff_reason, CutoffReason.MAX_EXPANDED_NODES
        )


if __name__ == "__main__":
    unittest.main()


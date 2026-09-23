from __future__ import annotations

import unittest

import torch

from connect4_ai import AGENT_NAMES, create_agent
from connect4_ai.agents import (
    AlphaBetaAgent,
    AlphaGoAgent,
    AlphaZeroAgent,
    DepthMinimaxAgent,
    MCTSAgent,
    PositionEvalAgent,
    RandomAgent,
    RuleBasedThink3Agent,
)


class AgentFactoryTests(unittest.TestCase):
    def test_public_catalog_contains_exactly_eight_agents(self) -> None:
        self.assertEqual(
            (
                "RandomAgent",
                "RuleBasedThink3Agent",
                "DepthMinimaxAgent",
                "AlphaBetaAgent",
                "PositionEvalAgent",
                "MCTSAgent",
                "AlphaGoAgent",
                "AlphaZeroAgent",
            ),
            AGENT_NAMES,
        )

    def test_factory_constructs_all_eight_agents(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        cases = (
            ("RandomAgent", RandomAgent, {}),
            ("RuleBasedThink3Agent", RuleBasedThink3Agent, {}),
            ("DepthMinimaxAgent", DepthMinimaxAgent, {"depth": 1}),
            ("AlphaBetaAgent", AlphaBetaAgent, {"depth": 1}),
            ("PositionEvalAgent", PositionEvalAgent, {"depth": 0}),
            ("MCTSAgent", MCTSAgent, {"num_rollouts": 1}),
            (
                "AlphaGoAgent",
                AlphaGoAgent,
                {"depth": 1, "num_rollouts": 1},
            ),
            ("AlphaZeroAgent", AlphaZeroAgent, {"num_rollouts": 1}),
        )
        for name, expected_type, options in cases:
            with self.subTest(agent=name):
                agent = create_agent(name, seed=123, **options)
                self.assertIsInstance(agent, expected_type)

    def test_unknown_agent_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            create_agent("UnknownAgent")


if __name__ == "__main__":
    unittest.main()

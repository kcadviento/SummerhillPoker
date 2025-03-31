"""Basic tests for SummerhillPoker core functionality."""

import unittest
from summerhillpoker import Game, PlayerProfile, State, Hand, Card, Rank, Suit

class TestCards(unittest.TestCase):
    """Test card and hand evaluation functionality."""
    
    def test_card_creation(self):
        """Test creating cards."""
        card = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(str(card), "As")
        
        card2 = Card.from_str("Kh")
        self.assertEqual(card2.rank, Rank.KING)
        self.assertEqual(card2.suit, Suit.HEARTS)
    
    def test_hand_evaluation(self):
        """Test hand evaluation."""
        # Test a flush
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
        ]
        hand = Hand(cards)
        self.assertEqual(hand.hand_type.name, "FLUSH")
        
        # Test a pair
        cards2 = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.CLUBS),
        ]
        hand2 = Hand(cards2)
        self.assertEqual(hand2.hand_type.name, "PAIR")
        
        # Test comparison
        self.assertTrue(hand > hand2)  # Flush beats pair

class TestGame(unittest.TestCase):
    """Test game functionality."""
    
    def test_game_creation(self):
        """Test creating a game."""
        game = Game.texas_holdem(small_blind=1, big_blind=2)
        self.assertIsNotNone(game)
        
        # Test creating a state
        state = game.create_state(num_players=6, starting_stack=1000)
        self.assertEqual(len(state.players), 6)
        self.assertEqual(state.players[0].stack, 1000)
        
        # Test dealing cards
        game.deal_cards(state)
        self.assertEqual(len(state.players[0].hole_cards), 2)  # Texas Hold'em has 2 hole cards

class TestPlayerProfile(unittest.TestCase):
    """Test player profile functionality."""
    
    def test_profile_creation(self):
        """Test creating player profiles."""
        profile = PlayerProfile("aggressive", stack=2000)
        self.assertGreater(profile.aggression_factor_range[0], 2.0)  # Aggressive profile has high aggression
        
        profile2 = PlayerProfile("passive", stack=1500)
        self.assertLess(profile2.aggression_factor_range[1], 2.0)  # Passive profile has low aggression

if __name__ == '__main__':
    unittest.main()
"""Module for different poker game variants and configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from summerhillpoker.hands import Card, Rank, Suit
from summerhillpoker.state import State, PlayerState, BettingRound


class Game(ABC):
    """Base class for poker game variants."""
    
    @abstractmethod
    def create_state(self, num_players: int, starting_stack: int) -> State:
        """Create an initial game state."""
        pass
    
    @abstractmethod
    def deal_cards(self, state: State) -> None:
        """Deal cards to players and board according to game rules."""
        pass
    
    @abstractmethod
    def evaluate_hands(self, state: State) -> Dict[int, int]:
        """Evaluate player hands and distribute the pot.
        
        Returns a dictionary mapping player IDs to their winnings.
        """
        pass
        
    @classmethod
    def texas_holdem(cls, small_blind: int = 1, big_blind: int = 2, ante: int = 0) -> TexasHoldem:
        """Factory method to create a Texas Hold'em game."""
        return TexasHoldem(small_blind, big_blind, ante)
        
    @classmethod
    def omaha(cls, small_blind: int = 1, big_blind: int = 2, ante: int = 0) -> Omaha:
        """Factory method to create an Omaha game."""
        return Omaha(small_blind, big_blind, ante)
    
    def create_deck(self) -> List[Card]:
        """Create a standard deck of 52 cards."""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank, suit))
        return deck
    
    def shuffle_deck(self, deck: List[Card]) -> List[Card]:
        """Shuffle the deck."""
        import random
        shuffled = deck.copy()
        random.shuffle(shuffled)
        return shuffled


class TexasHoldem(Game):
    """Texas Hold'em poker game implementation."""
    
    def __init__(self, small_blind: int = 1, big_blind: int = 2, ante: int = 0):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
    
    def create_state(self, num_players: int, starting_stack: int) -> State:
        """Create an initial Texas Hold'em game state."""
        # Create players
        players = [PlayerState(i, starting_stack) for i in range(num_players)]
        
        # Create deck
        deck = self.shuffle_deck(self.create_deck())
        
        # Create state
        state = State(
            players=players,
            deck=deck,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            antes=self.ante,
        )
        
        # Set positions
        for i, player in enumerate(players):
            player.position = i
        
        # Post blinds
        self._post_blinds(state)
        
        return state
    
    def _post_blinds(self, state: State) -> None:
        """Post blinds for a new hand."""
        num_players = len(state.players)
        
        # Post antes if any
        if state.antes > 0:
            for player in state.players:
                player.stack -= state.antes
                player.total_bet += state.antes
                state.pot += state.antes
        
        # Post small blind
        sb_idx = (state.dealer_idx + 1) % num_players
        sb_player = state.players[sb_idx]
        sb_player.stack -= state.small_blind
        sb_player.current_bet = state.small_blind
        sb_player.total_bet += state.small_blind
        state.pot += state.small_blind
        
        # Post big blind
        bb_idx = (state.dealer_idx + 2) % num_players
        bb_player = state.players[bb_idx]
        bb_player.stack -= state.big_blind
        bb_player.current_bet = state.big_blind
        bb_player.total_bet += state.big_blind
        state.pot += state.big_blind
        
        # Set current bet to big blind
        state.current_bet = state.big_blind
        
        # Set first to act (after big blind)
        state.current_player_idx = (state.dealer_idx + 3) % num_players
    
    def deal_cards(self, state: State) -> None:
        """Deal hole cards to players and community cards to the board."""
        # Deal 2 hole cards to each player
        for _ in range(2):
            for player in state.players:
                player.hole_cards.append(state.deck.pop())
                
        # Community cards are dealt during gameplay in advance_betting_round
    
    def evaluate_hands(self, state: State) -> Dict[int, int]:
        """Evaluate player hands and distribute the pot."""
        from summerhillpoker.hands import Hand
        
        # Only evaluate active players
        active_players = [p for p in state.players if p.is_active]
        
        # Create hands for each player using hole cards and community cards
        best_hands = {}
        for player in active_players:
            # For Texas Hold'em, a player can use any 5 cards from their 2 hole cards and 5 community cards
            # This is a simplified implementation for now
            all_cards = player.hole_cards + state.board
            hand = Hand(all_cards)  # In a real implementation, we'd find the best 5-card hand
            best_hands[player.player_id] = hand
        
        # Find winners (may be multiple in case of ties)
        winners = []
        best_hand = None
        
        for player_id, hand in best_hands.items():
            if not best_hand or hand > best_hand:
                winners = [player_id]
                best_hand = hand
            elif hand == best_hand:
                winners.append(player_id)
        
        # Distribute pot
        pot_per_winner = state.pot // len(winners)
        result = {player_id: 0 for player_id in range(len(state.players))}
        
        for winner_id in winners:
            state.players[winner_id].stack += pot_per_winner
            state.players[winner_id].hands_won += 1
            state.players[winner_id].total_profit += pot_per_winner - state.players[winner_id].total_bet
            result[winner_id] = pot_per_winner
        
        # Record hand history
        hand_record = {
            "winners": winners,
            "pot": state.pot,
            "board": [str(card) for card in state.board],
            "player_hands": {player_id: str(hand) for player_id, hand in best_hands.items()},
            "actions": [(a.player_id, a.action_type.name, a.amount) for a in state.actions]
        }
        state.hand_history.append(hand_record)
        
        return result


class Omaha(Game):
    """Omaha poker game implementation."""
    
    def __init__(self, small_blind: int = 1, big_blind: int = 2, ante: int = 0):
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.ante = ante
    
    def create_state(self, num_players: int, starting_stack: int) -> State:
        """Create an initial Omaha game state."""
        # Similar to Texas Hold'em but with 4 hole cards per player
        # Create players
        players = [PlayerState(i, starting_stack) for i in range(num_players)]
        
        # Create deck
        deck = self.shuffle_deck(self.create_deck())
        
        # Create state
        state = State(
            players=players,
            deck=deck,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            antes=self.ante,
        )
        
        # Set positions
        for i, player in enumerate(players):
            player.position = i
        
        # Post blinds (same as Texas Hold'em)
        self._post_blinds(state)
        
        return state
    
    def _post_blinds(self, state: State) -> None:
        """Post blinds for a new hand."""
        # Same as Texas Hold'em
        num_players = len(state.players)
        
        # Post antes if any
        if state.antes > 0:
            for player in state.players:
                player.stack -= state.antes
                player.total_bet += state.antes
                state.pot += state.antes
        
        # Post small blind
        sb_idx = (state.dealer_idx + 1) % num_players
        sb_player = state.players[sb_idx]
        sb_player.stack -= state.small_blind
        sb_player.current_bet = state.small_blind
        sb_player.total_bet += state.small_blind
        state.pot += state.small_blind
        
        # Post big blind
        bb_idx = (state.dealer_idx + 2) % num_players
        bb_player = state.players[bb_idx]
        bb_player.stack -= state.big_blind
        bb_player.current_bet = state.big_blind
        bb_player.total_bet += state.big_blind
        state.pot += state.big_blind
        
        # Set current bet to big blind
        state.current_bet = state.big_blind
        
        # Set first to act (after big blind)
        state.current_player_idx = (state.dealer_idx + 3) % num_players
    
    def deal_cards(self, state: State) -> None:
        """Deal hole cards to players and community cards to the board."""
        # Deal 4 hole cards to each player (instead of 2 in Texas Hold'em)
        for _ in range(4):
            for player in state.players:
                player.hole_cards.append(state.deck.pop())
                
        # Community cards are dealt during gameplay in advance_betting_round
    
    def evaluate_hands(self, state: State) -> Dict[int, int]:
        """Evaluate player hands and distribute the pot."""
        from summerhillpoker.hands import Hand
        
        # Only evaluate active players
        active_players = [p for p in state.players if p.is_active]
        
        # Create hands for each player using hole cards and community cards
        # In Omaha, players must use exactly 2 of their 4 hole cards and 3 of the 5 community cards
        best_hands = {}
        for player in active_players:
            # Simplified implementation - in a real implementation we'd check all combinations
            # of 2 hole cards and 3 community cards to find the best hand
            all_cards = player.hole_cards[:2] + state.board[:3]  # Just use first 2 hole cards and first 3 community cards
            hand = Hand(all_cards)
            best_hands[player.player_id] = hand
        
        # Find winners (may be multiple in case of ties)
        winners = []
        best_hand = None
        
        for player_id, hand in best_hands.items():
            if not best_hand or hand > best_hand:
                winners = [player_id]
                best_hand = hand
            elif hand == best_hand:
                winners.append(player_id)
        
        # Distribute pot
        pot_per_winner = state.pot // len(winners)
        result = {player_id: 0 for player_id in range(len(state.players))}
        
        for winner_id in winners:
            state.players[winner_id].stack += pot_per_winner
            state.players[winner_id].hands_won += 1
            state.players[winner_id].total_profit += pot_per_winner - state.players[winner_id].total_bet
            result[winner_id] = pot_per_winner
        
        # Record hand history
        hand_record = {
            "winners": winners,
            "pot": state.pot,
            "board": [str(card) for card in state.board],
            "player_hands": {player_id: str(hand) for player_id, hand in best_hands.items()},
            "actions": [(a.player_id, a.action_type.name, a.amount) for a in state.actions]
        }
        state.hand_history.append(hand_record)
        
        return result

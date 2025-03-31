"""Module for poker game simulation with opponent modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from collections import defaultdict

from summerhillpoker.state import State, Action, ActionType, BettingRound, PlayerState
from summerhillpoker.games import Game, TexasHoldem, Omaha
from summerhillpoker.modeling import PlayerProfile, BehaviorTracker, ProfileCluster
from summerhillpoker.hands import Card, Hand


@dataclass
class GameResult:
    """Contains the results of a simulated poker game or hand."""
    winners: List[int]  # IDs of winning players
    pot: int  # Total pot size
    player_stacks: Dict[int, int]  # Final player stacks
    player_hands: Dict[int, List[Card]]  # Player hands
    board: List[Card]  # Community cards
    actions: List[Tuple[int, ActionType, int]]  # Actions taken in the hand
    
    def summary(self) -> str:
        """Get a human-readable summary of the game result."""
        winner_str = ", ".join(str(w) for w in self.winners)
        board_str = " ".join(str(c) for c in self.board)
        
        hands_str = ""
        for player_id, cards in self.player_hands.items():
            hands_str += f"\nPlayer {player_id}: {' '.join(str(c) for c in cards)}"
        
        return f"Winners: {winner_str}\nPot: {self.pot}\nBoard: {board_str}{hands_str}"


class Simulator:
    """Simulates poker games with opponent modeling and strategy adaptation."""
    
    def __init__(self, game: Game, player_profiles: List[PlayerProfile]):
        self.game = game
        self.player_profiles = player_profiles
        self.behavior_tracker = BehaviorTracker()
        self.results = []
        self.state = None
    
    def run_simulation(self, num_hands: int = 100) -> List[GameResult]:
        """Run a full simulation for a specified number of hands."""
        self.results = []
        
        for _ in range(num_hands):
            result = self._play_hand()
            self.results.append(result)
        
        return self.results
    
    def _play_hand(self) -> GameResult:
        """Play a single hand of poker with the current game and profiles."""
        # Create initial state
        num_players = len(self.player_profiles)
        starting_stacks = [profile.stack for profile in self.player_profiles]
        
        # Initialize state with actual stacks from previous hands if available
        if self.state and self.state.players:
            starting_stacks = [player.stack for player in self.state.players]
        
        # Create new state for this hand
        self.state = self.game.create_state(num_players, 1000)  # Default stack
        
        # Update stacks from previous hand
        for i, stack in enumerate(starting_stacks):
            if i < len(self.state.players):
                self.state.players[i].stack = stack
        
        # Deal cards
        self.game.deal_cards(self.state)
        
        # Initialize tracking for the new hand
        self.behavior_tracker.start_hand([p.player_id for p in self.state.players])
        
        # Play through betting rounds
        while not self.state.is_terminal():
            # Complete current betting round
            self._play_betting_round()
            
            # Check if hand is over
            if self.state.is_terminal():
                break
                
            # Advance to next betting round
            self.state.advance_betting_round()
        
        # Evaluate hands and distribute pot
        winnings = self.game.evaluate_hands(self.state)
        winners = [player_id for player_id, amount in winnings.items() if amount > 0]
        
        # Create result object
        result = GameResult(
            winners=winners,
            pot=self.state.pot,
            player_stacks={p.player_id: p.stack for p in self.state.players},
            player_hands={p.player_id: p.hole_cards for p in self.state.players},
            board=self.state.board,
            actions=[(a.player_id, a.action_type, a.amount) for a in self.state.actions]
        )
        
        return result
    
    def _play_betting_round(self) -> None:
        """Play through a single betting round."""
        # Betting continues until everyone has matched the current bet or folded
        players_to_act = [p for p in self.state.players if p.is_active]
        players_acted = set()
        current_bet_at_start = self.state.current_bet
        
        while True:
            current_player = self.state.players[self.state.current_player_idx]
            
            # Skip if player is not active
            if not current_player.is_active:
                self.state._advance_player()
                continue
            
            # Get legal actions
            legal_actions = self.state.get_legal_actions(self.state.current_player_idx)
            
            # Get player profile and decide action
            profile = self.player_profiles[current_player.player_id]
            action = self._decide_action(current_player, legal_actions, profile)
            
            # Apply action to state
            self.state.apply_action(action)
            
            # Track action for modeling
            self.behavior_tracker.track_action(current_player.player_id, action, self.state)
            
            # Add player to acted set
            players_acted.add(current_player.player_id)
            
            # Check if betting round is complete
            active_players = [p for p in self.state.players if p.is_active]
            
            # Betting round is over if:
            # 1. Everyone has acted at least once, and
            # 2. Either all active players have matched the current bet, or only one player is active
            all_acted = all(p.player_id in players_acted for p in active_players)
            all_matched = all(p.current_bet == self.state.current_bet for p in active_players)
            
            if (all_acted and all_matched) or len(active_players) <= 1:
                break
            
            # If the current bet has changed and a player has already acted,
            # they need to act again
            if self.state.current_bet > current_bet_at_start:
                # Reset acted set for players who haven't matched the new bet
                for player in active_players:
                    if player.current_bet < self.state.current_bet:
                        players_acted.discard(player.player_id)
    
    def _decide_action(self, player: PlayerState, legal_actions: List[ActionType], profile: PlayerProfile) -> Action:
        """Decide an action based on the player's profile and the current state.
        
        This is where most of the opponent modeling and strategy comes into play.
        """
        # Implementation would use player profiles, behavioral history, and potentially CFR
        # For now, we have a simplistic implementation
        
        # Get random values within the profile ranges
        vpip = random.uniform(profile.vpip_range[0], profile.vpip_range[1])
        pfr = random.uniform(profile.pfr_range[0], profile.pfr_range[1])
        aggression = random.uniform(profile.aggression_factor_range[0], profile.aggression_factor_range[1])
        
        # Pre-flop decision logic
        if self.state.betting_round == BettingRound.PREFLOP:
            # Simplified hand strength
            # In a real implementation, we would evaluate the actual hand
            hand_strength = random.random()  # 0-1 random value to simulate hand strength
            
            # Simple check for VPIP threshold
            if hand_strength < vpip:
                # This hand is worth playing
                if ActionType.RAISE in legal_actions and hand_strength < pfr:
                    # Raise if hand is strong and raising is legal
                    amount = self.state.current_bet * 2  # Simple raise sizing
                    return Action(ActionType.RAISE, amount, player.player_id)
                elif ActionType.CALL in legal_actions:
                    # Call if hand is playable but not strong enough to raise
                    amount = self.state.current_bet - player.current_bet
                    return Action(ActionType.CALL, amount, player.player_id)
                elif ActionType.CHECK in legal_actions:
                    # Check if no bet to call
                    return Action(ActionType.CHECK, 0, player.player_id)
            
            # Fold if hand is not worth playing or no legal alternative
            if ActionType.FOLD in legal_actions:
                return Action(ActionType.FOLD, 0, player.player_id)
            else:
                # Check if possible
                return Action(ActionType.CHECK, 0, player.player_id)
        
        # Post-flop decision logic
        else:
            # In post-flop, decisions would be much more sophisticated based on:
            # - Board texture
            # - Hand strength
            # - Opponent modeling
            # - Pot odds
            
            # Simple placeholder strategy based on profile aggressiveness
            aggression_threshold = 0.5  # Base threshold
            rand_factor = random.random()  # Random factor
            
            # More aggressive players will bet/raise more often
            if rand_factor < (aggression / 5.0):  # Scale aggression to 0-1 range
                if ActionType.RAISE in legal_actions:
                    amount = min(self.state.current_bet * 2, player.stack)
                    return Action(ActionType.RAISE, amount, player.player_id)
                elif ActionType.BET in legal_actions:
                    amount = min(self.state.pot // 2, player.stack)  # Bet half pot
                    return Action(ActionType.BET, amount, player.player_id)
            
            # Otherwise, take passive action
            if ActionType.CALL in legal_actions:
                amount = self.state.current_bet - player.current_bet
                return Action(ActionType.CALL, amount, player.player_id)
            elif ActionType.CHECK in legal_actions:
                return Action(ActionType.CHECK, 0, player.player_id)
            else:
                return Action(ActionType.FOLD, 0, player.player_id)
    
    def get_player_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each player from the simulation."""
        stats = {}
        
        for player_id in range(len(self.player_profiles)):
            player_stats = self.behavior_tracker.get_player_profile(player_id)
            
            # Calculate additional metrics
            hands_won = sum(1 for result in self.results if player_id in result.winners)
            total_profit = 0
            
            if self.results:
                initial_stack = self.player_profiles[player_id].stack
                final_stack = self.results[-1].player_stacks.get(player_id, initial_stack)
                total_profit = final_stack - initial_stack
            
            stats[player_id] = {
                **player_stats,  # Include behavioral stats
                "hands_played": len(self.results),
                "hands_won": hands_won,
                "win_rate": hands_won / len(self.results) if self.results else 0,
                "total_profit": total_profit
            }
        
        return stats

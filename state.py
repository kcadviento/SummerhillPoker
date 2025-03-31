"""Module for poker game state management with opponent tracking capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Set

from summerhillpoker.hands import Card, Hand


class ActionType(Enum):
    """Types of actions a player can take."""
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()
    ALL_IN = auto()


@dataclass
class Action:
    """Represents a player action in the game."""
    action_type: ActionType
    amount: int = 0  # Amount of chips for betting actions
    player_id: int = -1
    
    # Additional fields for tracking behavior patterns
    pot_size_ratio: float = 0.0  # Action amount relative to pot size
    stack_ratio: float = 0.0     # Action amount relative to player stack
    position_type: str = ""      # Position type (early, middle, late)
    
    def is_aggressive(self) -> bool:
        """Determine if the action is aggressive (bet/raise vs check/call)."""
        return self.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
    
    def is_passive(self) -> bool:
        """Determine if the action is passive."""
        return self.action_type in (ActionType.CHECK, ActionType.CALL)
    
    def is_fold(self) -> bool:
        """Determine if the action is a fold."""
        return self.action_type == ActionType.FOLD


class BettingRound(Enum):
    """Represents the betting rounds in poker."""
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()


@dataclass
class PlayerState:
    """Represents the state of a player in the game."""
    player_id: int
    stack: int
    hole_cards: List[Card] = field(default_factory=list)
    is_active: bool = True  # Whether player is still in the hand
    current_bet: int = 0    # Amount bet in the current betting round
    total_bet: int = 0      # Total amount bet in the current hand
    position: int = 0       # Position at the table (0 = button)
    
    # History tracking for behavior analysis
    action_history: List[Action] = field(default_factory=list)
    hands_played: int = 0
    hands_won: int = 0
    total_profit: int = 0
    
    # Behavioral metrics (updated based on play)
    aggression_factor: float = 0.5  # Ratio of aggressive to passive actions
    vpip: float = 0.0              # Voluntarily put money in pot %
    pfr: float = 0.0               # Pre-flop raise %
    af: float = 0.0                # Aggression frequency
    
    def reset_for_hand(self) -> None:
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.is_active = True
        self.current_bet = 0
        self.total_bet = 0

    def add_action(self, action: Action) -> None:
        """Add an action to the player's history and update metrics."""
        self.action_history.append(action)
        
        # Update behavioral metrics based on the action
        self._update_metrics(action)
    
    def _update_metrics(self, action: Action) -> None:
        """Update behavioral metrics based on the action."""
        # Simple example of metrics update logic
        aggressive_actions = sum(1 for a in self.action_history if a.is_aggressive())
        passive_actions = sum(1 for a in self.action_history if a.is_passive())
        
        if passive_actions > 0:
            self.aggression_factor = aggressive_actions / passive_actions
        else:
            self.aggression_factor = aggressive_actions if aggressive_actions > 0 else 0


@dataclass
class State:
    """Represents the complete state of a poker game."""
    players: List[PlayerState] = field(default_factory=list)
    board: List[Card] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    pot: int = 0
    current_player_idx: int = 0
    dealer_idx: int = 0
    min_bet: int = 0
    current_bet: int = 0  # Highest bet in the current betting round
    betting_round: BettingRound = BettingRound.PREFLOP
    
    # Game configuration
    small_blind: int = 0
    big_blind: int = 0
    antes: int = 0
    
    # Round history for analysis
    actions: List[Action] = field(default_factory=list)
    hand_history: List[Dict] = field(default_factory=list) 
    
    # Information sets for CFR
    info_sets: Dict[str, Dict] = field(default_factory=dict)
    
    def apply_action(self, action: Action) -> State:
        """Apply an action to the state and return the new state."""
        # Make a copy of the current state
        # In a real implementation, this would create a proper deep copy
        
        # Record the action
        self.actions.append(action)
        
        # Update player state
        player = self.players[self.current_player_idx]
        player.add_action(action)
        
        if action.action_type == ActionType.FOLD:
            player.is_active = False
        elif action.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            player.current_bet = action.amount
            player.total_bet += action.amount
            player.stack -= action.amount
            self.pot += action.amount
            
            if action.amount > self.current_bet:
                self.current_bet = action.amount
        
        # Move to next player
        self._advance_player()
        
        # Update info sets for CFR
        self._update_info_sets(action)
        
        return self
    
    def _advance_player(self) -> None:
        """Move to the next active player."""
        active_players = [i for i, p in enumerate(self.players) if p.is_active]
        if not active_players:
            return
            
        # Find next active player
        next_idx = (self.current_player_idx + 1) % len(self.players)
        while next_idx != self.current_player_idx:
            if self.players[next_idx].is_active:
                self.current_player_idx = next_idx
                break
            next_idx = (next_idx + 1) % len(self.players)
    
    def _update_info_sets(self, action: Action) -> None:
        """Update information sets based on the action."""
        # This would contain logic to update information sets for CFR
        # Simplified placeholder implementation
        player = self.players[action.player_id]
        info_key = self._get_info_set_key(player)
        
        if info_key not in self.info_sets:
            self.info_sets[info_key] = {}
            
        # Track action frequencies in this information set
        action_key = f"{action.action_type.name}_{action.amount}"
        if action_key in self.info_sets[info_key]:
            self.info_sets[info_key][action_key] += 1
        else:
            self.info_sets[info_key][action_key] = 1
    
    def _get_info_set_key(self, player: PlayerState) -> str:
        """Generate a key for the information set."""
        # Typically includes player cards, public cards, and betting history
        # Simplified implementation
        hole_cards = "".join(str(card) for card in player.hole_cards)
        board_cards = "".join(str(card) for card in self.board)
        betting = "".join(f"{a.action_type.name[0]}{a.amount}" for a in self.actions[-5:])  # Last 5 actions
        return f"{hole_cards}|{board_cards}|{betting}|{self.betting_round.name}"
    
    def get_legal_actions(self, player_idx: int) -> List[ActionType]:
        """Get the list of legal actions for a player."""
        player = self.players[player_idx]
        legal_actions = [ActionType.FOLD]
        
        # Can check if no bets have been made
        if self.current_bet == 0 or player.current_bet == self.current_bet:
            legal_actions.append(ActionType.CHECK)
        
        # Can call if there's a bet to call
        if self.current_bet > player.current_bet:
            call_amount = self.current_bet - player.current_bet
            if call_amount < player.stack:
                legal_actions.append(ActionType.CALL)
        
        # Can bet if no bets have been made
        if self.current_bet == 0:
            legal_actions.append(ActionType.BET)
        
        # Can raise if there's a bet to raise
        elif self.current_bet > 0 and player.stack > (self.current_bet - player.current_bet):
            legal_actions.append(ActionType.RAISE)
        
        # All-in is always an option if player has chips
        if player.stack > 0:
            legal_actions.append(ActionType.ALL_IN)
            
        return legal_actions
    
    def is_terminal(self) -> bool:
        """Check if the state is terminal (hand is over)."""
        # A state is terminal if only one player is active or we've reached showdown
        active_players = sum(1 for p in self.players if p.is_active)
        return active_players <= 1 or (self.betting_round == BettingRound.RIVER and self._betting_complete())
    
    def _betting_complete(self) -> bool:
        """Check if betting is complete for the current round."""
        # Betting is complete when all active players have bet the same amount
        active_players = [p for p in self.players if p.is_active]
        return all(p.current_bet == self.current_bet for p in active_players)
    
    def advance_betting_round(self) -> None:
        """Advance to the next betting round."""
        if self.betting_round == BettingRound.PREFLOP:
            self.betting_round = BettingRound.FLOP
            # Deal flop cards
            self.board.extend(self.deck.pop() for _ in range(3))
        elif self.betting_round == BettingRound.FLOP:
            self.betting_round = BettingRound.TURN
            # Deal turn card
            self.board.append(self.deck.pop())
        elif self.betting_round == BettingRound.TURN:
            self.betting_round = BettingRound.RIVER
            # Deal river card
            self.board.append(self.deck.pop())
        
        # Reset betting for new round
        self.current_bet = 0
        for player in self.players:
            player.current_bet = 0
        
        # First active player after dealer acts first
        self.current_player_idx = (self.dealer_idx + 1) % len(self.players)
        while not self.players[self.current_player_idx].is_active:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

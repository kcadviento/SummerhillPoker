"""Module for poker hand evaluation and card representation."""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from functools import total_ordering
from typing import Dict, List, Optional, Set, Tuple


class Suit(Enum):
    """Card suits."""
    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"
    
    def __str__(self) -> str:
        return self.value


class Rank(Enum):
    """Card ranks."""
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def value_int(self) -> int:
        """Get the integer value of the rank."""
        values = {
            Rank.TWO: 2,
            Rank.THREE: 3,
            Rank.FOUR: 4,
            Rank.FIVE: 5, 
            Rank.SIX: 6,
            Rank.SEVEN: 7,
            Rank.EIGHT: 8,
            Rank.NINE: 9,
            Rank.TEN: 10,
            Rank.JACK: 11,
            Rank.QUEEN: 12,
            Rank.KING: 13,
            Rank.ACE: 14
        }
        return values[self]


@dataclass(frozen=True)
class Card:
    """Represents a playing card."""
    rank: Rank
    suit: Suit
    
    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"
    
    @classmethod
    def from_str(cls, card_str: str) -> Card:
        """Create a card from string representation."""
        if len(card_str) != 2:
            raise ValueError(f"Invalid card string: {card_str}")
            
        rank_char = card_str[0].upper()
        suit_char = card_str[1].lower()
        
        rank = next((r for r in Rank if r.value == rank_char), None)
        suit = next((s for s in Suit if s.value == suit_char), None)
        
        if not rank or not suit:
            raise ValueError(f"Invalid card string: {card_str}")
            
        return cls(rank, suit)


class HandType(Enum):
    """Types of poker hands from weakest to strongest."""
    HIGH_CARD = auto()
    PAIR = auto()
    TWO_PAIR = auto()
    THREE_OF_A_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_OF_A_KIND = auto()
    STRAIGHT_FLUSH = auto()
    ROYAL_FLUSH = auto()
    
    def __lt__(self, other: HandType) -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Hand:
    """Represents a poker hand with evaluation capabilities."""
    
    def __init__(self, cards: List[Card]):
        self.cards = cards
    
    @property
    def hand_type(self) -> HandType:
        """Determine the type of this hand."""
        if self._is_royal_flush():
            return HandType.ROYAL_FLUSH
        elif self._is_straight_flush():
            return HandType.STRAIGHT_FLUSH
        elif self._is_four_of_a_kind():
            return HandType.FOUR_OF_A_KIND
        elif self._is_full_house():
            return HandType.FULL_HOUSE
        elif self._is_flush():
            return HandType.FLUSH
        elif self._is_straight():
            return HandType.STRAIGHT
        elif self._is_three_of_a_kind():
            return HandType.THREE_OF_A_KIND
        elif self._is_two_pair():
            return HandType.TWO_PAIR
        elif self._is_pair():
            return HandType.PAIR
        else:
            return HandType.HIGH_CARD
    
    def _rank_counts(self) -> Dict[Rank, int]:
        """Count occurrences of each rank in the hand."""
        counts = {}
        for card in self.cards:
            if card.rank in counts:
                counts[card.rank] += 1
            else:
                counts[card.rank] = 1
        return counts
    
    def _is_flush(self) -> bool:
        """Check if the hand is a flush."""
        return len(set(card.suit for card in self.cards)) == 1
    
    def _is_straight(self) -> bool:
        """Check if the hand is a straight."""
        if len(self.cards) < 5:
            return False
            
        # Sort by rank value
        sorted_cards = sorted(self.cards, key=lambda card: card.rank.value_int)
        
        # Check for A-5 straight
        if (sorted_cards[-1].rank == Rank.ACE and
            sorted_cards[0].rank == Rank.TWO and
            sorted_cards[1].rank == Rank.THREE and
            sorted_cards[2].rank == Rank.FOUR and
            sorted_cards[3].rank == Rank.FIVE):
            return True
        
        # Check for regular straight
        for i in range(len(sorted_cards) - 1):
            if sorted_cards[i+1].rank.value_int - sorted_cards[i].rank.value_int != 1:
                return False
        return True
    
    def _is_straight_flush(self) -> bool:
        """Check if the hand is a straight flush."""
        return self._is_straight() and self._is_flush()
    
    def _is_royal_flush(self) -> bool:
        """Check if the hand is a royal flush."""
        if not self._is_straight_flush():
            return False
            
        ranks = {card.rank for card in self.cards}
        royal_ranks = {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE}
        return ranks == royal_ranks
    
    def _is_four_of_a_kind(self) -> bool:
        """Check if the hand has four of a kind."""
        counts = self._rank_counts()
        return 4 in counts.values()
    
    def _is_full_house(self) -> bool:
        """Check if the hand is a full house."""
        counts = self._rank_counts()
        return 3 in counts.values() and 2 in counts.values()
    
    def _is_three_of_a_kind(self) -> bool:
        """Check if the hand has three of a kind."""
        counts = self._rank_counts()
        return 3 in counts.values()
    
    def _is_two_pair(self) -> bool:
        """Check if the hand has two pairs."""
        counts = self._rank_counts()
        pairs = sum(1 for count in counts.values() if count == 2)
        return pairs == 2
    
    def _is_pair(self) -> bool:
        """Check if the hand has a pair."""
        counts = self._rank_counts()
        return 2 in counts.values()
    
    def get_kickers(self, n: int = 5) -> List[Card]:
        """Get the kickers for this hand."""
        # This is a simplified implementation
        sorted_cards = sorted(self.cards, key=lambda card: card.rank.value_int, reverse=True)
        return sorted_cards[:n]
    
    def compare(self, other: Hand) -> int:
        """Compare this hand with another hand.
        
        Returns:
            -1 if this hand is weaker
             0 if the hands are equal
             1 if this hand is stronger
        """
        if self.hand_type.value < other.hand_type.value:
            return -1
        elif self.hand_type.value > other.hand_type.value:
            return 1
        else:
            # Same hand type, compare kickers
            self_kickers = self.get_kickers()
            other_kickers = other.get_kickers()
            
            for self_card, other_card in zip(self_kickers, other_kickers):
                if self_card.rank.value_int < other_card.rank.value_int:
                    return -1
                elif self_card.rank.value_int > other_card.rank.value_int:
                    return 1
            
            return 0  # Equal hands
    
    def __lt__(self, other: Hand) -> bool:
        return self.compare(other) < 0
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hand):
            return NotImplemented
        return self.compare(other) == 0
    
    def __gt__(self, other: Hand) -> bool:
        return self.compare(other) > 0
    
    def __str__(self) -> str:
        return f"{self.hand_type.name}: {', '.join(str(card) for card in self.cards)}"

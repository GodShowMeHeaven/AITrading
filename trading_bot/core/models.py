"""
Core domain models for trading bot
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    margin: float
    liquidation_price: float
    timestamp: datetime

@dataclass
class Trade:
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "new"
    metadata: Dict = field(default_factory=dict)

@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    params: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

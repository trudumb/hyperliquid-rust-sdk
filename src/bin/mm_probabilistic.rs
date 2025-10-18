use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;

//RUST_LOG=info cargo run --bin mm_probabilistic

// ============================================
// PROBABILISTIC MARKET MAKER V4
// ============================================
// CRITICAL IMPROVEMENTS TO TOXIC FLOW DETECTION:
// 
// 1. MULTI-TIMEFRAME TOXICITY ANALYSIS
//    - Analyzes flow over 10s, 60s, and 300s windows
//    - Detects momentum (consecutive trades in same direction)
//    - Measures volume surges (recent vs average)
//    - Calculates price impact over each timeframe
//
// 2. VOLUME-WEIGHTED DIRECTIONAL FLOW
//    - Uses actual trade volume, not just trade count
//    - Dynamic prior = 10% of total volume (not fixed)
//    - Prevents 0% buy ratio from small sample bias
//    - Statistical confidence based on both sample size AND total volume
//
// 3. ADVERSE SELECTION TRACKING
//    - Tracks when trades move against our orders
//    - Detects when someone hits our bid and price drops (adverse)
//    - Detects when someone lifts our offer and price rises (adverse)
//    - Cumulative slippage measurement
//    - Uses adverse ratio to adjust toxicity probability
//
// 4. VWAP CALCULATION
//    - Maintains volume-weighted average price for buy and sell sides
//    - Helps identify if we're getting filled at bad prices
//    - Updated incrementally as trades arrive
//
// 5. ONE-SIDED FLOW DETECTION
//    - Identifies when flow is >80% in one direction
//    - Only blocks orders when one-sided AND high confidence (>70%)
//    - Prevents getting run over by momentum
//
// 6. IMPROVED ORDER PLACEMENT LOGIC
//    - Checks adverse selection ratio before placing orders
//    - Blocks if adverse ratio > 70%
//    - More lenient thresholds for reducing positions
//    - Better handling of edge cases
// ============================================

const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone)]
struct SimulatedOrder {
    oid: u64,
    price: f64,
    size: f64,
    is_buy: bool,
    timestamp: u64,
}

#[derive(Debug)]
struct RiskLimits {
    max_position_size: f64,
    max_loss_per_day: f64,
    max_drawdown_pct: f64,
    startup_grace_period_ms: u64,
    max_margin_usage_pct: f64,
    liquidation_buffer_pct: f64,
    min_expected_value_bps: f64,
}

#[derive(Debug, Clone)]
struct TradeEvent {
    timestamp: u64,
    price: f64,
    size: f64,
    is_buy: bool,
    is_aggressive: bool,
}

/// Improved multi-timeframe flow analyzer with adverse selection tracking
#[derive(Debug)]
struct ImprovedFlowAnalyzer {
    trade_events: VecDeque<TradeEvent>,
    window_ms: u64,
    
    // Multi-timeframe analysis
    short_term_ms: u64,  // 10 seconds
    medium_term_ms: u64, // 60 seconds  
    long_term_ms: u64,   // 300 seconds
    
    // Volume-weighted metrics
    vwap_buy: f64,
    vwap_sell: f64,
    total_buy_volume: f64,
    total_sell_volume: f64,
    
    // Adverse selection tracking
    trades_against_us: u32,
    trades_favorable: u32,
    cumulative_slippage: f64,
    
    // Legacy baseline for compatibility
    baseline_lambda: f64,
}

impl ImprovedFlowAnalyzer {
    fn new(window_seconds: u64) -> Self {
        Self {
            trade_events: VecDeque::new(),
            window_ms: window_seconds * 1000,
            short_term_ms: 10_000,
            medium_term_ms: 60_000,
            long_term_ms: 300_000,
            vwap_buy: 0.0,
            vwap_sell: 0.0,
            total_buy_volume: 0.0,
            total_sell_volume: 0.0,
            trades_against_us: 0,
            trades_favorable: 0,
            cumulative_slippage: 0.0,
            baseline_lambda: 0.0,
        }
    }
    
    fn add_trade(&mut self, event: TradeEvent, our_bid: Option<f64>, our_ask: Option<f64>) {
        info!("📊 Trade added: {:.2} @ ${:.4} (Buy: {}, Aggressive: {})", 
              event.size, event.price, event.is_buy, event.is_aggressive);
        
        // Track if trade was adverse to us
        if let Some(bid) = our_bid {
            if event.price <= bid && !event.is_buy {
                // Someone hit our bid then price went down - adverse
                self.trades_against_us += 1;
                self.cumulative_slippage += (bid - event.price).abs();
                info!("⚠️  Adverse trade detected against our bid");
            } else if event.price > bid {
                self.trades_favorable += 1;
            }
        }
        
        if let Some(ask) = our_ask {
            if event.price >= ask && event.is_buy {
                // Someone lifted our offer then price went up - adverse
                self.trades_against_us += 1;
                self.cumulative_slippage += (event.price - ask).abs();
                info!("⚠️  Adverse trade detected against our ask");
            } else if event.price < ask {
                self.trades_favorable += 1;
            }
        }
        
        let timestamp = event.timestamp;
        
        // Update VWAP
        if event.is_buy {
            let old_weight = self.total_buy_volume;
            let new_weight = event.size;
            self.total_buy_volume += new_weight;
            
            if self.total_buy_volume > 0.0 {
                self.vwap_buy = (self.vwap_buy * old_weight + event.price * new_weight) 
                                / self.total_buy_volume;
            }
        } else {
            let old_weight = self.total_sell_volume;
            let new_weight = event.size;
            self.total_sell_volume += new_weight;
            
            if self.total_sell_volume > 0.0 {
                self.vwap_sell = (self.vwap_sell * old_weight + event.price * new_weight)
                                 / self.total_sell_volume;
            }
        }
        
        self.trade_events.push_back(event);
        
        // Cleanup old events
        let cutoff = timestamp.saturating_sub(self.window_ms);
        while let Some(e) = self.trade_events.front() {
            if e.timestamp < cutoff {
                // Remove from VWAP calculation too
                if e.is_buy {
                    self.total_buy_volume = (self.total_buy_volume - e.size).max(0.0);
                } else {
                    self.total_sell_volume = (self.total_sell_volume - e.size).max(0.0);
                }
                self.trade_events.pop_front();
            } else {
                break;
            }
        }
        
        // Update baseline lambda for legacy compatibility
        if self.trade_events.len() >= 2 {
            let time_span = (timestamp - self.trade_events.front().unwrap().timestamp) as f64 / 1000.0;
            if time_span > 0.0 {
                let current_rate = self.trade_events.len() as f64 / time_span;
                if self.baseline_lambda < EPSILON {
                    self.baseline_lambda = current_rate;
                } else {
                    self.baseline_lambda = 0.95 * self.baseline_lambda + 0.05 * current_rate;
                }
            }
        }
    }
    
    fn calculate_multi_timeframe_toxicity(&self) -> (f64, f64, f64) {
        let now = self.trade_events.back()
            .map(|e| e.timestamp)
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            });
            
        // Short-term (10s)
        let short_cutoff = now.saturating_sub(self.short_term_ms);
        let short_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= short_cutoff)
            .collect();
            
        let short_toxicity = self.calculate_toxicity_for_trades(&short_trades);
        
        // Medium-term (60s)
        let medium_cutoff = now.saturating_sub(self.medium_term_ms);
        let medium_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= medium_cutoff)
            .collect();
            
        let medium_toxicity = self.calculate_toxicity_for_trades(&medium_trades);
        
        // Long-term (300s)
        let long_cutoff = now.saturating_sub(self.long_term_ms);
        let long_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= long_cutoff)
            .collect();
        let long_toxicity = self.calculate_toxicity_for_trades(&long_trades);
        
        (short_toxicity, medium_toxicity, long_toxicity)
    }
    
    fn calculate_toxicity_for_trades(&self, trades: &[&TradeEvent]) -> f64 {
        if trades.len() < 5 {  // INCREASED from 3 to 5 - require more data
            return 0.0;
        }
        
        // 1. Check for momentum (consecutive trades in same direction)
        let mut consecutive_buys = 0;
        let mut consecutive_sells = 0;
        let mut max_consecutive_buys = 0;
        let mut max_consecutive_sells = 0;
        let mut last_was_buy = false;
        
        for trade in trades {
            if trade.is_buy {
                if last_was_buy {
                    consecutive_buys += 1;
                } else {
                    max_consecutive_sells = max_consecutive_sells.max(consecutive_sells);
                    consecutive_sells = 0;
                    consecutive_buys = 1;
                }
                last_was_buy = true;
            } else {
                if !last_was_buy {
                    consecutive_sells += 1;
                } else {
                    max_consecutive_buys = max_consecutive_buys.max(consecutive_buys);
                    consecutive_buys = 0;
                    consecutive_sells = 1;
                }
                last_was_buy = false;
            }
        }
        
        max_consecutive_buys = max_consecutive_buys.max(consecutive_buys);
        max_consecutive_sells = max_consecutive_sells.max(consecutive_sells);
        
        // LESS SENSITIVE: Require 8 consecutive instead of 5 to be toxic
        let momentum_score = (max_consecutive_buys.max(max_consecutive_sells) as f64 / 8.0).min(1.0);
        
        // 2. Volume surge detection - LESS SENSITIVE
        let avg_size = trades.iter().map(|t| t.size).sum::<f64>() / trades.len() as f64;
        let recent_size = trades.last().map(|t| t.size).unwrap_or(avg_size);
        let volume_surge = ((recent_size / avg_size.max(0.1)) / 5.0).min(1.0); // INCREASED from 3.0 to 5.0
        
        // 3. Price impact - LESS SENSITIVE
        if trades.len() >= 2 {
            let start_price = trades[0].price;
            let end_price = trades[trades.len() - 1].price;
            let price_move = ((end_price - start_price) / start_price).abs();
            let price_impact = ((price_move * 10000.0) / 10.0).min(1.0); // INCREASED from 5 bps to 10 bps threshold
            
            // Combine signals - REDUCE weight of momentum
            return momentum_score * 0.3 + volume_surge * 0.2 + price_impact * 0.5;
        }
        
        momentum_score * 0.4 + volume_surge * 0.6
    }

    fn get_trade_count(&self) -> usize {
        self.trade_events.len()
    }

    fn get_toxicity_probability(&self, _lookback_seconds: f64) -> f64 {
        // Use multi-timeframe analysis instead
        let (short, medium, long) = self.calculate_multi_timeframe_toxicity();
        
        // Weight recent toxicity more heavily
        let weighted_toxicity = short * 0.5 + medium * 0.3 + long * 0.2;
        
        // Factor in adverse selection ratio
        let total_trades = self.trades_against_us + self.trades_favorable;
        if total_trades > 10 {
            let adverse_ratio = self.trades_against_us as f64 / total_trades as f64;
            return weighted_toxicity * 0.7 + adverse_ratio * 0.3;
        }
        
        weighted_toxicity
    }
    
    fn get_improved_directional_flow(&self, window_seconds: f64) -> (f64, f64, bool) {
        if self.trade_events.is_empty() {
            return (0.5, 0.0, false);
        }
        
        let now = self.trade_events.back().unwrap().timestamp;
        let cutoff = now.saturating_sub((window_seconds * 1000.0) as u64);
        
        let recent_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= cutoff && e.is_aggressive)
            .collect();
        
        if recent_trades.is_empty() {
            return (0.5, 0.0, false);
        }
        
        // Volume-weighted instead of count-based
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        
        for trade in &recent_trades {
            if trade.is_buy {
                buy_volume += trade.size;
            } else {
                sell_volume += trade.size;
            }
        }
        
        // Add meaningful prior (10% of total volume, not fixed)
        let total = buy_volume + sell_volume;
        let prior = total * 0.1;
        buy_volume += prior;
        sell_volume += prior;
        
        let buy_ratio = buy_volume / (buy_volume + sell_volume);
        
        // Check for one-sided flow - MORE STRICT THRESHOLD
        let is_one_sided = buy_ratio > 0.85 || buy_ratio < 0.15;  // INCREASED from 0.8/0.2 to 0.85/0.15
        
        // Statistical confidence based on sample size and volume
        let sample_confidence = (recent_trades.len() as f64 / 20.0).min(1.0);
        let volume_confidence = (total / 100.0).min(1.0);
        let confidence = (sample_confidence + volume_confidence) / 2.0;
        
        (buy_ratio, confidence, is_one_sided)
    }

    fn get_directional_toxicity(&self, lookback_seconds: f64) -> (f64, f64) {
        let (buy_ratio, confidence, _) = self.get_improved_directional_flow(lookback_seconds);
        (buy_ratio, confidence)
    }

    fn get_fill_probability(&self, our_price: f64, is_bid: bool, current_mid: f64) -> f64 {
        if self.trade_events.is_empty() || self.baseline_lambda < EPSILON {
            return 0.5;
        }

        let distance_bps = ((our_price - current_mid).abs() / current_mid) * 10000.0;

        let lookback_seconds = 30.0;
        let now = self.trade_events.back().unwrap().timestamp;
        let cutoff = now.saturating_sub((lookback_seconds * 1000.0) as u64);

        let recent_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= cutoff)
            .collect();

        if recent_trades.len() < 3 {
            // More generous fallback formula
            return (-distance_bps / 100.0).exp();
        }

        let hits = recent_trades.iter()
            .filter(|e| {
                if is_bid {
                    e.price <= our_price && !e.is_buy
                } else {
                    e.price >= our_price && e.is_buy
                }
            })
            .count();

        let alpha = 0.5;
        (hits as f64 + alpha) / (recent_trades.len() as f64 + 2.0 * alpha)
    }
    
    fn get_adverse_selection_ratio(&self) -> f64 {
        let total = self.trades_against_us + self.trades_favorable;
        if total < 5 {
            return 0.5; // Not enough data
        }
        self.trades_against_us as f64 / total as f64
    }
}

#[derive(Debug)]
struct BayesianStateEstimator {
    prob_trending: f64,
    prob_mean_reverting: f64,
    prob_volatile: f64,
    prob_toxic_flow: f64,
    observation_count: u32,
}

impl BayesianStateEstimator {
    fn new() -> Self {
        Self {
            prob_trending: 0.25,
            prob_mean_reverting: 0.5,
            prob_volatile: 0.15,
            prob_toxic_flow: 0.1,
            observation_count: 0,
        }
    }

    fn update(
        &mut self,
        price_change_bps: f64,
        volatility: f64,
        flow_toxicity: f64,
        flow_imbalance: f64,
    ) {
        self.observation_count += 1;
        
        let alpha = (10.0 / (10.0 + self.observation_count as f64)).max(0.05);

        let trending_evidence = (price_change_bps.abs() / 10.0).min(1.0) * 
                                (1.0 - volatility).max(0.0);
        self.prob_trending = (1.0 - alpha) * self.prob_trending + alpha * trending_evidence;

        let mean_revert_evidence = if price_change_bps.abs() > 20.0 && volatility < 0.02 {
            0.7
        } else if price_change_bps.abs() < 5.0 {
            0.6
        } else {
            0.3
        };
        self.prob_mean_reverting = (1.0 - alpha) * self.prob_mean_reverting + 
                                   alpha * mean_revert_evidence;

        let volatile_evidence = (volatility / 0.05).min(1.0);
        self.prob_volatile = (1.0 - alpha) * self.prob_volatile + alpha * volatile_evidence;

        let toxic_evidence = flow_toxicity + flow_imbalance * 0.1;
        self.prob_toxic_flow = (1.0 - alpha) * self.prob_toxic_flow + alpha * toxic_evidence;

        let regime_sum = self.prob_trending + self.prob_mean_reverting + self.prob_volatile;
        if regime_sum > EPSILON {
            let scale = 1.0 / regime_sum;
            self.prob_trending *= scale;
            self.prob_mean_reverting *= scale;
            self.prob_volatile *= scale;
        }
    }

    fn get_confidence(&self) -> f64 {
        (self.observation_count as f64 / (self.observation_count as f64 + 20.0)).min(0.95)
    }
}

#[derive(Debug)]
struct VolatilityTracker {
    price_samples: VecDeque<(u64, f64)>,
    window_ms: u64,
}

impl VolatilityTracker {
    fn new(window_seconds: u64) -> Self {
        Self {
            price_samples: VecDeque::new(),
            window_ms: window_seconds * 1000,
        }
    }

    fn add_sample(&mut self, timestamp: u64, price: f64) {
        self.price_samples.push_back((timestamp, price));

        let cutoff = timestamp.saturating_sub(self.window_ms);
        while let Some((ts, _)) = self.price_samples.front() {
            if *ts < cutoff {
                self.price_samples.pop_front();
            } else {
                break;
            }
        }
    }

    fn get_volatility(&self) -> f64 {
        if self.price_samples.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self
            .price_samples
            .iter()
            .zip(self.price_samples.iter().skip(1))
            .map(|((_, p1), (_, p2))| p2 / p1 - 1.0)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / returns.len() as f64;

        variance.sqrt()
    }

    fn get_recent_price_change_bps(&self, lookback_seconds: u64) -> f64 {
        if self.price_samples.len() < 2 {
            return 0.0;
        }

        let now = self.price_samples.back().unwrap().0;
        let cutoff = now.saturating_sub(lookback_seconds * 1000);

        let start_price = self.price_samples.iter()
            .find(|(ts, _)| *ts >= cutoff)
            .map(|(_, p)| *p)
            .unwrap_or_else(|| self.price_samples.front().unwrap().1);

        let end_price = self.price_samples.back().unwrap().1;

        ((end_price - start_price) / start_price) * 10000.0
    }
}

#[derive(Debug)]
struct MarketMakerState {
    initial_capital: f64,
    current_capital: f64,
    leverage: f64,
    margin_used: f64,
    liquidation_price: f64,
    
    current_bid: f64,
    current_ask: f64,
    mid_price: f64,
    spread: f64,
    bid_depth: f64,
    ask_depth: f64,

    our_bid_order: Option<SimulatedOrder>,
    our_ask_order: Option<SimulatedOrder>,

    position: f64,
    average_entry: f64,

    realized_pnl: f64,
    unrealized_pnl: f64,
    total_fills: u32,
    buy_fills: u32,
    sell_fills: u32,
    profitable_fills: u32,
    losing_fills: u32,

    maker_fee_rate: f64,
    total_fees_paid: f64,

    max_buy_size: f64,
    max_sell_size: f64,

    next_oid: u64,

    flow_analyzer: ImprovedFlowAnalyzer,
    bayesian_estimator: BayesianStateEstimator,
    volatility_tracker: VolatilityTracker,

    session_start_time: u64,
    peak_pnl: f64,
    
    last_update_ts: u64,
}

impl MarketMakerState {
    fn new(initial_capital: f64, leverage: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            initial_capital,
            current_capital: initial_capital,
            leverage,
            margin_used: 0.0,
            liquidation_price: 0.0,
            current_bid: 0.0,
            current_ask: 0.0,
            mid_price: 0.0,
            spread: 0.0,
            bid_depth: 0.0,
            ask_depth: 0.0,
            our_bid_order: None,
            our_ask_order: None,
            position: 0.0,
            average_entry: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            total_fills: 0,
            buy_fills: 0,
            sell_fills: 0,
            profitable_fills: 0,
            losing_fills: 0,
            maker_fee_rate: 0.00015,
            total_fees_paid: 0.0,
            max_buy_size: 0.0,
            max_sell_size: 0.0,
            next_oid: 1,
            flow_analyzer: ImprovedFlowAnalyzer::new(300),
            bayesian_estimator: BayesianStateEstimator::new(),
            volatility_tracker: VolatilityTracker::new(300),
            session_start_time: now,
            peak_pnl: 0.0,
            last_update_ts: now,
        }
    }

    fn update_bbo(&mut self, bid: f64, ask: f64, bid_sz: f64, ask_sz: f64) {
        self.current_bid = bid;
        self.current_ask = ask;
        self.mid_price = (bid + ask) / 2.0;
        self.spread = ask - bid;
        self.bid_depth = bid_sz;
        self.ask_depth = ask_sz;

        // Log order book depth
        info!("📈 Order Book: Bid ${:.4} ({:.1}) / Ask ${:.4} ({:.1}) | Spread: {:.2} bps", 
              bid, bid_sz, ask, ask_sz, (self.spread / self.mid_price) * 10000.0);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        self.volatility_tracker.add_sample(timestamp, self.mid_price);

        let price_change_bps = self.volatility_tracker.get_recent_price_change_bps(30);
        let volatility = self.volatility_tracker.get_volatility();
        let flow_toxicity = self.flow_analyzer.get_toxicity_probability(10.0);
        let (buy_ratio, _) = self.flow_analyzer.get_directional_toxicity(30.0);
        let flow_imbalance = (buy_ratio - 0.5).abs() * 2.0;

        self.bayesian_estimator.update(price_change_bps, volatility, flow_toxicity, flow_imbalance);

        if self.position != 0.0 {
            self.unrealized_pnl = (self.mid_price - self.average_entry) * self.position;
        }

        self.update_margin();
        self.update_capital();

        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl > self.peak_pnl {
            self.peak_pnl = total_pnl;
        }

        self.last_update_ts = timestamp;
    }

    fn update_margin(&mut self) {
        if self.position != 0.0 {
            let position_size = self.position.abs();
            let position_notional = position_size * self.average_entry;
            
            self.margin_used = position_notional / self.leverage;
            
            let account_capital = self.initial_capital + self.realized_pnl - self.total_fees_paid;
            
            if self.position > 0.0 {
                let numerator = account_capital - self.average_entry * position_size;
                let denominator = position_size * ((1.0 / (2.0 * self.leverage)) - 1.0);
                
                if denominator.abs() < 1e-10 {
                    self.liquidation_price = 0.0;
                } else {
                    self.liquidation_price = numerator / denominator;
                }
            } else {
                let numerator = account_capital + self.average_entry * position_size;
                let denominator = position_size * (1.0 + 1.0 / (2.0 * self.leverage));
                self.liquidation_price = numerator / denominator;
            }
            
            self.liquidation_price = self.liquidation_price.max(0.0);
        } else {
            self.margin_used = 0.0;
            self.liquidation_price = 0.0;
        }
    }

    fn update_capital(&mut self) {
        self.current_capital = self.initial_capital + self.realized_pnl - self.total_fees_paid;
    }

    fn get_available_margin(&self) -> f64 {
        let equity = self.current_capital + self.unrealized_pnl;
        let used_margin = self.margin_used;
        
        let position_notional = if self.position != 0.0 {
            self.position.abs() * self.mid_price
        } else {
            0.0
        };
        
        let transfer_requirement = (position_notional * 0.1).max(used_margin);
        
        (equity - transfer_requirement).max(0.0)
    }

    fn is_near_liquidation(&self, buffer_pct: f64) -> bool {
        if self.position == 0.0 || self.liquidation_price == 0.0 {
            return false;
        }

        let buffer_multiplier = 1.0 + (buffer_pct / 100.0);

        if self.position > 0.0 {
            self.mid_price < self.liquidation_price * buffer_multiplier
        } else {
            self.mid_price > self.liquidation_price / buffer_multiplier
        }
    }

    fn update_capacity(&mut self, max_buy: f64, max_sell: f64) {
        self.max_buy_size = max_buy;
        self.max_sell_size = max_sell;
    }

    fn place_simulated_order(&mut self, is_buy: bool, price: f64, size: f64) -> SimulatedOrder {
        let oid = self.next_oid;
        self.next_oid += 1;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        SimulatedOrder {
            oid,
            price,
            size,
            is_buy,
            timestamp,
        }
    }

    fn record_trade(&mut self, trade_price: f64, trade_size: f64, is_aggressive_buy: bool) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let our_bid = self.our_bid_order.as_ref().map(|o| o.price);
        let our_ask = self.our_ask_order.as_ref().map(|o| o.price);

        self.flow_analyzer.add_trade(
            TradeEvent {
                timestamp,
                price: trade_price,
                size: trade_size,
                is_buy: is_aggressive_buy,
                is_aggressive: true,
            },
            our_bid,
            our_ask,
        );
    }

    fn check_fills(&mut self, trade_price: f64, is_aggressive_buy: bool, size: f64) -> Vec<String> {
        self.record_trade(trade_price, size, is_aggressive_buy);
        
        let mut fills = Vec::new();

        if let Some(bid_order) = &self.our_bid_order {
            if trade_price <= bid_order.price {
                fills.push(self.execute_fill(bid_order.clone(), trade_price));
                self.our_bid_order = None;
            }
        }

        if let Some(ask_order) = &self.our_ask_order {
            if trade_price >= ask_order.price {
                fills.push(self.execute_fill(ask_order.clone(), trade_price));
                self.our_ask_order = None;
            }
        }

        fills
    }

    fn execute_fill(&mut self, order: SimulatedOrder, fill_price: f64) -> String {
        self.total_fills += 1;

        let notional = fill_price * order.size;
        let fee = notional * self.maker_fee_rate;
        self.total_fees_paid += fee;

        let fill_msg = if order.is_buy {
            self.buy_fills += 1;

            let old_position = self.position;
            let old_avg_entry = self.average_entry;

            self.position += order.size;

            if old_position < 0.0 {
                let closed_size = old_position.abs().min(order.size);
                let pnl = (old_avg_entry - fill_price) * closed_size - fee;
                self.realized_pnl += pnl;

                if pnl > 0.0 {
                    self.profitable_fills += 1;
                } else {
                    self.losing_fills += 1;
                }

                if self.position > 0.0 {
                    self.average_entry = fill_price;
                } else if self.position == 0.0 {
                    self.average_entry = 0.0;
                }
            } 
            else if old_position >= 0.0 {
                self.average_entry =
                    (old_avg_entry * old_position + fill_price * order.size) / self.position;
            }

            format!(
                "✅ BUY FILL: {:.2} @ ${:.4} (OID: {}) | Fee: ${:.4}",
                order.size, fill_price, order.oid, fee
            )
        } else {
            self.sell_fills += 1;

            let old_position = self.position;
            let old_avg_entry = self.average_entry;

            self.position -= order.size;

            if old_position > 0.0 {
                let closed_size = old_position.min(order.size);
                let pnl = (fill_price - old_avg_entry) * closed_size - fee;
                self.realized_pnl += pnl;

                if pnl > 0.0 {
                    self.profitable_fills += 1;
                } else {
                    self.losing_fills += 1;
                }

                if self.position < 0.0 {
                    self.average_entry = fill_price;
                } else if self.position == 0.0 {
                    self.average_entry = 0.0;
                }
            } 
            else if old_position <= 0.0 {
                self.average_entry = (old_avg_entry * old_position.abs() + fill_price * order.size)
                    / self.position.abs();
            }

            format!(
                "✅ SELL FILL: {:.2} @ ${:.4} (OID: {}) | Fee: ${:.4}",
                order.size, fill_price, order.oid, fee
            )
        };

        fill_msg
    }

    fn get_current_drawdown_pct(&self) -> f64 {
        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        
        if self.peak_pnl < 5.0 {
            if total_pnl < 0.0 {
                return total_pnl.abs();
            }
            return 0.0;
        }
        
        ((self.peak_pnl - total_pnl) / self.peak_pnl.abs()) * 100.0
    }

    fn check_risk_limits(&self, limits: &RiskLimits) -> (bool, Vec<String>) {
        let mut violations = Vec::new();
        let mut is_safe = true;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let time_elapsed = now.saturating_sub(self.session_start_time);
        let in_grace_period = time_elapsed < limits.startup_grace_period_ms;

        let equity = self.current_capital + self.unrealized_pnl;
        
        if self.position != 0.0 {
            let position_notional = self.position.abs() * self.mid_price;
            let maintenance_margin_ratio = 1.0 / (2.0 * self.leverage);
            let required_maintenance = position_notional * maintenance_margin_ratio;
            
            if equity < required_maintenance {
                violations.push(format!(
                    "⚠️  LIQUIDATION IMMINENT! Equity ${:.2} < Maintenance Margin ${:.2}",
                    equity, required_maintenance
                ));
                is_safe = false;
            }
        }

        if self.is_near_liquidation(limits.liquidation_buffer_pct) {
            violations.push(format!(
                "Near liquidation! Current: ${:.4}, Liquidation: ${:.4}",
                self.mid_price, self.liquidation_price
            ));
            is_safe = false;
        }

        if equity > 0.0 {
            let margin_usage_pct = (self.margin_used / equity) * 100.0;
            if margin_usage_pct > limits.max_margin_usage_pct {
                violations.push(format!(
                    "Margin usage too high: {:.1}% > {:.1}%",
                    margin_usage_pct, limits.max_margin_usage_pct
                ));
                is_safe = false;
            }
        }

        if self.position.abs() > limits.max_position_size {
            violations.push(format!(
                "Position limit breached: {:.2} > {:.2}",
                self.position.abs(),
                limits.max_position_size
            ));
            is_safe = false;
        }

        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl < -limits.max_loss_per_day {
            violations.push(format!(
                "Daily loss limit breached: ${:.2} < -${:.2}",
                total_pnl, limits.max_loss_per_day
            ));
            is_safe = false;
        }

        if !in_grace_period {
            let drawdown_value = self.get_current_drawdown_pct();
            
            if self.peak_pnl >= 5.0 {
                if drawdown_value > limits.max_drawdown_pct {
                    violations.push(format!(
                        "Drawdown limit breached: {:.2}% > {:.2}%",
                        drawdown_value, limits.max_drawdown_pct
                    ));
                    is_safe = false;
                }
            } else {
                if drawdown_value > limits.max_loss_per_day / 2.0 {
                    violations.push(format!(
                        "Early-stage loss limit: ${:.2} > ${:.2}",
                        drawdown_value, limits.max_loss_per_day / 2.0
                    ));
                    is_safe = false;
                }
            }
        }

        (is_safe, violations)
    }

    fn print_status(&self) {
        let total_pnl = self.realized_pnl + self.unrealized_pnl;
        let net_pnl = total_pnl - self.total_fees_paid;
        let volatility = self.volatility_tracker.get_volatility() * 100.0;
        let drawdown = self.get_current_drawdown_pct();

        let toxicity_prob = self.flow_analyzer.get_toxicity_probability(10.0);
        let (buy_ratio, dir_confidence) = self.flow_analyzer.get_directional_toxicity(30.0);
        let (_, _, is_one_sided) = self.flow_analyzer.get_improved_directional_flow(30.0);
        let (short_tox, medium_tox, long_tox) = self.flow_analyzer.calculate_multi_timeframe_toxicity();
        let adverse_ratio = self.flow_analyzer.get_adverse_selection_ratio();
        let bayesian_confidence = self.bayesian_estimator.get_confidence();
        let trade_count = self.flow_analyzer.get_trade_count();

        let equity = self.current_capital + self.unrealized_pnl;
        let margin_usage_pct = if equity > 0.0 {
            (self.margin_used / equity) * 100.0
        } else {
            0.0
        };

        let roe = if self.initial_capital > 0.0 {
            (net_pnl / self.initial_capital) * 100.0
        } else {
            0.0
        };

        let fee_ratio = if self.realized_pnl.abs() > 0.01 {
            (self.total_fees_paid / self.realized_pnl.abs()) * 100.0
        } else {
            0.0
        };

        info!("╔════════════════════════════════════════════════════════╗");
        info!("📊 PROBABILISTIC MM STATUS");
        info!("╠════════════════════════════════════════════════════════╣");
        info!(
            "Market: Bid ${:.4} ({:.1}) / Ask ${:.4} ({:.1}) | Spread: ${:.4} ({:.2} bps)",
            self.current_bid,
            self.bid_depth,
            self.current_ask,
            self.ask_depth,
            self.spread,
            (self.spread / self.mid_price) * 10000.0
        );
        info!("Volatility: {:.3}%", volatility);

        info!("────────────────────────────────────────────────────────");
        info!("🎲 IMPROVED FLOW ANALYSIS (Trades: {})", trade_count);
        info!(
            "Multi-Timeframe Toxicity: 10s={:.1}% | 60s={:.1}% | 300s={:.1}%",
            short_tox * 100.0, medium_tox * 100.0, long_tox * 100.0
        );
        info!(
            "Overall Toxicity: {:.1}% | Adverse Selection Ratio: {:.1}%",
            toxicity_prob * 100.0, adverse_ratio * 100.0
        );
        info!(
            "Directional Flow: Buy {:.1}% (confidence: {:.1}%) {}",
            buy_ratio * 100.0, dir_confidence * 100.0,
            if is_one_sided { "⚠️  ONE-SIDED" } else { "" }
        );
        info!(
            "Bayesian Estimates (confidence: {:.1}%):",
            bayesian_confidence * 100.0
        );
        info!(
            "  • Trending: {:.1}% | Mean-Reverting: {:.1}% | Volatile: {:.1}%",
            self.bayesian_estimator.prob_trending * 100.0,
            self.bayesian_estimator.prob_mean_reverting * 100.0,
            self.bayesian_estimator.prob_volatile * 100.0
        );
        info!("  • Toxic Flow: {:.1}%", self.bayesian_estimator.prob_toxic_flow * 100.0);

        info!("────────────────────────────────────────────────────────");
        info!("💰 CAPITAL & RISK");
        info!(
            "Capital: ${:.2} → ${:.2} | Equity: ${:.2} | ROE: {:.2}%",
            self.initial_capital, self.current_capital, equity, roe
        );
        info!(
            "Margin: ${:.2} / ${:.2} ({:.1}% used) | Leverage: {}x",
            self.margin_used, equity, margin_usage_pct, self.leverage
        );
        if self.position != 0.0 {
            info!(
                "Liquidation Price: ${:.4} | Distance: {:.2}%",
                self.liquidation_price,
                ((self.mid_price - self.liquidation_price) / self.mid_price * 100.0).abs()
            );
        }

        info!("────────────────────────────────────────────────────────");

        if let Some(bid) = &self.our_bid_order {
            let edge = ((self.mid_price - bid.price) / self.mid_price) * 10000.0;
            let fill_prob = self.flow_analyzer.get_fill_probability(bid.price, true, self.mid_price);
            let age = self.last_update_ts.saturating_sub(bid.timestamp) / 1000;
            info!(
                "Our Bid: {:.2} @ ${:.4} (OID: {}, age: {}s) [{:.1} bps edge, {:.1}% fill prob]",
                bid.size, bid.price, bid.oid, age, edge, fill_prob * 100.0
            );
        } else {
            info!("Our Bid: None");
        }

        if let Some(ask) = &self.our_ask_order {
            let edge = ((ask.price - self.mid_price) / self.mid_price) * 10000.0;
            let fill_prob = self.flow_analyzer.get_fill_probability(ask.price, false, self.mid_price);
            let age = self.last_update_ts.saturating_sub(ask.timestamp) / 1000;
            info!(
                "Our Ask: {:.2} @ ${:.4} (OID: {}, age: {}s) [{:.1} bps edge, {:.1}% fill prob]",
                ask.size, ask.price, ask.oid, age, edge, fill_prob * 100.0
            );
        } else {
            info!("Our Ask: None");
        }

        info!("────────────────────────────────────────────────────────");
        info!(
            "Position: {:.2} HYPE @ ${:.4} avg entry",
            self.position, self.average_entry
        );
        info!("Realized PnL: ${:.2}", self.realized_pnl);
        info!("Unrealized PnL: ${:.2}", self.unrealized_pnl);
        info!("Fees Paid: ${:.2} (Fee Ratio: {:.1}%)", self.total_fees_paid, fee_ratio);
        info!("Net PnL: ${:.2}", net_pnl);
        
        if self.peak_pnl >= 5.0 {
            info!("Peak PnL: ${:.2} | Drawdown: {:.2}%", self.peak_pnl, drawdown);
        } else {
            info!("Peak PnL: ${:.2} | Loss: ${:.2}", self.peak_pnl, drawdown);
        }

        info!("────────────────────────────────────────────────────────");
        info!(
            "Fills: {} total ({} buys, {} sells)",
            self.total_fills, self.buy_fills, self.sell_fills
        );
        
        let closed_trades = self.profitable_fills + self.losing_fills;
        if closed_trades > 0 {
            let win_rate = (self.profitable_fills as f64 / closed_trades as f64) * 100.0;
            info!(
                "Closed Trades: {} ({} wins, {} losses) | Win Rate: {:.1}%",
                closed_trades, self.profitable_fills, self.losing_fills, win_rate
            );
        } else {
            info!("Closed Trades: 0 (all fills are opening new positions)");
        }

        if self.total_fills > 0 {
            let pnl_per_fill = net_pnl / self.total_fills as f64;
            info!("Avg PnL per Fill: ${:.3}", pnl_per_fill);
        }

        info!("╚════════════════════════════════════════════════════════╝");
    }
}



fn calculate_minimum_spread(state: &MarketMakerState) -> f64 {
    // Base: Cover fees on both sides + minimum profit
    let fee_coverage = state.maker_fee_rate * 2.0 * 10000.0; // 3 bps for 0.015% fee
    
    // Add buffer for adverse selection based on actual performance
    let adverse_rate = if state.total_fills > 10 {
        let losing_rate = state.losing_fills as f64 / state.total_fills.max(1) as f64;
        losing_rate * 3.0 // More lenient for tight markets
    } else {
        0.3 // Very small initial buffer
    };
    
    // Volatility adjustment - less aggressive
    let volatility = state.volatility_tracker.get_volatility();
    let vol_adjustment = (volatility * 30.0).max(0.2); // Even more lenient
    
    // Inventory risk (when holding position)
    let inventory_cost = (state.position.abs() / 30.0) * 0.8; // Reduced
    
    // CRITICAL: For ultra-tight spread markets like HYPE (0.3-1.5 bps)
    // We need to be VERY aggressive or we'll never place asks
    let min_spread = fee_coverage + adverse_rate + vol_adjustment + inventory_cost + 0.2;
    
    // ABSOLUTE MINIMUM: 2.5 bps total spread to be profitable
    min_spread.max(2.5)
}

// ============================================
// FIX 3: POSITION MANAGEMENT
// ============================================

fn calculate_position_limits(state: &MarketMakerState, base_max: f64) -> (f64, f64) {
    // Dynamic position limits based on P&L
    let pnl_ratio = (state.realized_pnl + state.unrealized_pnl) / state.initial_capital;
    
    // Reduce position limits when losing
    let scaling_factor = if pnl_ratio < -0.02 {
        info!("📉 Reducing position limits to 50% due to -{:.1}% P&L", pnl_ratio.abs() * 100.0);
        0.5 // Half size when down 2%
    } else if pnl_ratio < -0.01 {
        info!("📉 Reducing position limits to 75% due to -{:.1}% P&L", pnl_ratio.abs() * 100.0);
        0.75 // 75% size when down 1%
    } else {
        1.0
    };
    
    let max_long = base_max * scaling_factor;
    let max_short = -base_max * scaling_factor;
    
    (max_long, max_short)
}



// ============================================
// FIX 4: SMARTER ORDER PLACEMENT
// ============================================

fn calculate_smart_quotes(
    state: &MarketMakerState,
    base_size: f64,
    max_position: f64,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
    let (short_tox, medium_tox, long_tox) = state.flow_analyzer.calculate_multi_timeframe_toxicity();
    
    // Weight recent toxicity more, but also discount long-term less
    let weighted_toxicity = short_tox * 0.6 + medium_tox * 0.3 + long_tox * 0.1; // Increased short-term weight
    
    // MUCH MORE LENIENT: Only stay out if EXTREMELY toxic (was 0.8, now 0.95)
    if weighted_toxicity > 0.95 {
        info!("🚨 Flow too toxic ({:.1}%), staying out", weighted_toxicity * 100.0);
        return (None, None);
    }
    
    let min_spread = calculate_minimum_spread(state);
    let (buy_ratio, confidence, is_one_sided) = state.flow_analyzer.get_improved_directional_flow(30.0);
    
    // Adjust spread based on toxicity - MUCH LESS AGGRESSIVE MULTIPLIER
    let spread_multiplier = 1.0 + weighted_toxicity * 0.8; // REDUCED from 2.0x to 0.8x - max 1.8x instead of 3x
    let adjusted_spread = min_spread * spread_multiplier;
    
    // Position skew
    let position_ratio = state.position / max_position;
    let position_skew = position_ratio * 5.0; // 5 bps per full position
    
    // Calculate prices - SYMMETRIC PRICING
    let half_spread = adjusted_spread / 2.0;
    
    // Skew quotes based on inventory and flow
    // FIXED: Ensure both sides get similar edge to place both orders
    let bid_adjustment = if state.position > max_position * 0.5 {
        // Already long, make bid less aggressive
        half_spread + position_skew.abs() + (buy_ratio - 0.5) * 8.0 * confidence
    } else {
        half_spread - position_skew + (buy_ratio - 0.5) * 4.0 * confidence
    };
    
    let ask_adjustment = if state.position < -max_position * 0.5 {
        // Already short, make ask less aggressive  
        half_spread + position_skew.abs() - (buy_ratio - 0.5) * 8.0 * confidence
    } else {
        half_spread + position_skew - (buy_ratio - 0.5) * 4.0 * confidence
    };
    
    let bid_price = state.mid_price * (1.0 - bid_adjustment / 10000.0);
    let ask_price = state.mid_price * (1.0 + ask_adjustment / 10000.0);
    
    // Size based on confidence and position - MORE AGGRESSIVE
    let bid_size = if state.position > max_position * 0.67 {
        base_size * 0.5 // INCREASED from 0.3 to 0.5 - less reduction when long
    } else if is_one_sided && buy_ratio > 0.8 {  // INCREASED threshold from 0.7 to 0.8
        base_size * 0.7 // INCREASED from 0.5 to 0.7 - less reduction in buy pressure
    } else {
        base_size * (1.0 - weighted_toxicity * 0.5)  // Only reduce by 50% of toxicity
    };
    
    let ask_size = if state.position < -max_position * 0.67 {
        base_size * 0.5 // INCREASED from 0.3 to 0.5 - less reduction when short
    } else if is_one_sided && buy_ratio < 0.2 {  // DECREASED threshold from 0.3 to 0.2
        base_size * 0.7 // INCREASED from 0.5 to 0.7 - less reduction in sell pressure
    } else {
        base_size * (1.0 - weighted_toxicity * 0.5)  // Only reduce by 50% of toxicity
    };
    
    let bid = if bid_size >= 0.1 && state.position < max_position * 0.83 {
        Some((bid_price, bid_size))
    } else {
        None
    };
    
    let ask = if ask_size >= 0.1 && state.position > -max_position * 0.83 {
        Some((ask_price, ask_size))
    } else {
        None
    };
    
    (bid, ask)
}

fn should_place_order(
    state: &MarketMakerState,
    is_buy: bool,
    size: f64,
    price: f64,
    expected_value_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) -> (bool, String) {
    let is_reducing_position = (is_buy && state.position < -5.0) || (!is_buy && state.position > 5.0);
    
    // FIXED: Make EV threshold much more lenient, especially for reducing positions
    let ev_threshold = if is_reducing_position {
        risk_limits.min_expected_value_bps * 0.1  // SUPER lenient when reducing
    } else {
        risk_limits.min_expected_value_bps
    };
    
    // CRITICAL FIX: In ultra-tight markets, ANY positive edge is good
    // Don't be too picky or we'll never place asks
    if expected_value_bps < ev_threshold {
        // Still allow if edge is positive and we're trying to balance inventory
        if expected_value_bps > 0.0 && state.position.abs() > max_position * 0.3 {
            // Allow it - we need to manage inventory
        } else {
            return (false, format!("EV too low: {:.3} < {:.3} bps", expected_value_bps, ev_threshold));
        }
    }

    let toxic_prob = state.bayesian_estimator.prob_toxic_flow;
    if toxic_prob > 0.7 {
        return (false, format!("Toxic flow probability too high: {:.1}%", toxic_prob * 100.0));
    }

    // Use improved directional flow detection
    let (buy_ratio, dir_confidence, is_one_sided) = state.flow_analyzer.get_improved_directional_flow(30.0);
    
    // Only block orders on truly one-sided flow with high confidence
    if is_one_sided && dir_confidence > 0.7 {
        if is_buy && buy_ratio > 0.85 {
            return (false, format!("Strong one-sided buying pressure: {:.1}%", buy_ratio * 100.0));
        }
        if !is_buy && buy_ratio < 0.15 {
            return (false, format!("Strong one-sided selling pressure: {:.1}%", buy_ratio * 100.0));
        }
    }
    
    // Additional check: adverse selection ratio
    let adverse_ratio = state.flow_analyzer.get_adverse_selection_ratio();
    if adverse_ratio > 0.7 {
        return (false, format!("High adverse selection: {:.1}%", adverse_ratio * 100.0));
    }

    let position_ratio = state.position / max_position;
    if is_buy && position_ratio > 0.8 && !is_reducing_position {
        return (false, "Near max long position".to_string());
    }
    if !is_buy && position_ratio < -0.8 && !is_reducing_position {
        return (false, "Near max short position".to_string());
    }

    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        if is_buy && state.position >= 0.0 {
            return (false, "Near liquidation - can only reduce".to_string());
        }
        if !is_buy && state.position <= 0.0 {
            return (false, "Near liquidation - can only reduce".to_string());
        }
    }

    let notional = price * size;
    let required_margin = notional / state.leverage;
    let available_margin = state.get_available_margin();
    
    if required_margin > available_margin * 0.95 {
        return (false, "Insufficient margin".to_string());
    }

    if is_buy && state.position + size > max_position {
        return (false, "Would exceed max position".to_string());
    }
    if !is_buy && state.position - size < -max_position {
        return (false, "Would exceed max position".to_string());
    }

    (true, format!("EV: {:.2} bps", expected_value_bps))
}

fn should_update_orders(
    state: &MarketMakerState,
    last_mid: f64,
    _threshold_pct: f64,  // Kept for API compatibility but not used
) -> bool {
    if state.our_bid_order.is_none() || state.our_ask_order.is_none() {
        return true;
    }

    // DYNAMIC THRESHOLD: Scale with volatility
    // In volatile markets, update more frequently (lower threshold)
    // In calm markets, update less frequently (higher threshold)
    let volatility = state.volatility_tracker.get_volatility();
    
    // Base threshold: 3 bps for normal volatility (0.01 or 1%)
    // Scale: 0.5x to 2x based on volatility
    // Low vol (0.005): 6 bps threshold (more patient)
    // Normal vol (0.01): 3 bps threshold
    // High vol (0.02): 1.5 bps threshold (more responsive)
    let base_threshold_bps = 3.0;
    let vol_ratio = volatility / 0.01; // Normalize to 1% volatility
    let vol_multiplier = (1.0 / vol_ratio.max(0.5).min(2.0)).max(0.5).min(2.0);
    let dynamic_threshold_bps = base_threshold_bps * vol_multiplier;
    
    if last_mid > 0.0 {
        let price_change_pct = ((state.mid_price - last_mid) / last_mid).abs();
        let price_change_bps = price_change_pct * 10000.0;
        
        if price_change_bps > dynamic_threshold_bps {
            info!("📊 Price moved {:.2} bps > {:.2} bps threshold (vol: {:.3}%)", 
                  price_change_bps, dynamic_threshold_bps, volatility * 100.0);
            return true;
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    if let Some(bid) = &state.our_bid_order {
        // FIXED: For ultra-tight spreads, accept tighter edges
        let bid_edge_bps = ((state.mid_price - bid.price) / state.mid_price) * 10000.0;
        if bid_edge_bps > 150.0 || bid_edge_bps < 0.3 {  // Accept down to 0.3 bps edge
            return true;
        }
        
    // FIXED: Reduce order churn - keep orders longer
    let order_age = now.saturating_sub(bid.timestamp) / 1000;
    if order_age > 60 {  // Only cancel after 1 minute (was 5 minutes)
        info!("🕐 Bid order aged out ({}s)", order_age);
        return true;
    }
}

    if let Some(ask) = &state.our_ask_order {
        // FIXED: For ultra-tight spreads, accept tighter edges
        let ask_edge_bps = ((ask.price - state.mid_price) / state.mid_price) * 10000.0;
        if ask_edge_bps > 150.0 || ask_edge_bps < 0.3 {  // Accept down to 0.3 bps edge
            return true;
        }    // FIXED: Reduce order churn - keep orders longer
    let order_age = now.saturating_sub(ask.timestamp) / 1000;
    if order_age > 60 {  // Only cancel after 1 minute (was 5 minutes)
        info!("🕐 Ask order aged out ({}s)", order_age);
        return true;
    }
}

// FIXED: Don't update too frequently - reduce churn
let time_since_update = now.saturating_sub(state.last_update_ts) as f64 / 1000.0;

if time_since_update > 45.0 {  // Only force update after 45 seconds
    return true;
}    // Relaxed: Increased toxic flow threshold from 0.6 to 0.8
    let toxic_prob = state.bayesian_estimator.prob_toxic_flow;
    if toxic_prob > 0.8 {
        return true;
    }

    false
}

fn cancel_simulated_orders(state: &mut MarketMakerState) {
    if let Some(bid) = &state.our_bid_order {
        info!("❌ Cancelled bid order {} @ ${:.4}", bid.oid, bid.price);
        state.our_bid_order = None;
    }

    if let Some(ask) = &state.our_ask_order {
        info!("❌ Cancelled ask order {} @ ${:.4}", ask.oid, ask.price);
        state.our_ask_order = None;
    }
}

fn place_probabilistic_orders(
    state: &mut MarketMakerState,
    base_order_size: f64,
    _base_spread_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) {
    // Apply dynamic position limits based on P&L
    let (max_long, max_short) = calculate_position_limits(state, max_position);
    let effective_max = max_long.min(max_short.abs());
    
    // Use smart quotes calculation for better toxicity handling
    let (bid_quote, ask_quote) = calculate_smart_quotes(state, base_order_size, effective_max);
    
    // Place bid if smart quotes says we should
    if let Some((bid_price, bid_size)) = bid_quote {
        // Round price and size
        let our_bid_price = (bid_price * 10000.0).round() / 10000.0;
        let our_bid_size = (bid_size * 100.0).round() / 100.0;
        
        // Calculate EV for this price
        let bid_fill_prob = state.flow_analyzer.get_fill_probability(our_bid_price, true, state.mid_price);
        let bid_edge_bps = ((state.mid_price - our_bid_price) / state.mid_price) * 10000.0;
        let bid_ev_bps = bid_edge_bps * bid_fill_prob;
        
        let (can_place_bid, bid_reason) = should_place_order(
            state, true, our_bid_size, our_bid_price, bid_ev_bps, effective_max, risk_limits
        );
        
        if can_place_bid && our_bid_size >= 0.01 {
            let bid_order = state.place_simulated_order(true, our_bid_price, our_bid_size);
            info!(
                "🔹 BID: {:.2} @ ${:.4} (OID: {}) | {:.2} bps edge | {:.1}% fill prob | EV: {:.2} bps | {}",
                our_bid_size, our_bid_price, bid_order.oid, bid_edge_bps, bid_fill_prob * 100.0, bid_ev_bps, bid_reason
            );
            state.our_bid_order = Some(bid_order);
        } else {
            info!("⚠️  Skipping bid - {}", bid_reason);
        }
    } else {
        info!("⚠️  No bid: Smart quotes declined (toxicity or position limits)");
    }

    // Place ask if smart quotes says we should
    if let Some((ask_price, ask_size)) = ask_quote {
        // Round price and size
        let our_ask_price = (ask_price * 10000.0).round() / 10000.0;
        let our_ask_size = (ask_size * 100.0).round() / 100.0;
        
        // Calculate EV for this price
        let ask_fill_prob = state.flow_analyzer.get_fill_probability(our_ask_price, false, state.mid_price);
        let ask_edge_bps = ((our_ask_price - state.mid_price) / state.mid_price) * 10000.0;
        let ask_ev_bps = ask_edge_bps * ask_fill_prob;
        
        let (can_place_ask, ask_reason) = should_place_order(
            state, false, our_ask_size, our_ask_price, ask_ev_bps, effective_max, risk_limits
        );
        
        if can_place_ask && our_ask_size >= 0.01 {
            let ask_order = state.place_simulated_order(false, our_ask_price, our_ask_size);
            info!(
                "🔸 ASK: {:.2} @ ${:.4} (OID: {}) | {:.2} bps edge | {:.1}% fill prob | EV: {:.2} bps | {}",
                our_ask_size, our_ask_price, ask_order.oid, ask_edge_bps, ask_fill_prob * 100.0, ask_ev_bps, ask_reason
            );
            state.our_ask_order = Some(ask_order);
        } else {
            info!("⚠️  Skipping ask - {}", ask_reason);
        }
    } else {
        info!("⚠️  No ask: Smart quotes declined (toxicity or position limits)");
    }
}

fn should_stop_trading(state: &MarketMakerState) -> bool {
    let net_pnl = state.realized_pnl + state.unrealized_pnl - state.total_fees_paid;
    
    if net_pnl < -100.0 {
        info!("🛑 Stopping: Daily loss limit hit (${:.2})", net_pnl);
        return true;
    }
    
    if state.total_fills > 20 && state.realized_pnl > 1.0 {
        let fee_ratio = state.total_fees_paid / state.realized_pnl.abs();
        if fee_ratio > 0.8 {
            info!("🛑 Stopping: Fees eating >80% of gross profit");
            return true;
        }
    }
    
    let closed_trades = state.profitable_fills + state.losing_fills;
    if closed_trades >= 10 {
        let win_rate = state.profitable_fills as f64 / closed_trades as f64;
        if win_rate < 0.35 {
            info!("🛑 Stopping: Win rate too low ({:.1}% over {} closed trades)", 
                  win_rate * 100.0, closed_trades);
            return true;
        }
    }
    
    if state.unrealized_pnl < -50.0 {
        info!("🛑 Stopping: Large unrealized loss (${:.2})", state.unrealized_pnl);
        return true;
    }
    
    false
}

// ============================================
// FIX 5: EMERGENCY POSITION UNWINDING
// ============================================

fn should_unwind_position(state: &MarketMakerState, max_position: f64) -> bool {
    // Unwind if:
    // 1. Position is near max and losing money
    let position_ratio = state.position.abs() / max_position;
    let is_losing = state.unrealized_pnl < -5.0;
    
    if position_ratio > 0.8 && is_losing {
        info!("🚨 Emergency unwind: Position ratio {:.1}% with loss ${:.2}", 
              position_ratio * 100.0, state.unrealized_pnl);
        return true;
    }
    
    // 2. Fees are eating all profits
    if state.realized_pnl > 0.0 {
        let fee_ratio = state.total_fees_paid / state.realized_pnl;
        if fee_ratio > 0.9 {
            info!("🚨 Emergency unwind: Fees eating {:.1}% of profits", fee_ratio * 100.0);
            return true;
        }
    }
    
    // 3. Drawdown is too high
    let total_pnl = state.realized_pnl + state.unrealized_pnl - state.total_fees_paid;
    if total_pnl < -20.0 {
        info!("🚨 Emergency unwind: Total loss ${:.2} exceeds limit", total_pnl);
        return true;
    }
    
    false
}

fn place_unwind_orders(state: &mut MarketMakerState) {
    if state.position == 0.0 {
        return;
    }
    
    let unwind_size = state.position.abs().min(5.0); // Unwind up to 5 units at a time
    
    if state.position > 0.0 {
        // Need to sell to unwind long position
        // Place aggressive sell order (slightly below mid)
        let unwind_price = state.mid_price * 0.9995; // 0.5 bps below mid
        let ask_order = state.place_simulated_order(false, unwind_price, unwind_size);
        info!("🚨 UNWINDING LONG: Selling {:.2} @ ${:.4} (OID: {})", 
              unwind_size, unwind_price, ask_order.oid);
        state.our_ask_order = Some(ask_order);
    } else {
        // Need to buy to unwind short position
        // Place aggressive buy order (slightly above mid)
        let unwind_price = state.mid_price * 1.0005; // 0.5 bps above mid
        let bid_order = state.place_simulated_order(true, unwind_price, unwind_size);
        info!("🚨 UNWINDING SHORT: Buying {:.2} @ ${:.4} (OID: {})", 
              unwind_size, unwind_price, bid_order.oid);
        state.our_bid_order = Some(bid_order);
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let coin = "HYPE".to_string();
    let user = address!("0xB249153BE6B73B431AB8Adc0c7c922Bb1d38A6B7");

    let initial_capital = 500.0;
    let leverage = 10.0;

    // V4 Configuration - FIXED for ultra-tight spread markets
    let base_order_size = 2.5;  // Moderate size
    let base_spread_bps = 5.0;
    let max_position = 15.0;  // REDUCED for better inventory control
    let update_threshold_pct = 0.002;  // Legacy parameter - now dynamically adjusted based on volatility
    let status_interval = 5;
    let min_trades_before_start = 5;

    let risk_limits = RiskLimits {
        max_position_size: max_position,
        max_loss_per_day: 100.0,
        max_drawdown_pct: 15.0,
        startup_grace_period_ms: 120000,
        max_margin_usage_pct: 75.0,
        liquidation_buffer_pct: 15.0,
        min_expected_value_bps: 0.01,  // FIXED: 0.01 bps (0.001%) is realistic for tight spread markets
    };

    let mut info_client = InfoClient::new(None, Some(BaseUrl::Mainnet))
        .await
        .unwrap();
    let (sender, mut receiver) = unbounded_channel();
    let mut state = MarketMakerState::new(initial_capital, leverage);
    let mut update_counter = 0;
    let mut last_update_mid = 0.0;
    let mut trading_enabled = false;
    let mut warmup_complete = false;

    info!("✅ Subscribing to data for {}", coin);
    
    info_client
        .subscribe(
            Subscription::ActiveAssetData {
                user,
                coin: coin.clone(),
            },
            sender.clone(),
        )
        .await
        .unwrap();

    info_client
        .subscribe(
            Subscription::Bbo {
                coin: coin.clone(),
            },
            sender.clone(),
        )
        .await
        .unwrap();

    info_client
        .subscribe(
            Subscription::Trades {
                coin: coin.clone(),
            },
            sender.clone(),
        )
        .await
        .unwrap();
    
    info!("✅ Subscribed to trades for {}", coin);

    info!("🤖 PROBABILISTIC MARKET MAKER V4 Started for {}", coin);
    info!("╔════════════════════════════════════════════════════════╗");
    info!("🎲 V4 IMPROVED FLOW ANALYZER");
    info!("   ✓ Multi-timeframe toxicity (10s/60s/300s)");
    info!("   ✓ Volume-weighted directional flow (not count-based)");
    info!("   ✓ Adverse selection tracking against our orders");
    info!("   ✓ VWAP calculation for buy/sell sides");
    info!("   ✓ Momentum detection (consecutive trades)");
    info!("   ✓ Volume surge detection");
    info!("   ✓ Price impact measurement");
    info!("   ✓ One-sided flow detection");
    info!("   ✓ Dynamic prior (10% of volume, not fixed)");
    info!("   ✓ Statistical confidence based on sample size & volume");
    info!("╚════════════════════════════════════════════════════════╝");

    while let Some(message) = receiver.recv().await {
        match message {
            Message::Trades(trades) => {
                for trade in &trades.data {
                    let trade_price = trade.px.parse::<f64>().unwrap();
                    let trade_size = trade.sz.parse::<f64>().unwrap();
                    
                    let is_aggressive_buy = (trade_price - state.current_ask).abs() < 
                                           (trade_price - state.current_bid).abs();

                    let fills = state.check_fills(trade_price, is_aggressive_buy, trade_size);
                    for fill_msg in fills {
                        info!("{}", fill_msg);
                        let net_pnl = state.realized_pnl + state.unrealized_pnl - state.total_fees_paid;
                        info!(
                            "💰 Position: {:.2} HYPE | Realized: ${:.2} | Unrealized: ${:.2} | Net: ${:.2}",
                            state.position,
                            state.realized_pnl,
                            state.unrealized_pnl,
                            net_pnl
                        );

                        state.print_status();
                    }
                }
                
                if !warmup_complete && state.flow_analyzer.get_trade_count() >= min_trades_before_start {
                    warmup_complete = true;
                    trading_enabled = true;
                    info!("✅ Warmup complete! {} trades collected. Trading enabled.", 
                          state.flow_analyzer.get_trade_count());
                }
            }
            Message::Bbo(bbo) => {
                if let (Some(bid), Some(ask)) = (&bbo.data.bbo[0], &bbo.data.bbo[1]) {
                    let bid_price = bid.px.parse::<f64>().unwrap();
                    let ask_price = ask.px.parse::<f64>().unwrap();
                    let bid_sz = bid.sz.parse::<f64>().unwrap();
                    let ask_sz = ask.sz.parse::<f64>().unwrap();

                    state.update_bbo(bid_price, ask_price, bid_sz, ask_sz);

                    if !warmup_complete {
                        if state.flow_analyzer.get_trade_count() == 0 {
                            info!("⏳ Warming up... waiting for trade data (0/{} trades)", min_trades_before_start);
                        }
                        continue;
                    }

                    if should_stop_trading(&state) {
                        if trading_enabled {
                            info!("🛑 STOPPING TRADING DUE TO PERFORMANCE LIMITS 🛑");
                            cancel_simulated_orders(&mut state);
                            trading_enabled = false;
                            state.print_status();
                        }
                        continue;
                    }

                    let (is_safe, violations) = state.check_risk_limits(&risk_limits);
                    if !is_safe {
                        if trading_enabled {
                            info!("🚨 RISK LIMIT BREACHED - STOPPING TRADING 🚨");
                            for violation in violations {
                                info!("   ⛔ {}", violation);
                            }
                            cancel_simulated_orders(&mut state);
                            trading_enabled = false;
                        }
                        continue;
                    }

                    if trading_enabled {
                        // Check if emergency unwinding is needed
                        if should_unwind_position(&state, max_position) {
                            info!("🚨 EMERGENCY UNWINDING POSITION 🚨");
                            cancel_simulated_orders(&mut state);
                            place_unwind_orders(&mut state);
                            state.print_status();
                            continue;
                        }
                        
                        let should_update = should_update_orders(
                            &state,
                            last_update_mid,
                            update_threshold_pct,
                        );

                        if should_update {
                            last_update_mid = state.mid_price;

                            cancel_simulated_orders(&mut state);

                            place_probabilistic_orders(
                                &mut state,
                                base_order_size,
                                base_spread_bps,
                                max_position,
                                &risk_limits,
                            );

                            update_counter += 1;
                            if update_counter % status_interval == 0 {
                                state.print_status();
                            }
                        }
                    }
                }
            }
            Message::ActiveAssetData(data) => {
                let max_buy = data.data.max_trade_szs[0].parse::<f64>().unwrap_or(0.0);
                let max_sell = data.data.max_trade_szs[1].parse::<f64>().unwrap_or(0.0);
                state.update_capacity(max_buy, max_sell);
            }
            _ => {}
        }
    }
}
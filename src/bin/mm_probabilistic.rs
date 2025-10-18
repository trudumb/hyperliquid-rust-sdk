use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;

//RUST_LOG=info cargo run --bin mm_probabilistic

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

/// Poisson-based order flow analyzer
#[derive(Debug)]
struct PoissonFlowAnalyzer {
    trade_events: VecDeque<TradeEvent>,
    window_ms: u64,
    baseline_lambda: f64,
}

impl PoissonFlowAnalyzer {
    fn new(window_seconds: u64) -> Self {
        Self {
            trade_events: VecDeque::new(),
            window_ms: window_seconds * 1000,
            baseline_lambda: 0.0,
        }
    }

    fn add_trade(&mut self, event: TradeEvent) {
        info!("📊 Trade added: {:.2} @ ${:.4} (Buy: {}, Aggressive: {})", 
              event.size, event.price, event.is_buy, event.is_aggressive);
        
        let timestamp = event.timestamp;
        self.trade_events.push_back(event);

        let cutoff = timestamp.saturating_sub(self.window_ms);
        while let Some(e) = self.trade_events.front() {
            if e.timestamp < cutoff {
                self.trade_events.pop_front();
            } else {
                break;
            }
        }

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

    fn get_trade_count(&self) -> usize {
        self.trade_events.len()
    }

    fn get_toxicity_probability(&self, lookback_seconds: f64) -> f64 {
        if self.baseline_lambda < EPSILON || self.trade_events.is_empty() {
            return 0.0;
        }

        let now = self.trade_events.back().unwrap().timestamp;
        let cutoff = now.saturating_sub((lookback_seconds * 1000.0) as u64);

        let recent_count = self.trade_events.iter()
            .filter(|e| e.timestamp >= cutoff)
            .count();

        let expected_count = self.baseline_lambda * lookback_seconds;

        if expected_count < EPSILON {
            return 0.0;
        }

        let lambda = expected_count;
        let k = recent_count as f64;

        if lambda > 10.0 {
            let z_score = (k - lambda) / lambda.sqrt();
            let p = 1.0 - normal_cdf(z_score);
            1.0 - p
        } else {
            let prob_extreme = poisson_tail_probability(lambda, k as u64);
            1.0 - prob_extreme
        }
    }

    fn get_directional_toxicity(&self, lookback_seconds: f64) -> (f64, f64) {
        if self.trade_events.is_empty() {
            return (0.5, 0.0);
        }

        let now = self.trade_events.back().unwrap().timestamp;
        let cutoff = now.saturating_sub((lookback_seconds * 1000.0) as u64);

        let recent_trades: Vec<&TradeEvent> = self.trade_events.iter()
            .filter(|e| e.timestamp >= cutoff && e.is_aggressive)
            .collect();

        if recent_trades.is_empty() {
            return (0.5, 0.0);
        }

        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in recent_trades.iter() {
            if trade.is_buy {
                buy_volume += trade.size;
            } else {
                sell_volume += trade.size;
            }
        }

        // CRITICAL FIX: Add prior to prevent buy_ratio from being 0%
        buy_volume += 0.2;  // Small prior
        sell_volume += 0.2;

        let total_volume = buy_volume + sell_volume;
        if total_volume < EPSILON {
            return (0.5, 0.0);
        }

        let buy_ratio = buy_volume / total_volume;

        let n = recent_trades.len();
        let k = recent_trades.iter().filter(|t| t.is_buy).count();

        let p_observed = k as f64 / n as f64;
        let p_null = 0.5;
        let se = (p_null * (1.0 - p_null) / n as f64).sqrt();
        
        if se < EPSILON {
            return (buy_ratio, 0.0);
        }

        let z = ((p_observed - p_null).abs()) / se;
        let confidence = 1.0 - 2.0 * (1.0 - normal_cdf(z));

        (buy_ratio, confidence.max(0.0))
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

    flow_analyzer: PoissonFlowAnalyzer,
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
            flow_analyzer: PoissonFlowAnalyzer::new(300),
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

        self.flow_analyzer.add_trade(TradeEvent {
            timestamp,
            price: trade_price,
            size: trade_size,
            is_buy: is_aggressive_buy,
            is_aggressive: true,
        });
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
        info!("🎲 PROBABILISTIC ANALYSIS (Trades: {})", trade_count);
        info!(
            "Toxicity Probability: {:.1}% | Buy Flow: {:.1}% (confidence: {:.1}%)",
            toxicity_prob * 100.0, buy_ratio * 100.0, dir_confidence * 100.0
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

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn poisson_tail_probability(lambda: f64, k: u64) -> f64 {
    let mut prob = 0.0;
    let mut term = (-lambda).exp();
    
    for i in 0..k {
        prob += term;
        term *= lambda / (i + 1) as f64;
    }
    
    1.0 - prob
}

fn calculate_minimum_spread(state: &MarketMakerState) -> f64 {
    let fee_coverage = state.maker_fee_rate * 2.0 * 10000.0;
    let adverse_selection_buffer = 0.3;  // Further reduced from 0.5
    let min_profit_target = 0.2;  // Further reduced from 0.3
    
    fee_coverage + adverse_selection_buffer + min_profit_target
}

/// Calculate expected value - V3 with minimal directional adjustment
fn calculate_expected_value_spread(
    state: &MarketMakerState,
    base_spread_bps: f64,
    max_position: f64,
    _risk_limits: &RiskLimits,
) -> (f64, f64, f64) {
    let mut spread_bps = base_spread_bps;
    
    let min_spread = calculate_minimum_spread(state);
    spread_bps = spread_bps.max(min_spread);
    
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.01 {
        spread_bps *= 1.0 + volatility * 20.0;
    }
    
    let position_ratio = (state.position.abs() / max_position).min(1.0);
    spread_bps *= 1.0 + position_ratio.powi(2) * 0.5;
    
    let toxic_prob = state.bayesian_estimator.prob_toxic_flow;
    let adverse_selection_cost = toxic_prob * 1.0;
    spread_bps += adverse_selection_cost;
    
    let (buy_ratio, dir_confidence) = state.flow_analyzer.get_directional_toxicity(30.0);
    
    // CRITICAL FIX V3: Further reduced from 5.0 to 2.0
    let directional_adjustment = (buy_ratio - 0.5) * 2.0 * dir_confidence * 2.0;
    
    let half_spread_bps = spread_bps / 2.0;
    
    let bid_edge_bps = half_spread_bps + directional_adjustment;
    let bid_price = state.mid_price * (1.0 - bid_edge_bps / 10000.0);
    let bid_fill_prob = state.flow_analyzer.get_fill_probability(bid_price, true, state.mid_price);
    
    // CRITICAL FIX V3: Further reduced from 0.2 to 0.1
    let bid_ev_bps = (bid_edge_bps - adverse_selection_cost * (1.0 - buy_ratio) * 0.1) * bid_fill_prob;
    
    info!("🔹 Bid EV: edge={:.2} bps, dir_adj={:.2}, fill_prob={:.1}%, adv_cost={:.2}, buy_ratio={:.1}%, final_ev={:.2}", 
          bid_edge_bps, directional_adjustment, bid_fill_prob * 100.0, adverse_selection_cost, buy_ratio * 100.0, bid_ev_bps);
    
    let ask_edge_bps = half_spread_bps - directional_adjustment;
    let ask_price = state.mid_price * (1.0 + ask_edge_bps / 10000.0);
    let ask_fill_prob = state.flow_analyzer.get_fill_probability(ask_price, false, state.mid_price);
    let ask_ev_bps = (ask_edge_bps - adverse_selection_cost * buy_ratio * 0.1) * ask_fill_prob;
    
    info!("🔸 Ask EV: edge={:.2} bps, dir_adj={:.2}, fill_prob={:.1}%, adv_cost={:.2}, final_ev={:.2}", 
          ask_edge_bps, directional_adjustment, ask_fill_prob * 100.0, adverse_selection_cost, ask_ev_bps);
    
    (spread_bps, bid_ev_bps, ask_ev_bps)
}

fn calculate_dynamic_skew(
    state: &MarketMakerState,
    max_position: f64,
) -> (f64, f64, f64, f64) {
    let position_ratio = state.position / max_position;
    
    let trending_bias = (state.bayesian_estimator.prob_trending - 0.25) * 20.0;
    let mean_revert_bias = (state.bayesian_estimator.prob_mean_reverting - 0.5) * -10.0;
    
    let inventory_urgency = if position_ratio.abs() > 0.5 {
        position_ratio.powi(2) * 100.0
    } else {
        position_ratio.powi(2) * position_ratio.signum() * 40.0
    };

    let total_skew_bps = inventory_urgency + trending_bias + mean_revert_bias;
    let skew_amount = state.mid_price * total_skew_bps / 10000.0;

    let confidence = state.bayesian_estimator.get_confidence();
    let mut size_multiplier = 0.8 + confidence * 0.4;
    
    size_multiplier *= 1.0 - position_ratio.abs() * 0.5;
    
    if state.bayesian_estimator.prob_volatile > 0.5 {
        size_multiplier *= 0.7;
    }

    let (buy_ratio, dir_confidence) = state.flow_analyzer.get_directional_toxicity(30.0);
    let mut flow_asymmetry = (buy_ratio - 0.5) * 2.0 * dir_confidence;
    
    // CRITICAL FIX: Cap directional adjustment to improve ask fill probability
    // Prevents asks from moving too far away during strong buy flow
    flow_asymmetry = flow_asymmetry.clamp(-0.5, 0.5);
    
    let bid_size_adj = (1.0 + flow_asymmetry * 0.3).max(0.7).min(1.3);
    let ask_size_adj = (1.0 - flow_asymmetry * 0.3).max(0.7).min(1.3);

    (skew_amount, size_multiplier, bid_size_adj, ask_size_adj)
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
    let ev_threshold = if is_reducing_position {
        risk_limits.min_expected_value_bps * 0.2
    } else {
        risk_limits.min_expected_value_bps
    };
    
    if expected_value_bps < ev_threshold {
        return (false, format!("EV too low: {:.2} < {:.2} bps", expected_value_bps, ev_threshold));
    }

    let toxic_prob = state.bayesian_estimator.prob_toxic_flow;
    if toxic_prob > 0.7 {
        return (false, format!("Toxic flow probability too high: {:.1}%", toxic_prob * 100.0));
    }

    // CRITICAL FIX V3: Ask threshold lowered from 0.05 to 0.01
    let (buy_ratio, dir_confidence) = state.flow_analyzer.get_directional_toxicity(30.0);
    if dir_confidence > 0.6 {
        if is_buy && buy_ratio > 0.9 {
            return (false, "Strong buying pressure detected".to_string());
        }
        if !is_buy && buy_ratio < 0.01 {  // Changed from 0.05
            return (false, format!("Strong selling pressure detected (buy_ratio: {:.1}%)", buy_ratio * 100.0));
        }
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
    threshold_pct: f64,
) -> bool {
    if state.our_bid_order.is_none() || state.our_ask_order.is_none() {
        return true;
    }

    // Relaxed: Increased from threshold_pct * 10.0 to threshold_pct * 25.0
    // This means we only cancel if price moves >5 bps (0.002 * 25 = 0.05 = 5 bps)
    if last_mid > 0.0 {
        let price_change_pct = ((state.mid_price - last_mid) / last_mid).abs();
        if price_change_pct > threshold_pct * 25.0 {
            return true;
        }
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    if let Some(bid) = &state.our_bid_order {
        // Relaxed: Widened edge bounds from [1.0, 150.0] to [0.5, 200.0]
        let bid_edge_bps = ((state.mid_price - bid.price) / state.mid_price) * 10000.0;
        if bid_edge_bps > 200.0 || bid_edge_bps < 0.5 {
            return true;
        }
        
        // Relaxed: Increased minimum order age from 180s to 300s (5 minutes)
        let order_age = now.saturating_sub(bid.timestamp) / 1000;
        if order_age > 300 {
            info!("🕐 Bid order too old ({}s), cancelling", order_age);
            return true;
        }
    }
    
    if let Some(ask) = &state.our_ask_order {
        // Relaxed: Widened edge bounds from [1.0, 150.0] to [0.5, 200.0]
        let ask_edge_bps = ((ask.price - state.mid_price) / state.mid_price) * 10000.0;
        if ask_edge_bps > 200.0 || ask_edge_bps < 0.5 {
            return true;
        }
        
        // Relaxed: Increased minimum order age from 180s to 300s (5 minutes)
        let order_age = now.saturating_sub(ask.timestamp) / 1000;
        if order_age > 300 {
            info!("🕐 Ask order too old ({}s), cancelling", order_age);
            return true;
        }
    }

    // Relaxed: Increased time since last update from 120s to 180s (3 minutes)
    let time_since_update = now.saturating_sub(state.last_update_ts) as f64 / 1000.0;
    
    if time_since_update > 180.0 {
        return true;
    }

    // Relaxed: Increased toxic flow threshold from 0.6 to 0.8
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
    base_spread_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) {
    let (spread_bps, bid_ev_bps, ask_ev_bps) = calculate_expected_value_spread(
        state, base_spread_bps, max_position, risk_limits
    );
    
    let half_spread_bps = spread_bps / 2.0;

    let (skew_adjustment, size_multiplier, bid_size_adj, ask_size_adj) =
        calculate_dynamic_skew(state, max_position);

    // CRITICAL FIX V3: Even tighter - 0.2 instead of 0.3
    let bid_spread_bps = (half_spread_bps + (skew_adjustment / state.mid_price) * 10000.0) * 0.2;
    let ask_spread_bps = (half_spread_bps - (skew_adjustment / state.mid_price) * 10000.0) * 0.2;

    let our_bid_price = state.mid_price * (1.0 - bid_spread_bps / 10000.0);
    let our_ask_price = state.mid_price * (1.0 + ask_spread_bps / 10000.0);

    let bid_size = (base_order_size * size_multiplier * bid_size_adj * 100.0).round() / 100.0;
    let ask_size = (base_order_size * size_multiplier * ask_size_adj * 100.0).round() / 100.0;

    let our_bid_price = (our_bid_price * 10000.0).round() / 10000.0;
    let our_ask_price = (our_ask_price * 10000.0).round() / 10000.0;

    let (can_place_bid, bid_reason) = should_place_order(
        state, true, bid_size, our_bid_price, bid_ev_bps, max_position, risk_limits
    );
    
    if can_place_bid && bid_size >= 0.01 {
        let bid_order = state.place_simulated_order(true, our_bid_price, bid_size);
        let bid_fill_prob = state.flow_analyzer.get_fill_probability(our_bid_price, true, state.mid_price);
        info!(
            "🔹 BID: {:.2} @ ${:.4} (OID: {}) | {:.2} bps EV | {:.1}% fill prob | {}",
            bid_size, our_bid_price, bid_order.oid, bid_ev_bps, bid_fill_prob * 100.0, bid_reason
        );
        state.our_bid_order = Some(bid_order);
    } else {
        info!("⚠️  Skipping bid - {}", bid_reason);
    }

    let (can_place_ask, ask_reason) = should_place_order(
        state, false, ask_size, our_ask_price, ask_ev_bps, max_position, risk_limits
    );
    
    if can_place_ask && ask_size >= 0.01 {
        let ask_order = state.place_simulated_order(false, our_ask_price, ask_size);
        let ask_fill_prob = state.flow_analyzer.get_fill_probability(our_ask_price, false, state.mid_price);
        info!(
            "🔸 ASK: {:.2} @ ${:.4} (OID: {}) | {:.2} bps EV | {:.1}% fill prob | {}",
            ask_size, our_ask_price, ask_order.oid, ask_ev_bps, ask_fill_prob * 100.0, ask_reason
        );
        state.our_ask_order = Some(ask_order);
    } else {
        info!("⚠️  Skipping ask - {}", ask_reason);
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

#[tokio::main]
async fn main() {
    env_logger::init();

    let coin = "HYPE".to_string();
    let user = address!("0xB249153BE6B73B431AB8Adc0c7c922Bb1d38A6B7");

    let initial_capital = 500.0;
    let leverage = 10.0;

    // V3 Configuration - Final aggressive tuning
    let base_order_size = 5.0;
    let base_spread_bps = 3.5;  // Reduced from 4.0
    let max_position = 30.0;
    let update_threshold_pct = 0.002;
    let status_interval = 5;
    let min_trades_before_start = 5;

    let risk_limits = RiskLimits {
        max_position_size: max_position,
        max_loss_per_day: 100.0,
        max_drawdown_pct: 15.0,
        startup_grace_period_ms: 120000,
        max_margin_usage_pct: 75.0,
        liquidation_buffer_pct: 15.0,
        min_expected_value_bps: 0.0,  // CRITICAL: Allow any positive EV
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

    info!("🤖 PROBABILISTIC MARKET MAKER V3 Started for {}", coin);
    info!("╔════════════════════════════════════════════════════════╗");
    info!("🎲 V3 CRITICAL FIXES");
    info!("   ✓ Directional adjustment: 5.0 → 2.0 (60% reduction)");
    info!("   ✓ Adverse selection: 20% → 10% (50% reduction)");
    info!("   ✓ Sell pressure threshold: 5% → 1%");
    info!("   ✓ Spread multiplier: 30% → 20% (even tighter)");
    info!("   ✓ Min spread components reduced");
    info!("   ✓ Buy ratio prior: +0.2 to prevent 0%");
    info!("   ✓ Min EV threshold: 0.005 → 0.0 (any positive)");
    info!("   ✓ Base spread: 4.0 → 3.5 bps");
    info!("   ✓ Order book depth logging");
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
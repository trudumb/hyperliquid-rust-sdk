use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;

//RUST_LOG=info cargo run --bin mm_with_trends

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
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
    Calm,
}

#[derive(Debug)]
struct TrendDetector {
    prices: VecDeque<f64>,
    timestamps: VecDeque<u64>,
    ema_short: Option<f64>,
    ema_long: Option<f64>,
    alpha_short: f64,
    alpha_long: f64,
    trade_flow: VecDeque<f64>,
    max_samples: usize,
}

impl TrendDetector {
    fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            prices: VecDeque::new(),
            timestamps: VecDeque::new(),
            ema_short: None,
            ema_long: None,
            alpha_short: 2.0 / (short_period as f64 + 1.0),
            alpha_long: 2.0 / (long_period as f64 + 1.0),
            trade_flow: VecDeque::new(),
            max_samples: long_period * 3,
        }
    }

    fn update(&mut self, price: f64, timestamp: u64) {
        self.prices.push_back(price);
        self.timestamps.push_back(timestamp);

        match self.ema_short {
            None => self.ema_short = Some(price),
            Some(ema) => self.ema_short = Some(self.alpha_short * price + (1.0 - self.alpha_short) * ema),
        }

        match self.ema_long {
            None => self.ema_long = Some(price),
            Some(ema) => self.ema_long = Some(self.alpha_long * price + (1.0 - self.alpha_long) * ema),
        }

        if self.prices.len() > self.max_samples {
            self.prices.pop_front();
            self.timestamps.pop_front();
        }
    }

    fn record_trade(&mut self, is_aggressive_buy: bool, size: f64) {
        let flow = if is_aggressive_buy { size } else { -size };
        self.trade_flow.push_back(flow);

        if self.trade_flow.len() > 100 {
            self.trade_flow.pop_front();
        }
    }

    fn get_trend_strength(&self) -> f64 {
        if self.prices.len() < 20 {
            return 0.0;
        }

        let mut score = 0.0;
        let mut weight_sum = 0.0;

        if let (Some(short), Some(long)) = (self.ema_short, self.ema_long) {
            let ema_diff = (short - long) / long;
            score += ema_diff * 20.0 * 0.4;
            weight_sum += 0.4;
        }

        if let Some(slope) = self.calculate_regression_slope() {
            score += slope * 0.3;
            weight_sum += 0.3;
        }

        if !self.trade_flow.is_empty() {
            let flow_sum: f64 = self.trade_flow.iter().sum();
            let flow_signal = flow_sum / self.trade_flow.len() as f64;
            score += flow_signal.signum() * flow_signal.abs().min(1.0) * 0.3;
            weight_sum += 0.3;
        }

        if weight_sum > 0.0 {
            (score / weight_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    fn get_momentum_z_score(&self) -> f64 {
        if self.prices.len() < 20 {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.prices.iter().rev().take(20).copied().collect();
        let current_price = recent_prices[0];
        
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let variance = recent_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            (current_price - mean) / std_dev
        } else {
            0.0
        }
    }

    fn calculate_regression_slope(&self) -> Option<f64> {
        if self.prices.len() < 10 {
            return None;
        }

        let recent: Vec<(f64, f64)> = self.prices.iter().rev()
            .take(30)
            .enumerate()
            .map(|(i, &p)| (i as f64, p))
            .collect();

        let n = recent.len() as f64;
        let sum_x: f64 = recent.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = recent.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = recent.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = recent.iter().map(|(x, _)| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let current_price = self.prices.back()?;
        Some(slope / current_price * 100.0)
    }

    fn classify_regime(&self, volatility: f64) -> MarketRegime {
        let trend_strength = self.get_trend_strength().abs();
        let z_score = self.get_momentum_z_score().abs();

        if volatility > 0.03 {
            return MarketRegime::Volatile;
        }

        if trend_strength > 0.4 || z_score > 2.0 {
            return MarketRegime::Trending;
        }

        if volatility < 0.01 && trend_strength < 0.2 {
            return MarketRegime::Calm;
        }

        MarketRegime::MeanReverting
    }

    fn get_flow_imbalance(&self) -> f64 {
        if self.trade_flow.is_empty() {
            return 0.0;
        }

        let total: f64 = self.trade_flow.iter().sum();
        let abs_total: f64 = self.trade_flow.iter().map(|f| f.abs()).sum();

        if abs_total > 0.0 {
            total / abs_total
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
struct MarketMakerState {
    // Capital and Leverage Management
    initial_capital: f64,
    current_capital: f64,
    leverage: f64,
    margin_used: f64,
    liquidation_price: f64,
    
    // Market data
    current_bid: f64,
    current_ask: f64,
    mid_price: f64,
    spread: f64,
    bid_depth: f64,
    ask_depth: f64,

    // Our simulated orders
    our_bid_order: Option<SimulatedOrder>,
    our_ask_order: Option<SimulatedOrder>,

    // Position tracking
    position: f64,
    average_entry: f64,

    // Performance tracking
    realized_pnl: f64,
    unrealized_pnl: f64,
    total_fills: u32,
    buy_fills: u32,
    sell_fills: u32,
    profitable_fills: u32,
    losing_fills: u32,

    // Fees
    maker_fee_rate: f64,
    taker_fee_rate: f64,
    total_fees_paid: f64,

    // Capacity
    max_buy_size: f64,
    max_sell_size: f64,

    // Order ID generator
    next_oid: u64,

    // Analytics
    volatility_tracker: VolatilityTracker,
    trend_detector: TrendDetector,

    // Session tracking
    session_start_time: u64,
    peak_pnl: f64,
    
    // Market regime
    current_regime: MarketRegime,
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
            taker_fee_rate: 0.00045,
            total_fees_paid: 0.0,
            max_buy_size: 0.0,
            max_sell_size: 0.0,
            next_oid: 1,
            volatility_tracker: VolatilityTracker::new(300),
            trend_detector: TrendDetector::new(10, 30),
            session_start_time: now,
            peak_pnl: 0.0,
            current_regime: MarketRegime::Calm,
        }
    }

    fn update_bbo(&mut self, bid: f64, ask: f64, bid_sz: f64, ask_sz: f64) {
        self.current_bid = bid;
        self.current_ask = ask;
        self.mid_price = (bid + ask) / 2.0;
        self.spread = ask - bid;
        self.bid_depth = bid_sz;
        self.ask_depth = ask_sz;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        self.volatility_tracker.add_sample(timestamp, self.mid_price);
        self.trend_detector.update(self.mid_price, timestamp);

        let volatility = self.volatility_tracker.get_volatility();
        self.current_regime = self.trend_detector.classify_regime(volatility);

        if self.position != 0.0 {
            self.unrealized_pnl = (self.mid_price - self.average_entry) * self.position;
        }

        self.update_margin();
        self.update_capital();

        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl > self.peak_pnl {
            self.peak_pnl = total_pnl;
        }
    }

    fn update_margin(&mut self) {
        if self.position != 0.0 {
            let position_notional = self.position.abs() * self.average_entry;
            
            // Initial margin required for this position
            self.margin_used = position_notional / self.leverage;
            
            // Maintenance margin is half of initial margin at max leverage
            // For leverage L, initial margin = 1/L, so maintenance = 1/(2*L)
            let maintenance_margin_ratio = 1.0 / (2.0 * self.leverage);
            let maintenance_margin = position_notional * maintenance_margin_ratio;
            
            // Calculate liquidation price
            // For LONG: liquidation when (equity = maintenance_margin)
            // equity = initial_margin + unrealized_pnl
            // unrealized_pnl = (current_price - entry_price) * position_size
            // At liquidation: initial_margin + (liq_price - entry_price) * position_size = maintenance_margin
            // Solving for liq_price:
            
            if self.position > 0.0 {
                // Long position
                // liq_price = entry_price - (initial_margin - maintenance_margin) / position_size
                let margin_buffer = self.margin_used - maintenance_margin;
                self.liquidation_price = self.average_entry - (margin_buffer / self.position);
            } else {
                // Short position
                // liq_price = entry_price + (initial_margin - maintenance_margin) / |position_size|
                let margin_buffer = self.margin_used - maintenance_margin;
                self.liquidation_price = self.average_entry + (margin_buffer / self.position.abs());
            }
            
            // Ensure liquidation price is positive
            self.liquidation_price = self.liquidation_price.max(0.0);
        } else {
            self.margin_used = 0.0;
            self.liquidation_price = 0.0;
        }
    }

    fn update_capital(&mut self) {
        // Current capital is initial capital + realized PnL - fees
        // Unrealized PnL is NOT added to capital (it's just mark-to-market)
        self.current_capital = self.initial_capital + self.realized_pnl - self.total_fees_paid;
    }

    fn get_available_margin(&self) -> f64 {
        // Available margin for new positions
        // Must account for both margin used AND maintain 10% buffer for transfers
        let equity = self.current_capital + self.unrealized_pnl;
        let used_margin = self.margin_used;
        
        // For new positions, we need to ensure we keep enough for existing positions
        // and maintain the 10% transfer requirement
        let position_notional = if self.position != 0.0 {
            self.position.abs() * self.mid_price
        } else {
            0.0
        };
        
        let transfer_requirement = (position_notional * 0.1).max(used_margin);
        
        (equity - transfer_requirement).max(0.0)
    }

    fn get_max_position_size(&self, price: f64, max_position_limit: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }
        
        let available_margin = self.get_available_margin();
        
        // With leverage L, margin_required = notional / L
        // So: max_notional = available_margin * L
        let max_notional_from_margin = available_margin * self.leverage;
        let max_size_from_margin = max_notional_from_margin / price;
        
        // Return the smaller of margin-based limit and configured limit
        max_size_from_margin.min(max_position_limit)
    }

    fn is_near_liquidation(&self, buffer_pct: f64) -> bool {
        if self.position == 0.0 || self.liquidation_price == 0.0 {
            return false;
        }

        let buffer_multiplier = 1.0 + (buffer_pct / 100.0);

        if self.position > 0.0 {
            // Long position: check if price is close to liquidation price below
            self.mid_price < self.liquidation_price * buffer_multiplier
        } else {
            // Short position: check if price is close to liquidation price above
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

    fn check_fills(&mut self, trade_price: f64, is_aggressive_buy: bool, size: f64) -> Vec<String> {
        self.trend_detector.record_trade(is_aggressive_buy, size);
        
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

            if old_position < 0.0 && self.position >= 0.0 {
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
                }
            } else if old_position >= 0.0 {
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

            if old_position > 0.0 && self.position <= 0.0 {
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
                }
            } else if old_position <= 0.0 {
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

    fn get_book_imbalance(&self) -> f64 {
        if self.bid_depth + self.ask_depth == 0.0 {
            return 0.0;
        }
        (self.bid_depth - self.ask_depth) / (self.bid_depth + self.ask_depth)
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
        let time_elapsed = now - self.session_start_time;
        let in_grace_period = time_elapsed < limits.startup_grace_period_ms;

        // Calculate account equity (for cross margin, this would be account value)
        let equity = self.current_capital + self.unrealized_pnl;
        
        // Maintenance margin check - most critical
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

        // Liquidation proximity check
        if self.is_near_liquidation(limits.liquidation_buffer_pct) {
            violations.push(format!(
                "Near liquidation! Current: ${:.4}, Liquidation: ${:.4}",
                self.mid_price, self.liquidation_price
            ));
            is_safe = false;
        }

        // Margin usage check - based on equity not just initial capital
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

        // Position limit
        if self.position.abs() > limits.max_position_size {
            violations.push(format!(
                "Position limit breached: {:.2} > {:.2}",
                self.position.abs(),
                limits.max_position_size
            ));
            is_safe = false;
        }

        // Loss limit
        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl < -limits.max_loss_per_day {
            violations.push(format!(
                "Daily loss limit breached: ${:.2} < -${:.2}",
                total_pnl, limits.max_loss_per_day
            ));
            is_safe = false;
        }

        // Drawdown check (outside grace period)
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
        } else if !violations.is_empty() {
            info!("⏱️  In grace period ({:.1}s remaining) - some checks suspended",
                (limits.startup_grace_period_ms - time_elapsed) as f64 / 1000.0);
        }

        (is_safe, violations)
    }

    fn print_status(&self) {
        let total_pnl = self.realized_pnl + self.unrealized_pnl;
        let net_pnl = total_pnl - self.total_fees_paid;
        let volatility = self.volatility_tracker.get_volatility() * 100.0;
        let imbalance = self.get_book_imbalance();
        let drawdown = self.get_current_drawdown_pct();

        let trend_strength = self.trend_detector.get_trend_strength();
        let momentum_z = self.trend_detector.get_momentum_z_score();
        let flow_imbalance = self.trend_detector.get_flow_imbalance();
        
        let trend_direction = if trend_strength > 0.3 {
            "📈 UPTREND"
        } else if trend_strength < -0.3 {
            "📉 DOWNTREND"
        } else {
            "↔️  SIDEWAYS"
        };

        let regime_emoji = match self.current_regime {
            MarketRegime::Trending => "🔥",
            MarketRegime::MeanReverting => "🔄",
            MarketRegime::Volatile => "⚡",
            MarketRegime::Calm => "😌",
        };

        let win_rate = if self.profitable_fills + self.losing_fills > 0 {
            (self.profitable_fills as f64
                / (self.profitable_fills + self.losing_fills) as f64)
                * 100.0
        } else {
            0.0
        };

        // Calculate equity and margin usage based on equity
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

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 IMPROVED MARKET MAKER STATUS");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!(
            "Market: Bid ${:.4} ({:.1}) / Ask ${:.4} ({:.1}) | Spread: ${:.4} ({:.2} bps)",
            self.current_bid,
            self.bid_depth,
            self.current_ask,
            self.ask_depth,
            self.spread,
            (self.spread / self.mid_price) * 10000.0
        );
        info!(
            "Volatility: {:.3}% | Book Imbalance: {:.2}",
            volatility, imbalance
        );

        info!("─────────────────────────────────────────────────────");
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

        info!("─────────────────────────────────────────────────────");
        info!("🎯 TREND ANALYSIS");
        info!(
            "Regime: {} {:?} | Direction: {} (strength: {:.2})",
            regime_emoji, self.current_regime, trend_direction, trend_strength
        );
        info!(
            "Momentum Z-Score: {:.2} | Flow Imbalance: {:.2}",
            momentum_z, flow_imbalance
        );

        info!("─────────────────────────────────────────────────────");

        if let Some(bid) = &self.our_bid_order {
            let edge = ((self.mid_price - bid.price) / self.mid_price) * 10000.0;
            info!(
                "Our Bid: {:.2} @ ${:.4} (OID: {}) [{:.1} bps from mid]",
                bid.size, bid.price, bid.oid, edge
            );
        } else {
            info!("Our Bid: None");
        }

        if let Some(ask) = &self.our_ask_order {
            let edge = ((ask.price - self.mid_price) / self.mid_price) * 10000.0;
            info!(
                "Our Ask: {:.2} @ ${:.4} (OID: {}) [{:.1} bps from mid]",
                ask.size, ask.price, ask.oid, edge
            );
        } else {
            info!("Our Ask: None");
        }

        info!("─────────────────────────────────────────────────────");
        info!(
            "Position: {:.2} HYPE @ ${:.4} avg entry",
            self.position, self.average_entry
        );
        info!("Realized PnL: ${:.2}", self.realized_pnl);
        info!("Unrealized PnL: ${:.2}", self.unrealized_pnl);
        info!("Fees Paid: ${:.2}", self.total_fees_paid);
        info!("Net PnL: ${:.2}", net_pnl);
        
        if self.peak_pnl >= 5.0 {
            info!("Peak PnL: ${:.2} | Drawdown: {:.2}%", self.peak_pnl, drawdown);
        } else {
            info!("Peak PnL: ${:.2} | Loss: ${:.2}", self.peak_pnl, drawdown);
        }

        info!("─────────────────────────────────────────────────────");
        info!(
            "Fills: {} total ({} buys, {} sells)",
            self.total_fills, self.buy_fills, self.sell_fills
        );
        info!(
            "Win Rate: {:.1}% ({} wins, {} losses)",
            win_rate, self.profitable_fills, self.losing_fills
        );

        if self.total_fills > 0 {
            let pnl_per_fill = net_pnl / self.total_fills as f64;
            info!("Avg PnL per Fill: ${:.3}", pnl_per_fill);
        }

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

fn calculate_adaptive_spread(
    state: &MarketMakerState,
    base_spread_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) -> f64 {
    let mut spread_multiplier = 1.0;

    // 1. Volatility adjustment - more aggressive
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.02 {
        spread_multiplier *= 1.0 + (volatility - 0.02) * 30.0; // Increased from 20.0
    }

    // 2. Order book imbalance
    let imbalance = state.get_book_imbalance();
    spread_multiplier *= 1.0 + imbalance.abs() * 0.5; // Increased from 0.3

    // 3. Position risk
    let position_ratio = state.position.abs() / max_position;
    spread_multiplier *= 1.0 + position_ratio * 0.7; // Increased from 0.5

    // 4. TREND-BASED ADJUSTMENT - Much more aggressive
    let trend_strength = state.trend_detector.get_trend_strength();
    let momentum_z = state.trend_detector.get_momentum_z_score();
    
    match state.current_regime {
        MarketRegime::Trending => {
            // Widen significantly to avoid adverse selection
            spread_multiplier *= 2.0 + trend_strength.abs() * 2.0; // Was 1.3 + 0.5
            
            // Additional widening for strong momentum
            if momentum_z.abs() > 2.0 {
                spread_multiplier *= 1.5;
            }
        }
        MarketRegime::Volatile => {
            spread_multiplier *= 2.5; // Was 1.5
        }
        MarketRegime::MeanReverting => {
            spread_multiplier *= 0.9;
        }
        MarketRegime::Calm => {
            spread_multiplier *= 0.85;
        }
    }

    // 5. Near liquidation - widen drastically
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        spread_multiplier *= 3.0;
    }

    // 6. Minimum spread covers fees with buffer
    let min_spread_bps = state.maker_fee_rate * 2.0 * 10000.0 * 2.0; // 2x buffer
    let target_spread_bps = (base_spread_bps * spread_multiplier).max(min_spread_bps);

    (state.mid_price * target_spread_bps) / 10000.0
}

fn calculate_dynamic_skew(
    state: &MarketMakerState,
    max_position: f64,
) -> (f64, f64, f64, f64) {
    let position_ratio = state.position / max_position;
    let trend_strength = state.trend_detector.get_trend_strength();
    let momentum_z = state.trend_detector.get_momentum_z_score();
    let flow_imbalance = state.trend_detector.get_flow_imbalance();

    // 1. CRITICAL: Inventory management with trend awareness
    let inventory_urgency = if position_ratio < -0.3 && trend_strength > 0.3 {
        // Short in uptrend - VERY urgent to cover
        -150.0 // Large negative skew = much tighter bids, much wider asks
    } else if position_ratio > 0.3 && trend_strength < -0.3 {
        // Long in downtrend - VERY urgent to sell
        150.0 // Large positive skew = much wider bids, much tighter asks
    } else if position_ratio.abs() > 0.5 {
        // Just generally too much position
        position_ratio.powi(3) * 100.0
    } else {
        position_ratio.powi(3) * 50.0
    };

    // 2. Trend-based skew - only when NOT fighting position
    let trend_skew_bps = if position_ratio * trend_strength < -0.1 {
        // Position and trend are opposite - DON'T add trend bias
        0.0
    } else if position_ratio.abs() < 0.2 {
        // Small position - can lean into trend
        trend_strength * 20.0
    } else {
        // Reduce trend bias when already positioned
        trend_strength * 10.0
    };

    // 3. Mean reversion skew - fade extremes
    let mean_revert_skew = match state.current_regime {
        MarketRegime::MeanReverting => {
            if momentum_z.abs() > 2.0 {
                -momentum_z.signum() * 15.0 // Fade strong moves
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    let total_skew_bps = inventory_urgency + trend_skew_bps + mean_revert_skew;
    let skew_amount = state.mid_price * total_skew_bps / 10000.0;

    // 4. Size adjustment based on regime and position
    let mut size_multiplier = 1.0 - position_ratio.abs() * 0.6; // More aggressive reduction
    
    match state.current_regime {
        MarketRegime::Volatile => {
            size_multiplier *= 0.6; // Was 0.7
        }
        MarketRegime::Calm => {
            size_multiplier *= 1.3; // Was 1.2
        }
        MarketRegime::Trending => {
            // Reduce size in trends to avoid adverse selection
            size_multiplier *= 0.8;
        }
        _ => {}
    }

    // 5. Asymmetric sizing based on position and trend
    let (bid_size_adj, ask_size_adj) = if position_ratio < -0.3 && trend_strength > 0.3 {
        // Short in uptrend - want to buy back aggressively
        (1.5, 0.5)
    } else if position_ratio > 0.3 && trend_strength < -0.3 {
        // Long in downtrend - want to sell aggressively
        (0.5, 1.5)
    } else if trend_strength > 0.4 {
        // Uptrend - prefer buying
        (1.3, 0.7)
    } else if trend_strength < -0.4 {
        // Downtrend - prefer selling
        (0.7, 1.3)
    } else {
        (1.0, 1.0)
    };

    (skew_amount, size_multiplier, bid_size_adj, ask_size_adj)
}

fn should_place_order(
    state: &MarketMakerState,
    is_buy: bool,
    size: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) -> bool {
    let trend_strength = state.trend_detector.get_trend_strength();
    let position_ratio = state.position / max_position;

    // 1. Don't add to position when fighting strong trends
    if !is_buy && position_ratio < -0.4 && trend_strength > 0.4 {
        // Already short in strong uptrend - don't sell more
        return false;
    }
    
    if is_buy && position_ratio > 0.4 && trend_strength < -0.4 {
        // Already long in strong downtrend - don't buy more
        return false;
    }

    // 2. Near liquidation - only reduce position
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        if is_buy && state.position >= 0.0 {
            return false;
        }
        if !is_buy && state.position <= 0.0 {
            return false;
        }
    }

    // 3. Check margin availability
    let notional = state.mid_price * size;
    let required_margin = notional / state.leverage;
    let available_margin = state.get_available_margin();
    
    if required_margin > available_margin * 0.9 {
        return false;
    }

    // 4. Position limits
    if is_buy && state.position + size > max_position {
        return false;
    }
    if !is_buy && state.position - size < -max_position {
        return false;
    }

    true
}

fn should_update_orders(
    state: &MarketMakerState,
    last_mid: f64,
    max_position: f64,
    threshold_pct: f64,
    risk_limits: &RiskLimits,
) -> bool {
    if state.our_bid_order.is_none() || state.our_ask_order.is_none() {
        return true;
    }

    // Price movement
    if last_mid > 0.0 {
        let price_change_pct = ((state.mid_price - last_mid) / last_mid).abs();
        if price_change_pct > threshold_pct {
            return true;
        }
    }

    // Position management - more urgent when large
    if state.position.abs() > max_position * 0.5 {
        return true;
    }

    // High volatility
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.03 {
        return true;
    }

    // Trend changes - update more frequently in strong trends
    let trend_strength = state.trend_detector.get_trend_strength().abs();
    if trend_strength > 0.4 {
        return true;
    }

    // Near liquidation
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
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

fn place_adaptive_orders(
    state: &mut MarketMakerState,
    base_order_size: f64,
    base_spread_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) {
    let total_spread = calculate_adaptive_spread(state, base_spread_bps, max_position, risk_limits);
    let half_spread = total_spread / 2.0;

    let (skew_adjustment, size_multiplier, bid_size_adj, ask_size_adj) =
        calculate_dynamic_skew(state, max_position);

    let mut our_bid_price = state.mid_price - half_spread - skew_adjustment;
    let mut our_ask_price = state.mid_price + half_spread - skew_adjustment;

    let bid_size = (base_order_size * size_multiplier * bid_size_adj * 100.0).round() / 100.0;
    let ask_size = (base_order_size * size_multiplier * ask_size_adj * 100.0).round() / 100.0;

    our_bid_price = (our_bid_price * 10000.0).round() / 10000.0;
    our_ask_price = (our_ask_price * 10000.0).round() / 10000.0;

    let bid_edge = ((state.mid_price - our_bid_price) / state.mid_price) * 10000.0;
    let ask_edge = ((our_ask_price - state.mid_price) / state.mid_price) * 10000.0;

    let trend_strength = state.trend_detector.get_trend_strength();
    let regime_str = format!("{:?}", state.current_regime);

    // Check if we should place bid
    let can_place_bid = should_place_order(state, true, bid_size, max_position, risk_limits);
    
    if can_place_bid && bid_size >= 0.01 {
        let bid_order = state.place_simulated_order(true, our_bid_price, bid_size);
        info!(
            "📗 BID: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Trend: {:.2} [{}]",
            bid_size, our_bid_price, bid_order.oid, bid_edge, trend_strength, regime_str
        );
        state.our_bid_order = Some(bid_order);
    } else if !can_place_bid {
        info!("⚠️  Skipping bid - risk management");
    }

    // Check if we should place ask
    let can_place_ask = should_place_order(state, false, ask_size, max_position, risk_limits);
    
    if can_place_ask && ask_size >= 0.01 {
        let ask_order = state.place_simulated_order(false, our_ask_price, ask_size);
        info!(
            "📕 ASK: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Trend: {:.2} [{}]",
            ask_size, our_ask_price, ask_order.oid, ask_edge, trend_strength, regime_str
        );
        state.our_ask_order = Some(ask_order);
    } else if !can_place_ask {
        info!("⚠️  Skipping ask - risk management");
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // Configuration
    let coin = "HYPE".to_string();
    let user = address!("0xB249153BE6B73B431AB8Adc0c7c922Bb1d38A6B7");

    // Capital and Leverage
    let initial_capital = 500.0;
    let leverage = 10.0;

    // Market making parameters - more conservative
    let base_order_size = 1.0;  // Reduced from 3.0
    let base_spread_bps = 30.0; // Increased from 20.0
    let max_position = 30.0;    // Reduced from 50.0 - more conservative
    let update_threshold_pct = 0.0015; // 0.15% - slightly tighter
    let status_interval = 25;

    // Risk limits
    let risk_limits = RiskLimits {
        max_position_size: max_position,
        max_loss_per_day: 100.0, // $100 loss limit (20% of capital)
        max_drawdown_pct: 15.0,
        startup_grace_period_ms: 120000,
        max_margin_usage_pct: 80.0, // Don't use more than 80% of capital as margin
        liquidation_buffer_pct: 10.0,
    };

    // Initialize client
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Mainnet))
        .await
        .unwrap();
    let (sender, mut receiver) = unbounded_channel();
    let mut state = MarketMakerState::new(initial_capital, leverage);
    let mut update_counter = 0;
    let mut last_update_mid = 0.0;
    let mut trading_enabled = true;

    // Subscribe to data streams
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

    info!("🤖 IMPROVED MARKET MAKER Started for {}", coin);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("💰 CAPITAL MANAGEMENT");
    info!("   Initial Capital: ${:.2}", initial_capital);
    info!("   Leverage: {}x", leverage);
    info!("   Max Buying Power: ${:.2}", initial_capital * leverage);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("🧠 STRATEGY");
    info!("   Base Order Size: {} HYPE", base_order_size);
    info!("   Base Spread: {:.1} bps (adaptive)", base_spread_bps);
    info!("   Max Position: ±{} HYPE", max_position);
    info!("   Maker Fee: {:.3}%", state.maker_fee_rate * 100.0);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("⚠️  RISK LIMITS");
    info!("   Max Daily Loss: ${:.2}", risk_limits.max_loss_per_day);
    info!("   Max Drawdown: {:.1}%", risk_limits.max_drawdown_pct);
    info!("   Max Margin Usage: {:.1}%", risk_limits.max_margin_usage_pct);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("🎯 KEY IMPROVEMENTS");
    info!("   ✓ Proper leverage & margin tracking");
    info!("   ✓ Wider spreads to avoid adverse selection");
    info!("   ✓ Position-aware order placement");
    info!("   ✓ Don't fight strong trends");
    info!("   ✓ Liquidation monitoring");
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    while let Some(message) = receiver.recv().await {
        match message {
            Message::Bbo(bbo) => {
                if let (Some(bid), Some(ask)) = (&bbo.data.bbo[0], &bbo.data.bbo[1]) {
                    let bid_price = bid.px.parse::<f64>().unwrap();
                    let ask_price = ask.px.parse::<f64>().unwrap();
                    let bid_sz = bid.sz.parse::<f64>().unwrap();
                    let ask_sz = ask.sz.parse::<f64>().unwrap();

                    state.update_bbo(bid_price, ask_price, bid_sz, ask_sz);

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
                            max_position,
                            update_threshold_pct,
                            &risk_limits,
                        );

                        if should_update {
                            last_update_mid = state.mid_price;

                            cancel_simulated_orders(&mut state);

                            place_adaptive_orders(
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
use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::unbounded_channel;

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
    max_position: f64,
    max_loss_per_day: f64,
    max_drawdown_pct: f64,
    startup_grace_period_ms: u64,
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

        // Remove old samples outside window
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

#[derive(Debug)]
struct MarketMakerState {
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

    // Fees - CORRECTED: no rebates, only fees
    maker_fee_rate: f64,
    taker_fee_rate: f64,
    total_fees_paid: f64,

    // Capacity
    max_buy_size: f64,
    max_sell_size: f64,

    // Order ID generator
    next_oid: u64,

    // Volatility tracking
    volatility_tracker: VolatilityTracker,

    // Session tracking
    session_start_time: u64,
    peak_pnl: f64,
}

impl MarketMakerState {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
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
            maker_fee_rate: 0.00015,  // 0.015% - we PAY this
            taker_fee_rate: 0.00045,  // 0.045% - we PAY this
            total_fees_paid: 0.0,
            max_buy_size: 0.0,
            max_sell_size: 0.0,
            next_oid: 1,
            volatility_tracker: VolatilityTracker::new(300), // 5 minute window
            session_start_time: now,
            peak_pnl: 0.0,
        }
    }

    fn update_bbo(&mut self, bid: f64, ask: f64, bid_sz: f64, ask_sz: f64) {
        self.current_bid = bid;
        self.current_ask = ask;
        self.mid_price = (bid + ask) / 2.0;
        self.spread = ask - bid;
        self.bid_depth = bid_sz;
        self.ask_depth = ask_sz;

        // Track volatility
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.volatility_tracker.add_sample(timestamp, self.mid_price);

        // Calculate unrealized PnL
        if self.position != 0.0 {
            let mark_price = self.mid_price;
            self.unrealized_pnl = (mark_price - self.average_entry) * self.position;
        }

        // Track peak PnL for drawdown calculation
        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl > self.peak_pnl {
            self.peak_pnl = total_pnl;
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

    fn check_fills(&mut self, trade_price: f64) -> Vec<String> {
        let mut fills = Vec::new();

        // Check if our bid order would be filled
        if let Some(bid_order) = &self.our_bid_order {
            if trade_price <= bid_order.price {
                fills.push(self.execute_fill(bid_order.clone(), trade_price));
                self.our_bid_order = None;
            }
        }

        // Check if our ask order would be filled
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

        // CORRECTED: Calculate fees (we PAY maker fees, no rebates)
        let notional = fill_price * order.size;
        let fee = notional * self.maker_fee_rate;
        self.total_fees_paid += fee;

        let fill_msg = if order.is_buy {
            self.buy_fills += 1;

            // Update position
            let old_position = self.position;
            let old_avg_entry = self.average_entry;

            self.position += order.size;

            // Update average entry price and track P&L
            if old_position < 0.0 && self.position >= 0.0 {
                // Closing a short or flipping to long
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
                // Adding to long
                self.average_entry =
                    (old_avg_entry * old_position + fill_price * order.size) / self.position;
            }

            format!(
                "✅ BUY FILL: {:.2} @ ${:.4} (OID: {}) | Fee: ${:.4}",
                order.size, fill_price, order.oid, fee
            )
        } else {
            self.sell_fills += 1;

            // Update position
            let old_position = self.position;
            let old_avg_entry = self.average_entry;

            self.position -= order.size;

            // Update average entry price and realized PnL
            if old_position > 0.0 && self.position <= 0.0 {
                // Closing a long or flipping to short
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
                // Adding to short
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
        
        // If peak is very small, return absolute loss instead of percentage
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

        // Check if we're still in grace period
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let time_elapsed = now - self.session_start_time;
        let in_grace_period = time_elapsed < limits.startup_grace_period_ms;

        // Check position limit (always enforced)
        if self.position.abs() > limits.max_position {
            violations.push(format!(
                "Position limit breached: {:.2} > {:.2}",
                self.position.abs(),
                limits.max_position
            ));
            is_safe = false;
        }

        // Check daily loss limit (always enforced)
        let total_pnl = self.realized_pnl + self.unrealized_pnl - self.total_fees_paid;
        if total_pnl < -limits.max_loss_per_day {
            violations.push(format!(
                "Daily loss limit breached: ${:.2} < -${:.2}",
                total_pnl, limits.max_loss_per_day
            ));
            is_safe = false;
        }

        // Check drawdown (skip during grace period)
        if !in_grace_period {
            let drawdown_value = self.get_current_drawdown_pct();
            
            if self.peak_pnl >= 5.0 {
                // Percentage drawdown
                if drawdown_value > limits.max_drawdown_pct {
                    violations.push(format!(
                        "Drawdown limit breached: {:.2}% > {:.2}%",
                        drawdown_value, limits.max_drawdown_pct
                    ));
                    is_safe = false;
                }
            } else {
                // Absolute loss (when peak is small)
                if drawdown_value > limits.max_loss_per_day / 2.0 {
                    violations.push(format!(
                        "Early-stage loss limit: ${:.2} > ${:.2}",
                        drawdown_value, limits.max_loss_per_day / 2.0
                    ));
                    is_safe = false;
                }
            }
        } else if !violations.is_empty() {
            info!("⏱️  In grace period ({:.1}s remaining) - drawdown check suspended",
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

        let win_rate = if self.profitable_fills + self.losing_fills > 0 {
            (self.profitable_fills as f64
                / (self.profitable_fills + self.losing_fills) as f64)
                * 100.0
        } else {
            0.0
        };

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 MARKET MAKER STATUS");
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

fn calculate_optimal_spread(
    state: &MarketMakerState,
    base_spread_bps: f64,
    max_position: f64,
) -> f64 {
    let mut spread_multiplier = 1.0;

    // 1. Adjust for volatility
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.02 {
        spread_multiplier *= 1.0 + (volatility - 0.02) * 20.0;
    }

    // 2. Adjust for order book imbalance
    let imbalance = state.get_book_imbalance();
    spread_multiplier *= 1.0 + imbalance.abs() * 0.3;

    // 3. Widen spread when carrying large position
    let position_ratio = state.position.abs() / max_position;
    spread_multiplier *= 1.0 + position_ratio * 0.5;

    // 4. CRITICAL: Minimum spread must cover round-trip fees
    // Round trip = buy (0.015%) + sell (0.015%) = 0.03% = 3 bps minimum
    let min_spread_bps = state.maker_fee_rate * 2.0 * 10000.0;
    let target_spread_bps = (base_spread_bps * spread_multiplier).max(min_spread_bps);

    (state.mid_price * target_spread_bps) / 10000.0
}

fn calculate_inventory_skew(position: f64, max_position: f64, mid_price: f64) -> (f64, f64, f64) {
    let position_ratio = position / max_position;

    // Exponential skew
    let skew_factor = position_ratio.powi(3);
    let skew_bps = skew_factor * 50.0;
    let skew_amount = mid_price * skew_bps / 10000.0;

    // Reduce order size as position grows
    let size_multiplier = 1.0 - position_ratio.abs() * 0.5;

    (skew_amount, size_multiplier, position_ratio)
}

fn should_update_orders(
    state: &MarketMakerState,
    last_mid: f64,
    max_position: f64,
    threshold_pct: f64,
) -> bool {
    if state.our_bid_order.is_none() || state.our_ask_order.is_none() {
        return true;
    }

    if last_mid > 0.0 {
        let price_change_pct = ((state.mid_price - last_mid) / last_mid).abs();
        if price_change_pct > threshold_pct {
            return true;
        }
    }

    if state.position.abs() > max_position * 0.6 {
        return true;
    }

    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.03 {
        return true;
    }

    false
}

fn cancel_simulated_orders(state: &mut MarketMakerState) {
    if let Some(bid) = &state.our_bid_order {
        info!(
            "❌ Cancelled bid order {} @ ${:.4}",
            bid.oid, bid.price
        );
        state.our_bid_order = None;
    }

    if let Some(ask) = &state.our_ask_order {
        info!(
            "❌ Cancelled ask order {} @ ${:.4}",
            ask.oid, ask.price
        );
        state.our_ask_order = None;
    }
}

fn place_simulated_orders(
    state: &mut MarketMakerState,
    base_order_size: f64,
    base_spread_bps: f64,
    max_position: f64,
) {
    let total_spread = calculate_optimal_spread(state, base_spread_bps, max_position);
    let half_spread = total_spread / 2.0;

    let (skew_adjustment, size_multiplier, position_ratio) =
        calculate_inventory_skew(state.position, max_position, state.mid_price);

    let mut our_bid_price = state.mid_price - half_spread - skew_adjustment;
    let mut our_ask_price = state.mid_price + half_spread - skew_adjustment;

    let bid_size = (base_order_size * size_multiplier * 100.0).round() / 100.0;
    let ask_size = (base_order_size * size_multiplier * 100.0).round() / 100.0;

    our_bid_price = (our_bid_price * 10000.0).round() / 10000.0;
    our_ask_price = (our_ask_price * 10000.0).round() / 10000.0;

    let can_buy = state.position < max_position;
    let can_sell = state.position > -max_position;

    let bid_edge = ((state.mid_price - our_bid_price) / state.mid_price) * 10000.0;
    let ask_edge = ((our_ask_price - state.mid_price) / state.mid_price) * 10000.0;

    if can_buy && bid_size > 0.01 {
        let bid_order = state.place_simulated_order(true, our_bid_price, bid_size);
        info!(
            "📗 BID: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Skew: {:.2}%",
            bid_size,
            our_bid_price,
            bid_order.oid,
            bid_edge,
            position_ratio * 100.0
        );
        state.our_bid_order = Some(bid_order);
    } else if !can_buy {
        info!(
            "⚠️  Cannot place bid - position limit ({:.2}/{:.2})",
            state.position, max_position
        );
    }

    if can_sell && ask_size > 0.01 {
        let ask_order = state.place_simulated_order(false, our_ask_price, ask_size);
        info!(
            "📕 ASK: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Skew: {:.2}%",
            ask_size,
            our_ask_price,
            ask_order.oid,
            ask_edge,
            position_ratio * 100.0
        );
        state.our_ask_order = Some(ask_order);
    } else if !can_sell {
        info!(
            "⚠️  Cannot place ask - position limit ({:.2}/{:.2})",
            state.position, -max_position
        );
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    // Configuration
    let coin = "HYPE".to_string();
    let user = address!("0xB249153BE6B73B431AB8Adc0c7c922Bb1d38A6B7");

    // Market making parameters
    let base_order_size = 3.0; // 3 HYPE per order
    let base_spread_bps = 25.0; // 25 bps base spread
    let max_position = 50.0; // Max position in HYPE
    let update_threshold_pct = 0.003; // 0.3% price move
    let status_interval = 30; // Print status every 30 updates

    // Risk limits
    let risk_limits = RiskLimits {
        max_position,
        max_loss_per_day: 50.0,
        max_drawdown_pct: 15.0,
        startup_grace_period_ms: 120000, // 2 minutes
    };

    // Initialize client
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Mainnet))
        .await
        .unwrap();
    let (sender, mut receiver) = unbounded_channel();
    let mut state = MarketMakerState::new();
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

    info!("🤖 Advanced Market Maker Started for {}", coin);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("📈 Base Order Size: {} HYPE", base_order_size);
    info!("📊 Base Spread: {:.1} bps", base_spread_bps);
    info!("📦 Max Position: ±{} HYPE", max_position);
    info!("💸 Maker Fee: {:.3}%", state.maker_fee_rate * 100.0);
    info!("💸 Taker Fee: {:.3}%", state.taker_fee_rate * 100.0);
    info!("⚠️  Min Profitable Spread: {:.1} bps (covers round-trip fees)", 
        state.maker_fee_rate * 2.0 * 10000.0);
    info!("⚠️  Max Daily Loss: ${:.2}", risk_limits.max_loss_per_day);
    info!("📉 Max Drawdown: {:.1}%", risk_limits.max_drawdown_pct);
    info!("⏱️  Grace Period: {:.0}s", risk_limits.startup_grace_period_ms as f64 / 1000.0);
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
                        );

                        if should_update {
                            last_update_mid = state.mid_price;

                            cancel_simulated_orders(&mut state);

                            place_simulated_orders(
                                &mut state,
                                base_order_size,
                                base_spread_bps,
                                max_position,
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

                    let fills = state.check_fills(trade_price);
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

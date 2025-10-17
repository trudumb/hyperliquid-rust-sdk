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

        if volatility > 0.04 {
            return MarketRegime::Volatile;
        }

        if trend_strength > 0.5 || z_score > 2.5 {
            return MarketRegime::Trending;
        }

        if volatility < 0.015 && trend_strength < 0.25 {
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

    fn get_max_position_size(&self, price: f64, max_position_limit: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }
        
        let available_margin = self.get_available_margin();
        let max_notional_from_margin = available_margin * self.leverage;
        let max_size_from_margin = max_notional_from_margin / price;
        
        max_size_from_margin.min(max_position_limit)
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

        // Calculate fee-to-profit ratio
        let fee_ratio = if self.realized_pnl.abs() > 0.01 {
            (self.total_fees_paid / self.realized_pnl.abs()) * 100.0
        } else {
            0.0
        };

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 PROFITABLE MM STATUS");
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
        info!("Fees Paid: ${:.2} (Fee Ratio: {:.1}%)", self.total_fees_paid, fee_ratio);
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

// NEW: Calculate minimum profitable spread
fn calculate_minimum_spread(state: &MarketMakerState) -> f64 {
    // Minimum spread = 2x maker fees + safety margin
    let fee_coverage = state.maker_fee_rate * 2.0 * 10000.0;  // ~3 bps
    let adverse_selection_buffer = 15.0;  // 15 bps buffer for toxic flow
    let min_profit_target = 5.0;  // 5 bps minimum profit
    
    fee_coverage + adverse_selection_buffer + min_profit_target  // ~23 bps minimum
}

// NEW: Detect toxic/adverse flow
fn detect_toxic_flow(state: &MarketMakerState) -> bool {
    let imbalance = state.trend_detector.get_flow_imbalance().abs();
    
    // Toxic if:
    // 1. Heavy one-sided flow (>70% imbalance)
    // 2. High volatility
    // 3. Strong trend
    imbalance > 0.7 || 
    state.volatility_tracker.get_volatility() > 0.02 ||
    state.trend_detector.get_trend_strength().abs() > 0.6
}

// NEW: Check if we should stop trading based on performance
fn should_stop_trading(state: &MarketMakerState) -> bool {
    let net_pnl = state.realized_pnl + state.unrealized_pnl - state.total_fees_paid;
    
    // Stop if:
    // 1. Daily loss limit hit
    if net_pnl < -100.0 { 
        info!("🛑 Stopping: Daily loss limit hit (${:.2})", net_pnl);
        return true; 
    }
    
    // 2. Negative PnL with high fee ratio
    if state.total_fills > 20 && 
       state.total_fees_paid > state.realized_pnl.abs() * 0.5 {
        info!("🛑 Stopping: Fees eating >50% of gross profit");
        return true;
    }
    
    // 3. Win rate too low
    if state.total_fills > 10 {
        let win_rate = state.profitable_fills as f64 / state.total_fills as f64;
        if win_rate < 0.45 { 
            info!("🛑 Stopping: Win rate too low ({:.1}%)", win_rate * 100.0);
            return true; 
        }
    }
    
    false
}

// IMPROVED: Conservative spread calculation for profitability
fn calculate_adaptive_spread(
    state: &MarketMakerState,
    base_spread_bps: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) -> f64 {
    let mut spread_bps = base_spread_bps;
    
    // 1. Minimum profitable spread
    let min_spread = calculate_minimum_spread(state);
    spread_bps = spread_bps.max(min_spread);
    
    // 2. Volatility adjustment - be more conservative
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.01 {  // Lower threshold
        spread_bps *= 1.0 + volatility * 50.0;  // Stronger adjustment
    }
    
    // 3. Inventory risk - widen spread as position grows
    let position_ratio = (state.position.abs() / max_position).min(1.0);
    spread_bps *= 1.0 + position_ratio.powi(2) * 0.5;
    
    // 4. Adverse selection detection
    let flow_imbalance = state.trend_detector.get_flow_imbalance().abs();
    if flow_imbalance > 0.5 {
        spread_bps *= 1.5;  // Widen significantly on toxic flow
    }
    
    // 5. Near liquidation - widen significantly
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        spread_bps *= 2.0;
    }

    // 6. Strong trend detection - avoid adverse selection
    let trend_strength = state.trend_detector.get_trend_strength().abs();
    if trend_strength > 0.5 {
        spread_bps *= 1.3;
    }

    (state.mid_price * spread_bps) / 10000.0
}

// IMPROVED: More conservative inventory management
fn calculate_dynamic_skew(
    state: &MarketMakerState,
    max_position: f64,
) -> (f64, f64, f64, f64) {
    let position_ratio = state.position / max_position;
    let trend_strength = state.trend_detector.get_trend_strength();
    let momentum_z = state.trend_detector.get_momentum_z_score();

    // 1. Inventory management - moderate urgency
    let inventory_urgency = if position_ratio.abs() > 0.5 {
        // Urgent when position is large (reduced from 0.7)
        position_ratio.powi(2) * 100.0
    } else if position_ratio < -0.3 && trend_strength > 0.4 {
        // Short in uptrend - cover
        -80.0
    } else if position_ratio > 0.3 && trend_strength < -0.4 {
        // Long in downtrend - sell
        80.0
    } else {
        position_ratio.powi(2) * position_ratio.signum() * 40.0
    };

    // 2. Trend-based skew - only when not fighting position
    let trend_skew_bps = if position_ratio * trend_strength < -0.2 {
        0.0  // Don't add trend bias when opposite to position
    } else if position_ratio.abs() < 0.3 {
        trend_strength * 12.0  // Moderate trend following
    } else {
        trend_strength * 6.0  // Reduced when positioned
    };

    // 3. Mean reversion - fade extremes
    let mean_revert_skew = match state.current_regime {
        MarketRegime::MeanReverting => {
            if momentum_z.abs() > 2.0 {
                -momentum_z.signum() * 10.0
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    let total_skew_bps = inventory_urgency + trend_skew_bps + mean_revert_skew;
    let skew_amount = state.mid_price * total_skew_bps / 10000.0;

    // 4. Size adjustment
    let mut size_multiplier = 1.0 - position_ratio.abs() * 0.5;
    
    match state.current_regime {
        MarketRegime::Volatile => {
            size_multiplier *= 0.7;
        }
        MarketRegime::Calm => {
            size_multiplier *= 1.1;
        }
        MarketRegime::Trending => {
            size_multiplier *= 0.85;
        }
        _ => {}
    }

    // 5. Asymmetric sizing
    let (bid_size_adj, ask_size_adj) = if position_ratio < -0.4 && trend_strength > 0.4 {
        (1.2, 0.8)
    } else if position_ratio > 0.4 && trend_strength < -0.4 {
        (0.8, 1.2)
    } else if trend_strength > 0.4 {
        (1.1, 0.9)
    } else if trend_strength < -0.4 {
        (0.9, 1.1)
    } else {
        (1.0, 1.0)
    };

    (skew_amount, size_multiplier, bid_size_adj, ask_size_adj)
}

// IMPROVED: More conservative order placement
fn should_place_order(
    state: &MarketMakerState,
    is_buy: bool,
    size: f64,
    max_position: f64,
    risk_limits: &RiskLimits,
) -> bool {
    let trend_strength = state.trend_detector.get_trend_strength();
    let position_ratio = state.position / max_position;

    // 1. Avoid toxic flow completely
    if detect_toxic_flow(state) {
        return false;
    }

    // 2. Don't fight strong trends with large position
    if !is_buy && position_ratio < -0.5 && trend_strength > 0.5 {
        return false;
    }
    
    if is_buy && position_ratio > 0.5 && trend_strength < -0.5 {
        return false;
    }

    // 3. Near liquidation - only reduce position
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        if is_buy && state.position >= 0.0 {
            return false;
        }
        if !is_buy && state.position <= 0.0 {
            return false;
        }
    }

    // 4. Check margin availability
    let notional = state.mid_price * size;
    let required_margin = notional / state.leverage;
    let available_margin = state.get_available_margin();
    
    if required_margin > available_margin * 0.95 {
        return false;
    }

    // 5. Position limits
    if is_buy && state.position + size > max_position {
        return false;
    }
    if !is_buy && state.position - size < -max_position {
        return false;
    }

    true
}

// IMPROVED: Update based on price movement and conditions
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

    // Large position - update to manage inventory
    if state.position.abs() > max_position * 0.5 {
        return true;
    }

    // High volatility
    let volatility = state.volatility_tracker.get_volatility();
    if volatility > 0.03 {
        return true;
    }

    // Near liquidation
    if state.is_near_liquidation(risk_limits.liquidation_buffer_pct) {
        return true;
    }

    // Toxic flow detected
    if detect_toxic_flow(state) {
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

    let can_place_bid = should_place_order(state, true, bid_size, max_position, risk_limits);
    
    if can_place_bid && bid_size >= 0.01 {
        let bid_order = state.place_simulated_order(true, our_bid_price, bid_size);
        info!(
            "📗 BID: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Trend: {:.2} [{}]",
            bid_size, our_bid_price, bid_order.oid, bid_edge, trend_strength, regime_str
        );
        state.our_bid_order = Some(bid_order);
    } else if !can_place_bid {
        info!("⚠️  Skipping bid - risk/flow management");
    }

    let can_place_ask = should_place_order(state, false, ask_size, max_position, risk_limits);
    
    if can_place_ask && ask_size >= 0.01 {
        let ask_order = state.place_simulated_order(false, our_ask_price, ask_size);
        info!(
            "📕 ASK: {:.2} @ ${:.4} (OID: {}) | Edge: {:.1} bps | Trend: {:.2} [{}]",
            ask_size, our_ask_price, ask_order.oid, ask_edge, trend_strength, regime_str
        );
        state.our_ask_order = Some(ask_order);
    } else if !can_place_ask {
        info!("⚠️  Skipping ask - risk/flow management");
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

    // PROFITABLE CONFIGURATION - Conservative parameters
    let base_order_size = 3.0;     // Reduced from 15.0
    let base_spread_bps = 25.0;    // Increased from 10.0
    let max_position = 30.0;       // Reduced from 100.0
    let update_threshold_pct = 0.002; // Increased from 0.001
    let status_interval = 5;

    // Risk limits
    let risk_limits = RiskLimits {
        max_position_size: max_position,
        max_loss_per_day: 100.0,    // Reduced from 150.0
        max_drawdown_pct: 15.0,     // Reduced from 20.0
        startup_grace_period_ms: 120000,
        max_margin_usage_pct: 75.0, // Reduced from 85.0
        liquidation_buffer_pct: 15.0, // Increased from 10.0
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

    info!("🤖 PROFITABLE MARKET MAKER Started for {}", coin);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("💰 CAPITAL MANAGEMENT");
    info!("   Initial Capital: ${:.2}", initial_capital);
    info!("   Leverage: {}x", leverage);
    info!("   Max Buying Power: ${:.2}", initial_capital * leverage);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("🎯 PROFITABLE STRATEGY");
    info!("   Base Order Size: {} HYPE (CONSERVATIVE)", base_order_size);
    info!("   Base Spread: {:.1} bps (PROFITABLE)", base_spread_bps);
    info!("   Min Spread: ~23 bps (Fee coverage + buffer)", );
    info!("   Max Position: ±{} HYPE (MANAGED)", max_position);
    info!("   Maker Fee: {:.3}%", state.maker_fee_rate * 100.0);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    info!("✨ KEY IMPROVEMENTS FOR PROFITABILITY");
    info!("   ✓ Wider spreads (25 bps vs 10 bps)");
    info!("   ✓ Smaller sizes (3 vs 15)");
    info!("   ✓ Lower position limits (30 vs 100)");
    info!("   ✓ Toxic flow detection and avoidance");
    info!("   ✓ Minimum profitable spread enforcement");
    info!("   ✓ Better inventory risk management");
    info!("   ✓ Performance-based trading stops");
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

                    // Check if we should stop trading based on performance
                    if should_stop_trading(&state) {
                        if trading_enabled {
                            info!("🛑 STOPPING TRADING DUE TO PERFORMANCE LIMITS 🛑");
                            cancel_simulated_orders(&mut state);
                            trading_enabled = false;
                            state.print_status();
                        }
                        continue;
                    }

                    // Check risk limits
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
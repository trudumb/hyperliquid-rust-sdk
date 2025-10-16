use alloy::primitives::address;
use hyperliquid_rust_sdk::{BaseUrl, InfoClient, Message, Subscription};
use log::info;
use tokio::sync::mpsc::unbounded_channel;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
struct SimulatedOrder {
    oid: u64,
    price: f64,
    size: f64,
    is_buy: bool,
    timestamp: u64,
}

#[derive(Debug)]
struct MarketMakerState {
    // Market data
    current_bid: f64,
    current_ask: f64,
    mid_price: f64,
    spread: f64,
    
    // Our simulated orders
    our_bid_order: Option<SimulatedOrder>,
    our_ask_order: Option<SimulatedOrder>,
    
    // Position tracking
    position: f64, // Positive = long, negative = short
    average_entry: f64,
    
    // Performance tracking
    realized_pnl: f64,
    unrealized_pnl: f64,
    total_fills: u32,
    buy_fills: u32,
    sell_fills: u32,
    
    // Capacity
    max_buy_size: f64,
    max_sell_size: f64,
    
    // Order ID generator
    next_oid: u64,
}

impl MarketMakerState {
    fn new() -> Self {
        Self {
            current_bid: 0.0,
            current_ask: 0.0,
            mid_price: 0.0,
            spread: 0.0,
            our_bid_order: None,
            our_ask_order: None,
            position: 0.0,
            average_entry: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            total_fills: 0,
            buy_fills: 0,
            sell_fills: 0,
            max_buy_size: 0.0,
            max_sell_size: 0.0,
            next_oid: 1,
        }
    }

    fn update_bbo(&mut self, bid: f64, ask: f64) {
        self.current_bid = bid;
        self.current_ask = ask;
        self.mid_price = (bid + ask) / 2.0;
        self.spread = ask - bid;
        
        // Calculate unrealized PnL
        if self.position != 0.0 {
            let mark_price = self.mid_price;
            self.unrealized_pnl = (mark_price - self.average_entry) * self.position;
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

    fn check_fills(&mut self) -> Vec<String> {
        let mut fills = Vec::new();
        
        // Check if our bid order would be filled
        if let Some(bid_order) = &self.our_bid_order {
            // Our bid gets filled if market trades at or below our bid price
            if self.current_bid >= bid_order.price {
                fills.push(self.execute_fill(bid_order.clone()));
                self.our_bid_order = None;
            }
        }
        
        // Check if our ask order would be filled
        if let Some(ask_order) = &self.our_ask_order {
            // Our ask gets filled if market trades at or above our ask price
            if self.current_ask <= ask_order.price {
                fills.push(self.execute_fill(ask_order.clone()));
                self.our_ask_order = None;
            }
        }
        
        fills
    }

    fn execute_fill(&mut self, order: SimulatedOrder) -> String {
        self.total_fills += 1;
        
        let fill_msg = if order.is_buy {
            self.buy_fills += 1;
            
            // Update position
            let old_position = self.position;
            let old_avg_entry = self.average_entry;
            
            self.position += order.size;
            
            // Update average entry price
            if old_position < 0.0 && self.position >= 0.0 {
                // Closing a short or flipping to long
                let closed_size = old_position.abs().min(order.size);
                self.realized_pnl += (old_avg_entry - order.price) * closed_size;
                
                if self.position > 0.0 {
                    self.average_entry = order.price;
                }
            } else if old_position >= 0.0 {
                // Adding to long
                self.average_entry = (old_avg_entry * old_position + order.price * order.size) 
                    / self.position;
            }
            
            format!("✅ BUY FILL: {:.2} @ ${:.3} (OID: {})", order.size, order.price, order.oid)
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
                self.realized_pnl += (order.price - old_avg_entry) * closed_size;
                
                if self.position < 0.0 {
                    self.average_entry = order.price;
                }
            } else if old_position <= 0.0 {
                // Adding to short
                self.average_entry = (old_avg_entry * old_position.abs() + order.price * order.size) 
                    / self.position.abs();
            }
            
            format!("✅ SELL FILL: {:.2} @ ${:.3} (OID: {})", order.size, order.price, order.oid)
        };
        
        fill_msg
    }

    fn print_status(&self) {
        let total_pnl = self.realized_pnl + self.unrealized_pnl;
        
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("📊 MARKET MAKER STATUS");
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("Market: Bid ${:.3} / Ask ${:.3} (Spread: ${:.3})", 
            self.current_bid, self.current_ask, self.spread);
        
        if let Some(bid) = &self.our_bid_order {
            info!("Our Bid: {:.2} @ ${:.3} (OID: {})", bid.size, bid.price, bid.oid);
        } else {
            info!("Our Bid: None");
        }
        
        if let Some(ask) = &self.our_ask_order {
            info!("Our Ask: {:.2} @ ${:.3} (OID: {})", ask.size, ask.price, ask.oid);
        } else {
            info!("Our Ask: None");
        }
        
        info!("Position: {:.2} HYPE @ ${:.3} avg", self.position, self.average_entry);
        info!("Realized PnL: ${:.2}", self.realized_pnl);
        info!("Unrealized PnL: ${:.2}", self.unrealized_pnl);
        info!("Total PnL: ${:.2}", total_pnl);
        info!("Fills: {} total ({} buys, {} sells)", 
            self.total_fills, self.buy_fills, self.sell_fills);
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    // Configuration
    let coin = "HYPE".to_string();
    let user = address!("0xB249153BE6B73B431AB8Adc0c7c922Bb1d38A6B7");
    
    // Market making parameters
    let order_size = 10.0; // HYPE tokens per order
    let target_spread_bps = 10; // 0.1% (10 basis points) - our spread target
    let max_position = 50.0; // Max inventory in HYPE
    let update_threshold_pct = 0.02; // 2% - cancel/replace if price moves this much
    let status_interval = 20; // Print status every 20 updates

    // Initialize client
    let mut info_client = InfoClient::new(None, Some(BaseUrl::Mainnet)).await.unwrap();
    let (sender, mut receiver) = unbounded_channel();
    let mut state = MarketMakerState::new();
    let mut update_counter = 0;
    let mut last_update_mid = 0.0;

    // Subscribe to data streams
    info_client.subscribe(
        Subscription::ActiveAssetData { user, coin: coin.clone() },
        sender.clone(),
    ).await.unwrap();

    info_client.subscribe(
        Subscription::Bbo { coin: coin.clone() },
        sender.clone(),
    ).await.unwrap();

    info!("🤖 SIMULATED Market Maker Started for {}", coin);
    info!("📈 Order Size: {} HYPE", order_size);
    info!("📊 Target Spread: {} bps", target_spread_bps);
    info!("📦 Max Position: ±{} HYPE", max_position);
    info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    while let Some(message) = receiver.recv().await {
        match message {
            Message::Bbo(bbo) => {
                if let (Some(bid), Some(ask)) = (&bbo.data.bbo[0], &bbo.data.bbo[1]) {
                    let bid_price = bid.px.parse::<f64>().unwrap();
                    let ask_price = ask.px.parse::<f64>().unwrap();
                    
                    state.update_bbo(bid_price, ask_price);
                    
                    // Check for fills on existing orders
                    let fills = state.check_fills();
                    for fill_msg in fills {
                        info!("{}", fill_msg);
                        info!("💰 Position: {:.2} HYPE | Realized PnL: ${:.2} | Unrealized PnL: ${:.2}", 
                            state.position, state.realized_pnl, state.unrealized_pnl);
                    }
                    
                    // Decide if we need to update orders
                    let should_update = should_update_orders(
                        &state, 
                        last_update_mid, 
                        max_position,
                        update_threshold_pct
                    );
                    
                    if should_update {
                        last_update_mid = state.mid_price;
                        
                        // Cancel existing orders
                        cancel_simulated_orders(&mut state);
                        
                        // Place new orders
                        place_simulated_orders(
                            &mut state, 
                            order_size, 
                            target_spread_bps,
                            max_position
                        );
                        
                        update_counter += 1;
                        if update_counter % status_interval == 0 {
                            state.print_status();
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

fn should_update_orders(
    state: &MarketMakerState,
    last_mid: f64,
    max_position: f64,
    threshold_pct: f64,
) -> bool {
    // Always place orders if we don't have any
    if state.our_bid_order.is_none() || state.our_ask_order.is_none() {
        return true;
    }
    
    // Update if market moved significantly
    if last_mid > 0.0 {
        let price_change_pct = ((state.mid_price - last_mid) / last_mid).abs();
        if price_change_pct > threshold_pct {
            return true;
        }
    }
    
    // Update if position is too large (inventory risk)
    if state.position.abs() > max_position * 0.8 {
        return true;
    }
    
    false
}

fn cancel_simulated_orders(state: &mut MarketMakerState) {
    if let Some(bid) = &state.our_bid_order {
        info!("❌ Cancelled bid order {} @ ${:.3}", bid.oid, bid.price);
        state.our_bid_order = None;
    }
    
    if let Some(ask) = &state.our_ask_order {
        info!("❌ Cancelled ask order {} @ ${:.3}", ask.oid, ask.price);
        state.our_ask_order = None;
    }
}

fn place_simulated_orders(
    state: &mut MarketMakerState,
    order_size: f64,
    target_spread_bps: u16,
    max_position: f64,
) {
    // Calculate half spread in dollars
    let half_spread = (state.mid_price * target_spread_bps as f64) / 20000.0;
    
    // Inventory skew - adjust prices based on position
    let position_ratio = state.position / max_position;
    let skew_adjustment = position_ratio * (state.mid_price * 0.001); // 0.1% skew per 100% position
    
    let mut our_bid_price = state.mid_price - half_spread - skew_adjustment;
    let mut our_ask_price = state.mid_price + half_spread - skew_adjustment;
    
    // Round to 3 decimals
    our_bid_price = (our_bid_price * 1000.0).round() / 1000.0;
    our_ask_price = (our_ask_price * 1000.0).round() / 1000.0;
    
    // Check position limits
    let can_buy = state.position < max_position;
    let can_sell = state.position > -max_position;
    
    // Place BUY order (if we're not too long)
    if can_buy {
        let bid_order = state.place_simulated_order(true, our_bid_price, order_size);
        info!("📗 BUY order placed: {:.2} @ ${:.3} (OID: {}) [Mid: ${:.3}]", 
            order_size, our_bid_price, bid_order.oid, state.mid_price);
        state.our_bid_order = Some(bid_order);
    } else {
        info!("⚠️ Cannot place buy order - position limit reached ({:.2}/{:.2})", 
            state.position, max_position);
    }
    
    // Place SELL order (if we're not too short)
    if can_sell {
        let ask_order = state.place_simulated_order(false, our_ask_price, order_size);
        info!("📕 SELL order placed: {:.2} @ ${:.3} (OID: {}) [Mid: ${:.3}]", 
            order_size, our_ask_price, ask_order.oid, state.mid_price);
        state.our_ask_order = Some(ask_order);
    } else {
        info!("⚠️ Cannot place sell order - position limit reached ({:.2}/{:.2})", 
            state.position, -max_position);
    }
}

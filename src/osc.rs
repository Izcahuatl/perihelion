use rosc::{OscPacket, OscType};
use std::net::UdpSocket;
use std::sync::mpsc::Sender;
use eframe::egui;

fn extract_perihelion_value(packet: &rosc::OscPacket) -> Option<bool> {
    match packet {
        rosc::OscPacket::Message(msg) => {
            if msg.addr == "/avatar/parameters/perihelion" {
                return match &msg.args[..] {
                    [rosc::OscType::Float(v)] => Some(*v > 0.5),
                    [rosc::OscType::Bool(v)] => Some(*v),
                    [rosc::OscType::Int(v)] => Some(*v > 0),
                    _ => Some(false),
                };
            }
            None
        }
        rosc::OscPacket::Bundle(bundle) => {
            for packet in &bundle.content {
                if let Some(v) = extract_perihelion_value(packet) {
                    return Some(v);
                }
            }
            None
        }
    }
}

pub fn run_osc_listener(tx: Sender<bool>, ctx: egui::Context) {
    let socket = match UdpSocket::bind("0.0.0.0:9001") {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut buf = [0u8; rosc::decoder::MTU];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((size, _)) => {
                if let Ok((_, packet)) = rosc::decoder::decode_udp(&buf[..size]) {
                    if let Some(on) = extract_perihelion_value(&packet) {
                        if tx.send(on).is_err() {
                            break;
                        }
                        ctx.request_repaint();
                    }
                }
            }
            Err(_) => break,
        }
    }
}

